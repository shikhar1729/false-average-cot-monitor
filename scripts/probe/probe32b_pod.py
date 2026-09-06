"""Runs ON a GPU pod (OUT_DIR). For each row of /workspace/probe32b_inputs.jsonl: last-token hidden states at the
8 probe layers (same layers/dtype/tokenization as pod_probe.py) and, per condition in row["conds"],
a greedy verdict generation with an optional additive intervention along the probe direction
(/workspace/probe32b_dirs.pt: {layer: {"u": unit vector, "delta": legit->obf mean gap}}).
  cond {"name","layer","k","mode"}: add k*delta*u to layer `layer`'s output; mode "last" = the last
  prompt token and every generated token, mode "all" = every position. "base" = no hook.
  conds == [] : features only, no generation.
Features are read by forward hooks (not output_hidden_states, which does not fit for 32B at 16k tokens),
at the true last prompt token and, when the prompt is longer than CAP2 tokens, also at position CAP2-1
(= the last token the paper's 32B probe run saw, which capped inputs at 6144; causal attention makes the
two identical to a truncated pass). RUN_SECONDS (env): stop taking new rows after this many seconds.
Writes /workspace/probe32b_out.jsonl incrementally (one line per row x condition) and /workspace/probe32b_feats.pt.
"""
import json, os, re, time, numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

T0 = time.time()
MODEL = os.environ.get("PROBE_MODEL", "Qwen/Qwen3-8B")
MAXTOK = 16384
CAP2 = int(os.environ.get("CAP2", "0"))                  # 0 = off (8B run); 6144 for the 32B run
RUN_SECONDS = float(os.environ.get("RUN_SECONDS", "1e9"))
SCORE_RE = re.compile(r'"suspicion_score"\s*:\s*([01](?:\.\d+)?|0?\.\d+)')
rows = [json.loads(l) for l in open("/workspace/probe32b_inputs.jsonl") if l.strip()]
dirs = torch.load("/workspace/probe32b_dirs.pt", map_location="cpu", weights_only=False)
print(f"{len(rows)} rows, {sum(len(r['conds']) for r in rows)} generations | {MODEL} | CAP2={CAP2} RUN_SECONDS={RUN_SECONDS:.0f}", flush=True)

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="cuda").eval()
NL = model.config.num_hidden_layers
LAYERS = [int(round(x)) for x in np.linspace(NL // 4, NL - 1, 8)]
VEC = {L: torch.tensor(d["u"] * d["delta"], dtype=torch.bfloat16, device="cuda") for L, d in dirs.items()}
STATE = {"vec": None, "mode": None, "layer": None, "capture": False}
CAP = {}   # layer -> (last-token vector, CAP2-position vector or None), filled by the capture hooks


def make_hook(L):
    def hook(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        if STATE["capture"] and L in LAYERS:
            CAP[L] = (h[0, -1].float().cpu().numpy(),
                      h[0, CAP2 - 1].float().cpu().numpy() if CAP2 and h.shape[1] > CAP2 else None)
        if STATE["vec"] is None or STATE["layer"] != L:
            return
        if STATE["mode"] == "all" or h.shape[1] == 1:
            h += STATE["vec"]
        else:
            h[:, -1, :] += STATE["vec"]
    return hook


# hidden_states[L] is the OUTPUT of decoder layer L-1 (hidden_states[0] = embeddings), so the probed
# representation at "layer L" is read/edited by hooking layers[L-1].
handles = {L: model.model.layers[L - 1].register_forward_hook(make_hook(L)) for L in sorted(set(LAYERS) | set(VEC))}
feats = {L: [] for L in LAYERS}; feats_cap2 = {L: [] for L in LAYERS}; meta = []
done = set()
if os.path.exists("/workspace/probe32b_out.jsonl"):
    done = {(json.loads(l)["sample_id"], json.loads(l)["variant"], json.loads(l)["cond"]) for l in open("/workspace/probe32b_out.jsonl")}
fout = open("/workspace/probe32b_out.jsonl", "a")


def save():
    torch.save({"feats": {L: np.stack(v) for L, v in feats.items()},
                "feats_cap2": {L: np.stack(v) for L, v in feats_cap2.items()}, "cap2": CAP2, "meta": meta}, "/workspace/probe32b_feats.pt")


for i, r in enumerate(rows):
    if time.time() - T0 > RUN_SECONDS:
        print(f"  DEADLINE after {i} rows ({time.time()-T0:.0f}s)", flush=True); break
    try:
        msgs = [{"role": "system", "content": r["system"]}, {"role": "user", "content": r["user"]}]
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", enable_thinking=False)[:, :MAXTOK].to("cuda")
        with torch.inference_mode():
            STATE["vec"] = None; STATE["capture"] = True; CAP.clear()
            model(ids, logits_to_keep=1)
            STATE["capture"] = False
            for L in LAYERS:
                last, c2 = CAP[L]
                feats[L].append(last); feats_cap2[L].append(last if c2 is None else c2)
            meta.append({**{k: r[k] for k in ("sample_id", "variant", "is_hack", "task_id")}, "ntok": int(ids.shape[1]),
                         "cap2_hit": bool(CAP2 and ids.shape[1] > CAP2)})
            for c in r["conds"]:
                if (r["sample_id"], r["variant"], c["name"]) in done:
                    continue
                STATE["vec"] = None if c["name"] == "base" else VEC[c["layer"]] * c["k"]
                STATE["mode"] = c.get("mode"); STATE["layer"] = c.get("layer")
                gen = model.generate(ids, max_new_tokens=220, do_sample=False, pad_token_id=tok.eos_token_id)
                txt = tok.decode(gen[0, ids.shape[1]:], skip_special_tokens=True)
                m = SCORE_RE.search(txt)
                fout.write(json.dumps({"sample_id": r["sample_id"], "variant": r["variant"], "is_hack": r["is_hack"],
                                       "task_id": r["task_id"], "cond": c["name"], "verdict": float(m.group(1)) if m else None,
                                       "text": txt[:600], "ntok": int(ids.shape[1])}) + "\n"); fout.flush()
                del gen
            STATE["vec"] = None
        del ids
        if (i + 1) % 10 == 0:
            torch.cuda.empty_cache()
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(rows)} ({time.time()-T0:.0f}s)", flush=True)
        if (i + 1) % 100 == 0:
            save()
    except Exception as e:
        print(f"  skip {r['sample_id']}/{r['variant']}: {type(e).__name__}: {str(e)[:120]}", flush=True)
        STATE["vec"] = None; STATE["capture"] = False; torch.cuda.empty_cache()
save()
print(f"DONE {len(meta)} rows in {time.time()-T0:.0f}s", flush=True)
