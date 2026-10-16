"""The probe on action-only inputs: does its edge over the verdict read the actions alone?

Two steps around one pod run (Appendix I, Table 6, action-only row):
  python -m generate.probe.action_only prepare
      writes runs/probe/internals_inputs_ao.jsonl: the 150 hacks and 150 legit solves of
      runs/probe/internals_inputs_obf.jsonl, rendered at action-only scope (the dataset's strip
      transform), everything else unchanged. A hack's actions are the same in its original and
      rewritten forms, so each hack enters once, as variant "obf". Asserts first that a fresh
      full-scope render reproduces every stored input.
  python -m generate.probe.launch_pod_8b --inputs runs/probe/internals_inputs_ao.jsonl --no-verbal --tag ao
  python -m generate.probe.action_only collect
      writes data/probe/probe_8b_action_only.json: the fetched results plus the population ids and
      each input's length in characters.
Needs the Terminal Wrench clone; the pod step needs RUNPOD_API_KEY, HF_TOKEN and runpodctl (under $1).
"""
import json
import sys

from generate.common import tw
from generate.common.paths import DATA
from generate.probe import pod

FULL = pod.RUNS / "internals_inputs_obf.jsonl"
AO = pod.RUNS / "internals_inputs_ao.jsonl"
OUT = DATA / "probe" / "probe_8b_action_only.json"


def prepare():
    by_sid = {tw.sid(s): s for s in tw.collect()}
    rows = [json.loads(line) for line in open(FULL) if line.strip()]
    out = []
    for r in rows:
        if r["variant"] == "obf":
            continue
        atif = tw.load(by_sid[r["sample_id"]])
        assert tw.render(atif, r["task_id"]) == r["user"] and r["system"] == tw.DEV_MSG, f"{r['sample_id']}: stored input differs"
        out.append({**r, "variant": "obf" if r["variant"] == "orig" else "legit",
                    "user": tw.render(atif, r["task_id"], scope="action-only")})
    AO.write_text("".join(json.dumps(r) + "\n" for r in out))
    print(f"wrote {len(out)} action-only inputs to {AO}")


def collect():
    res = json.load(open(pod.RUNS / "probe_results_ao.json"))
    rows = [json.loads(line) for line in open(AO) if line.strip()]
    res["population"] = {"hack_sids": [r["sample_id"] for r in rows if r["is_hack"]],
                         "legit_sids": [r["sample_id"] for r in rows if not r["is_hack"]]}
    res["input_chars"] = {"hack": [len(r["user"]) for r in rows if r["is_hack"]],
                          "legit": [len(r["user"]) for r in rows if not r["is_hack"]]}
    OUT.write_text(json.dumps(res, indent=1))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    {"prepare": prepare, "collect": collect}[sys.argv[1] if len(sys.argv) > 1 else "prepare"]()
