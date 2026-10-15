"""One OpenRouter chat call with retries, shared by every paid script."""
import asyncio
import os

URL = "https://openrouter.ai/api/v1/chat/completions"
NANO, SONNET, FLASH = "openai/gpt-5.4-nano", "anthropic/claude-sonnet-4.6", "google/gemini-2.5-flash"
OBFUSCATOR = "openai/gpt-5.4"


async def chat(client, sem, model, system, user, temperature=0, max_tokens=3000, timeout=180, retries=4):
    """Return the reply text, or None after `retries` failed attempts."""
    body = {"model": model, "temperature": temperature, "max_tokens": max_tokens,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    headers = {"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}", "X-Title": "cot-monitor-study"}
    async with sem:
        for attempt in range(retries):
            try:
                r = await client.post(URL, json=body, headers=headers, timeout=timeout)
                if r.status_code == 200:
                    return r.json()["choices"][0]["message"]["content"]
            except Exception:
                pass
            await asyncio.sleep(2 * (attempt + 1))
    return None
