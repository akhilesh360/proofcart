"""DeepSeek client (OpenAI-compatible) with a hard spend cap and a usage log. Key is read from .env and never printed."""
import json, os, pathlib, time, threading
from openai import OpenAI
ROOT = pathlib.Path(__file__).resolve().parent.parent
CAP_USD = 3.00
# Published DeepSeek prices per 1M tokens, PEAK rates (the higher of peak/off-peak), https://api-docs.deepseek.com/quick_start/pricing
PRICE = dict(in_miss=0.30, in_hit=0.006, out=1.20)
LOG = ROOT / "data/derived/usage.jsonl"
_lock = threading.Lock()

def _key():
    for line in open(ROOT / ".env"):
        if line.strip().startswith("DEEPSEEK_API_KEY="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("DEEPSEEK_API_KEY missing from .env")

client = OpenAI(api_key=_key(), base_url="https://api.deepseek.com", timeout=120)
MODEL = os.environ.get("LLM_MODEL", "deepseek-chat")

def spent():
    if not LOG.exists(): return 0.0
    return sum(json.loads(l)["usd"] for l in open(LOG))

def cost(u):
    hit = getattr(u, "prompt_cache_hit_tokens", 0) or 0
    miss = u.prompt_tokens - hit
    return (miss * PRICE["in_miss"] + hit * PRICE["in_hit"] + u.completion_tokens * PRICE["out"]) / 1e6

def chat(system, user, task, max_tokens=700, json_mode=True, _retry=True):
    with _lock:
        if spent() >= CAP_USD - 0.05: raise SystemExit(f"Budget cap reached: ${spent():.3f}")
    t0 = time.time()
    for attempt in range(4):
        try:
            r = client.chat.completions.create(model=MODEL, max_tokens=max_tokens, temperature=0,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                response_format={"type": "json_object"} if json_mode else None,
                extra_body={"thinking": {"type": "disabled"}})
            break
        except SystemExit: raise
        except Exception as e:
            if attempt == 3: raise
            time.sleep(2 * (attempt + 1))
    dt = time.time() - t0
    u = r.usage; usd = cost(u)
    with _lock, open(LOG, "a") as f:
        f.write(json.dumps(dict(task=task, model=r.model, pin=u.prompt_tokens, pout=u.completion_tokens,
            hit=getattr(u, "prompt_cache_hit_tokens", 0) or 0, usd=usd, sec=round(dt, 3))) + "\n")
    txt = r.choices[0].message.content
    if not json_mode: return txt, dt
    try: return json.loads(txt), dt
    except json.JSONDecodeError:
        if _retry: return chat(system, user, task, max_tokens * 2, json_mode, False)
        raise ValueError("model returned invalid JSON twice")
