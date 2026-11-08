"""Five common shopper questions per showcase product (top 100 by reviews), each answered with grounded retrieval."""
import json, sys
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, "scripts")
from common import *; from llm import chat
c = load(); pids = list(c["corpus"])[:100]; OUT = D/"showcase_qa.json"
SYS_Q = """List the 5 questions shoppers most commonly ask before buying this kind of product (fit, durability, comfort, safety, smell, value, etc.).
Short, natural questions. Return JSON: {"questions": [string, ...]}"""
def one(pid):
    qs = chat(SYS_Q, "Product: " + c["cat"][pid]["title"], "showcase_questions", 200)[0]["questions"][:5]
    out = []
    for q in qs:
        sn = retrieve(pid, q)
        try: a, dt = chat(SYS_GROUNDED, f"Question: {q}\n\nReviews:\n{fmt(sn)}", "showcase_answer", 400)
        except ValueError: continue
        out.append(dict(q=q, sec=round(dt, 2), rids=[r["rid"] for r, _ in sn], **a))
    return pid, out
with ThreadPoolExecutor(10) as ex: res = dict(ex.map(one, pids))
json.dump(res, open(OUT, "w")); print("showcase", len(res))
