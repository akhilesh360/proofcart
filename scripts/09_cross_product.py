"""Questions across products: semantic product search over the 1,500 assistant products, then review retrieval
inside each candidate, then one grounded answer that cites reviews from several products."""
import json, sys, numpy as np
sys.path.insert(0, "scripts")
from common import *; from llm import chat
c = load(); cat = json.load(open(D/"catalog.json")); idx = {p["id"]: i for i, p in enumerate(cat)}
PE = np.fromfile(D/"prod_emb.i8", dtype=np.int8).reshape(len(cat), 384).astype(np.float32) / 127
pids = list(c["corpus"]); P = PE[[idx[p] for p in pids]]
QS = ["Which hair diffuser works best for curly hair?", "Which exfoliating cloth is gentle enough for sensitive skin?",
      "Which beard trimmer is easiest to clean?", "Which hair brush is best for fine hair?", "Which blackhead remover actually works without bruising?",
      "Which eyebrow pen lasts all day without smudging?", "Which shower cap keeps long hair dry?", "Which curling iron holds curls longest?"]
SYS = SYS_GROUNDED.replace("1 to 3 sentences", "2 to 4 sentences; name the products you compare")
out = []
for q in QS:
    e = embed(q); top = np.argsort(-(P @ e))[:4]; sn = []
    for t in top:
        for r, s in retrieve(pids[t], q, 2): sn.append((dict(r, t=f'{c["cat"][pids[t]]["title"][:60]} | {r["t"]}'), s))
    a, dt = chat(SYS, f"Question: {q}\n\nReviews (each starts with the product name):\n{fmt(sn)}", "cross_answer", 500)
    out.append(dict(q=q, products=[pids[t] for t in top], rids=[r["rid"] for r, _ in sn], sec=round(dt, 2), **a))
json.dump(out, open(D/"cross_qa.json", "w"), indent=1); print("cross", len(out))
