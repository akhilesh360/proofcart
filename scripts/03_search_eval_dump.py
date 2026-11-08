"""Run the 40 hand-written queries through semantic search (int8, as in the browser) and a keyword baseline; dump top-5 for hand labelling."""
import json, re, numpy as np, pathlib
from sentence_transformers import SentenceTransformer
O = pathlib.Path("data/derived"); cat = json.load(open(O/"catalog.json"))
E = np.fromfile(O/"prod_emb.i8", dtype=np.int8).reshape(len(cat), 384).astype(np.float32)
m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
qs = [l.strip() for l in open("eval/search_queries.txt") if l.strip()]
toks = [set(re.findall(r"[a-z]+", c["title"].lower())) for c in cat]
pop = np.log1p(np.array([c["nr"] for c in cat], dtype=np.float32))
out = []
for q in qs:
    e = m.encode(q, normalize_embeddings=True)
    sem = np.argsort(-(E @ e))[:5]
    qt = set(re.findall(r"[a-z]+", q.lower())) - {"for","the","a","with","of","that","to","and"}
    kw = np.array([len(qt & t) for t in toks], dtype=np.float32) + pop*1e-3
    key = np.argsort(-kw)[:5]
    out.append(dict(q=q, semantic=[cat[i]["id"] for i in sem], keyword=[cat[i]["id"] for i in key]))
    print("\n#", q)
    for tag, idx in (("S", sem), ("K", key)):
        for i in idx: print(" ", tag, cat[i]["id"], cat[i]["title"][:95])
json.dump(out, open("eval/search_runs.json", "w"), indent=0)
