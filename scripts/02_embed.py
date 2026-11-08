"""Embed catalog titles and review chunks with all-MiniLM-L6-v2; store int8 (scale 127) for the browser."""
import json, numpy as np, pathlib
from sentence_transformers import SentenceTransformer
O = pathlib.Path("data/derived"); m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
cat = json.load(open(O/"catalog.json")); corpus = json.load(open(O/"corpus.json"))
def q(e): return np.clip(np.round(e*127), -127, 127).astype(np.int8)
pe = m.encode([c["title"] for c in cat], batch_size=256, normalize_embeddings=True, show_progress_bar=False)
np.save(O/"prod_emb.npy", pe.astype(np.float32)); q(pe).tofile(O/"prod_emb.i8")
ids, texts = [], []
for pid, rs in corpus.items():
    for r in rs: ids.append(r["rid"]); texts.append((r["t"]+". "+r["x"]))
re_ = m.encode(texts, batch_size=256, normalize_embeddings=True, show_progress_bar=False)
np.save(O/"rev_emb.npy", re_.astype(np.float32)); q(re_).tofile(O/"rev_emb.i8"); json.dump(ids, open(O/"rev_ids.json","w"))
print(pe.shape, re_.shape)
