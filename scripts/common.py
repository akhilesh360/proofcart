import json, pathlib, numpy as np
ROOT = pathlib.Path(__file__).resolve().parent.parent; D = ROOT / "data/derived"
REFUSAL = "Not enough evidence in reviews to answer this."
_cache = {}
def load():
    if not _cache:
        _cache["cat"] = {c["id"]: c for c in json.load(open(D/"catalog.json"))}
        _cache["corpus"] = json.load(open(D/"corpus.json"))
        ids = json.load(open(D/"rev_ids.json"))
        E = np.fromfile(D/"rev_emb.i8", dtype=np.int8).reshape(len(ids), 384).astype(np.float32) / 127
        _cache["row"] = {r: i for i, r in enumerate(ids)}; _cache["E"] = E
    return _cache
_model = None
def embed(text):
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    return _model.encode(text, normalize_embeddings=True)
def retrieve(pid, question, k=6):
    """Same math as the browser: int8 review vectors / 127, cosine vs the query."""
    c = load(); rs = c["corpus"][pid]; q = embed(question)
    sims = np.array([c["E"][c["row"][r["rid"]]] @ q for r in rs])
    order = np.argsort(-sims)[:k]
    return [(rs[i], float(sims[i])) for i in order]
def fmt(snips, n=420):
    return "\n".join(f'[{i+1}] ({r["r"]:.0f} stars) {r["t"]}. {r["x"][:n]}' for i, (r, _) in enumerate(snips))

SYS_GROUNDED = """You are a shopping assistant that answers ONLY from the customer reviews provided.
Rules:
- Every factual sentence must cite the review numbers that support it, like [2] or [1][4].
- Do not use outside knowledge, the product title, or assumptions. If the reviews do not directly address the question, refuse.
- If reviews disagree, say so and cite both sides.
- Be concise: 1 to 3 sentences.
Return JSON: {"refused": bool, "answer": string, "claims": [{"text": string, "cites": [int]}]}.
When refusing, answer must be exactly "%s" and claims must be []."""% REFUSAL
SYS_NAIVE = """You are a helpful shopping assistant. Answer the shopper's question about the product in 1 to 3 sentences.
Return JSON: {"answer": string, "claims": [{"text": string}]} where claims split the answer into its separate factual statements."""
SYS_SUMMARY = """Write a "Customers say" summary of the reviews below for a product page.
Rules: 3 to 5 short claims covering the most common themes, positive and negative, in proportion to how often they appear.
Every claim must cite the review numbers that directly support it. No outside knowledge.
Return JSON: {"claims": [{"text": string, "cites": [int]}]}"""
SYS_SUMMARY_NAIVE = """Write a "Customers say" summary of the reviews below for a product page: 3 to 5 short claims about what customers think.
Return JSON: {"claims": [{"text": string}]}"""
SYS_JUDGE = """You check whether a claim about a product is supported by the given customer review evidence.
"supported": the evidence directly states it. "partial": evidence supports part of it or it overgeneralizes. "unsupported": the evidence does not state it or contradicts it.
Return JSON: {"verdict": "supported"|"partial"|"unsupported", "reason": string (max 15 words)}"""
