"""Reviews-to-answer: for each hand-verified answerable question, walk ALL of the product's reviews in default
order (most helpful first, then newest) and find the first review that answers it (LLM-detected in batches of 10)."""
import json, sys, collections, statistics
sys.path.insert(0, "scripts")
from common import *; from llm import chat
A = json.load(open(ROOT/"eval/answers.json")); H = {h["i"]: h for h in map(json.loads, open(ROOT/"eval/hand_labels.jsonl"))}
want = {a["pid"] for i, a in enumerate(A) if H[i]["answerable"]}
allrev = collections.defaultdict(list)
for line in open(ROOT/"data/raw/All_Beauty.jsonl"):
    r = json.loads(line)
    if r["parent_asin"] in want: allrev[r["parent_asin"]].append(r)
SYS = """For each numbered customer review, decide if it contains information that answers the shopper's question (even partially).
Return JSON: {"answering": [review numbers]} (empty list if none)."""
rows = []
for i, a in enumerate(A):
    if not H[i]["answerable"]: continue
    rs = sorted(allrev[a["pid"]], key=lambda r: (-r.get("helpful_vote", 0), -r["timestamp"]))
    pos, words = None, 0
    for b in range(0, min(len(rs), 150), 10):
        batch = rs[b:b+10]
        txt = "\n".join(f'[{k+1}] {r.get("title","")}. {r.get("text","")[:700]}' for k, r in enumerate(batch))
        hit = chat(SYS, f'Question: {a["q"]}\n\nReviews:\n{txt}', "efficiency_scan", 60)[0].get("answering", [])
        hit = [h for h in hit if isinstance(h, int) and 1 <= h <= len(batch)]
        if hit: pos = b + min(hit); break
    if pos is None: pos = min(len(rs), 150)
    words = sum(len((r.get("title","") + " " + r.get("text","")).split()) for r in rs[:pos])
    ans_words = len(a["grounded"]["answer"].split())
    rows.append(dict(i=i, q=a["q"], pid=a["pid"], n_reviews=len(rs), reviews_to_answer=pos, words_to_answer=words, answer_words=ans_words))
json.dump(rows, open(ROOT/"eval/efficiency.json", "w"), indent=1)
print(len(rows), "median reviews", statistics.median(r["reviews_to_answer"] for r in rows))
