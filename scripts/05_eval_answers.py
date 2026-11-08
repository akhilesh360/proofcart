"""Grounded vs naive answers on the 60 hand-written eval questions, then an LLM-judge pass on every claim."""
import json, sys
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, "scripts")
from common import *; from llm import chat
c = load(); Q = [json.loads(l) for l in open(ROOT/"eval/questions.jsonl")]
def run(i):
    q = Q[i]; p = c["cat"][q["pid"]]; sn = retrieve(q["pid"], q["q"])
    g, gt = chat(SYS_GROUNDED, f'Question: {q["q"]}\n\nReviews:\n{fmt(sn)}', "eval_grounded", 400)
    n, nt = chat(SYS_NAIVE, f'Product: {p["title"]}\nQuestion: {q["q"]}', "eval_naive", 300)
    allrev = fmt([(r, 0) for r in c["corpus"][q["pid"]]], 300)
    for cl in g.get("claims", []):
        ev = "\n".join(f'[{k}] {sn[k-1][0]["x"][:420]}' for k in cl.get("cites", []) if 1 <= k <= len(sn)) or "(no evidence cited)"
        cl["judge"] = chat(SYS_JUDGE, f'Claim: {cl["text"]}\n\nEvidence:\n{ev}', "judge", 80)[0]
    for cl in n.get("claims", []):
        cl["judge"] = chat(SYS_JUDGE, f'Claim: {cl["text"]}\n\nEvidence (all available reviews of this product):\n{allrev}', "judge", 80)[0]
    return dict(**q, title=p["title"], snippets=[dict(rid=r["rid"], sim=round(s, 3), stars=r["r"], text=r["t"]+". "+r["x"]) for r, s in sn],
                grounded=g, grounded_sec=round(gt, 2), naive=n, naive_sec=round(nt, 2))
with ThreadPoolExecutor(8) as ex: out = list(ex.map(run, range(len(Q))))
json.dump(out, open(ROOT/"eval/answers.json", "w"), indent=1); print("done", len(out))
