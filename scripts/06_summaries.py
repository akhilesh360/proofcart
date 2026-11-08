"""Cited "Customers say" summaries for the 1,500 assistant products; naive vs cited comparison + judge on the first 100."""
import json, sys
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, "scripts")
from common import *; from llm import chat
c = load(); pids = list(c["corpus"]); OUT = D/"summaries.json"
done = json.load(open(OUT)) if OUT.exists() else {}
def revs(pid): return [(r, 0) for r in c["corpus"][pid][:15]]
def one(pid):
    if pid in done: return pid, done[pid]
    s, _ = chat(SYS_SUMMARY, "Reviews:\n" + fmt(revs(pid), 300), "summary", 400)
    return pid, s
with ThreadPoolExecutor(12) as ex:
    for k, (pid, s) in enumerate(ex.map(one, pids)):
        done[pid] = s
        if k % 100 == 0: json.dump(done, open(OUT, "w"))
json.dump(done, open(OUT, "w"))
# Before/after on the first 100 products
CMP = ROOT/"eval/summary_compare.json"
def cmp(pid):
    rv = revs(pid); allr = fmt(rv, 300)
    naive, _ = chat(SYS_SUMMARY_NAIVE, "Reviews:\n" + allr, "summary_naive", 300)
    for cl in naive.get("claims", []):
        cl["judge"] = chat(SYS_JUDGE, f'Claim: {cl["text"]}\n\nEvidence (all reviews shown to the writer):\n{allr}', "judge", 80)[0]
    cited = json.loads(json.dumps(done[pid]))
    for cl in cited.get("claims", []):
        ev = "\n".join(f'[{k}] {rv[k-1][0]["x"][:300]}' for k in cl.get("cites", []) if 1 <= k <= len(rv)) or "(no evidence cited)"
        cl["judge"] = chat(SYS_JUDGE, f'Claim: {cl["text"]}\n\nEvidence:\n{ev}', "judge", 80)[0]
    return dict(pid=pid, naive=naive, cited=cited)
with ThreadPoolExecutor(8) as ex: res = list(ex.map(cmp, pids[:100]))
json.dump(res, open(CMP, "w"), indent=1); print("summaries", len(done), "compared", len(res))
