"""Write compact, static data files for docs/ (GitHub Pages)."""
import json, base64, shutil, sys, datetime, numpy as np
sys.path.insert(0, "scripts")
from common import load, D, ROOT
c = load(); cat = json.load(open(D/"catalog.json"))
S = json.load(open(D/"summaries.json")) if (D/"summaries.json").exists() else {}
QA = json.load(open(D/"showcase_qa.json")) if (D/"showcase_qa.json").exists() else {}
W = ROOT/"docs/data"; (W/"p").mkdir(parents=True, exist_ok=True)
rows = [[p["id"], p["title"][:140], p["store"][:40], p["rating"], p["nr"], (p["img"] or "").replace("https://m.media-amazon.com/images/I/", ""), int(p["id"] in c["corpus"])] for p in cat]
json.dump(dict(cols=["id","title","store","rating","reviews","img","ai"], rows=rows), open(W/"catalog.json","w"), separators=(",",":"))
shutil.copy(D/"prod_emb.i8", W/"prod_emb.i8")
for pid, rs in c["corpus"].items():
    E = np.stack([c["E"][c["row"][r["rid"]]] for r in rs]); e8 = np.clip(np.round(E*127), -127, 127).astype(np.int8)
    revs = [dict(id=r["rid"], r=r["r"], t=r["t"], x=r["x"][:500], h=r["h"], v=int(r["v"]),
                 d=datetime.datetime.utcfromtimestamp(r["ts"]/1000).strftime("%b %Y")) for r in rs]
    json.dump(dict(reviews=revs, emb=base64.b64encode(e8.tobytes()).decode(), summary=S.get(pid), qa=QA.get(pid, [])),
              open(W/"p"/f"{pid}.json","w"), separators=(",",":"))
for f in ("cross_qa.json",):
    if (D/f).exists(): shutil.copy(D/f, W/f)
if (ROOT/"eval/metrics.json").exists(): shutil.copy(ROOT/"eval/metrics.json", W/"metrics.json")
print("site data written")
