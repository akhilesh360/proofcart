"""Build compact catalog + review corpus from Amazon Reviews 2023 (All_Beauty)."""
import json, collections, pathlib
R = pathlib.Path("data/raw"); O = pathlib.Path("data/derived"); O.mkdir(parents=True, exist_ok=True)
meta = {}
for line in open(R/"meta_All_Beauty.jsonl"):
    m = json.loads(line)
    if not m.get("title"): continue
    img = next((i.get("large") for i in m.get("images") or [] if i.get("large")), None)
    meta[m["parent_asin"]] = dict(id=m["parent_asin"], title=m["title"][:200], store=m.get("store") or "",
        rating=m.get("average_rating"), n=m.get("rating_number") or 0, price=m.get("price"), img=img)
revs = collections.defaultdict(list)
for line in open(R/"All_Beauty.jsonl"):
    r = json.loads(line); t = (r.get("text") or "").strip()
    if r["parent_asin"] in meta and len(t) >= 40:
        revs[r["parent_asin"]].append(dict(r=r["rating"], t=(r.get("title") or "")[:120], x=t[:600].replace("<br />"," "),
            h=r.get("helpful_vote",0), v=r.get("verified_purchase",False), ts=r["timestamp"]))
cnt = {k: len(v) for k, v in revs.items()}
catalog = [m for k, m in meta.items() if cnt.get(k, 0) >= 3]
catalog.sort(key=lambda m: -cnt[m["id"]])
for m in catalog: m["nr"] = cnt[m["id"]]
top = catalog[:1500]
corpus = {}
for m in top:
    rs = sorted(revs[m["id"]], key=lambda r: (-r["h"], -len(r["x"])))[:25]
    for i, r in enumerate(rs): r["rid"] = f'{m["id"]}-{i}'
    corpus[m["id"]] = rs
json.dump(catalog, open(O/"catalog.json","w")); json.dump(corpus, open(O/"corpus.json","w"))
print("meta", len(meta), "with>=3 reviews", len(catalog), "top", len(top), "min reviews in top", top[-1]["nr"],
      "chunks", sum(map(len, corpus.values())))
