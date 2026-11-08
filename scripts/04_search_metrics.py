import json
L = [l.split() for l in open("eval/search_labels.txt") if l[0] in "01"]
def m(col):
    p = sum(s.count("1")/5 for s in col)/len(col)
    rr = sum((1/(s.index("1")+1)) if "1" in s else 0 for s in col)/len(col)
    return round(p,3), round(rr,3)
s, k = m([a for a, b in L]), m([b for a, b in L])
r = dict(n_queries=len(L), semantic=dict(p_at_5=s[0], mrr=s[1]), keyword=dict(p_at_5=k[0], mrr=k[1]))
json.dump(r, open("eval/search_metrics.json","w"), indent=1); print(r)
