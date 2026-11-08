"""Aggregate every evaluation number into eval/metrics.json (used by README and the Eval panel)."""
import json, statistics as st, numpy as np, pathlib
R = pathlib.Path(__file__).resolve().parent.parent; E = R/"eval"; D = R/"data/derived"
A = json.load(open(E/"answers.json")); H = [json.loads(l) for l in open(E/"hand_labels.jsonl")]
ans = [h for h in H if h["answerable"]]; un = [h for h in H if not h["answerable"]]
g_claims = sum(len(A[h["i"]]["grounded"].get("claims", [])) for h in H)
g_uns = sum(h["grounded_unsupported"] for h in H)
cites = sum(len(c.get("cites", [])) for a in A for c in a["grounded"].get("claims", []))
wrong = sum(h["wrong_cites"] for h in H)
judge_g = [c["judge"]["verdict"] for a in A for c in a["grounded"].get("claims", [])]
judge_n = [c["judge"]["verdict"] for a in A for c in a["naive"].get("claims", [])]
# hand vs judge agreement on grounded claims: hand says all supported except listed
agree = sum(v != "unsupported" for v in judge_g) / len(judge_g)  # hand marked every grounded claim supported
answers = dict(n=len(H), n_products=len({a["pid"] for a in A}), n_answerable=len(ans), n_unanswerable=len(un),
  grounded=dict(n_claims=g_claims, unsupported_rate=g_uns / g_claims, citation_precision=(cites - wrong) / cites, n_cites=cites,
    correct_refusal=sum(A[h["i"]]["grounded"].get("refused", False) for h in un) / len(un),
    answer_rate=sum(not A[h["i"]]["grounded"].get("refused", False) for h in ans) / len(ans),
    judge_unsupported_rate=judge_g.count("unsupported") / len(judge_g)),
  naive=dict(n_claims=sum(h["naive_claims"] for h in H), unsupported_rate=sum(h["naive_unsupported"] for h in H) / sum(h["naive_claims"] for h in H),
    correct_refusal=sum(h["naive_declined"] for h in un) / len(un), answer_rate=sum(not h["naive_declined"] for h in ans) / len(ans),
    judge_unsupported_rate=judge_n.count("unsupported") / len(judge_n), judge_n_claims=len(judge_n)),
  judge_agreement=agree)
# live (retrieval-only) refusal threshold from top-1 similarity, answerable vs not
top = {h["i"]: A[h["i"]]["snippets"][0]["sim"] for h in H}
best = max(((sum(top[h["i"]] >= t for h in ans) + sum(top[h["i"]] < t for h in un)) / len(H), t) for t in np.arange(0.2, 0.7, 0.01))
live = dict(threshold=round(float(best[1]), 2), accuracy=best[0], median_top_sim_answerable=st.median(top[h["i"]] for h in ans), median_top_sim_unanswerable=st.median(top[h["i"]] for h in un))
# summaries
S = json.load(open(E/"summary_compare.json"))
jc = [c["judge"]["verdict"] for s in S for c in s["cited"].get("claims", [])]
jn = [c["judge"]["verdict"] for s in S for c in s["naive"].get("claims", [])]
SH = [json.loads(l) for l in open(E/"summary_hand_labels.jsonl")]
summ = dict(n_products=len(S), judge_cited_unsupported=jc.count("unsupported") / len(jc), judge_naive_unsupported=jn.count("unsupported") / len(jn),
  judge_cited_partial=jc.count("partial") / len(jc), judge_naive_partial=jn.count("partial") / len(jn), judge_n_cited=len(jc), judge_n_naive=len(jn),
  hand_n=len(SH), hand_unsupported=sum(h["verdict"] == "unsupported" for h in SH) / len(SH), hand_partial=sum(h["verdict"] == "partial" for h in SH) / len(SH),
  hand_citation_precision=sum(h["cites_ok"] for h in SH) / sum(h["cites"] for h in SH),
  hand_judge_agreement=sum(h["verdict"] == h["judge"] for h in SH) / len(SH))
# efficiency
F = json.load(open(E/"efficiency.json")); WPM = 238
eff = dict(n=len(F), median_reviews=st.median(f["reviews_to_answer"] for f in F), mean_reviews=st.mean(f["reviews_to_answer"] for f in F),
  median_words=st.median(f["words_to_answer"] for f in F), mean_words=st.mean(f["words_to_answer"] for f in F),
  median_answer_words=st.median(f["answer_words"] for f in F),
  median_minutes_saved=st.median((f["words_to_answer"] - f["answer_words"]) / WPM for f in F),
  mean_minutes_saved=st.mean((f["words_to_answer"] - f["answer_words"]) / WPM for f in F), wpm_assumed=WPM,
  share_answer_in_first_review=sum(f["reviews_to_answer"] == 1 for f in F) / len(F))
# cost and latency
U = [json.loads(l) for l in open(D/"usage.jsonl")]
def per(task): x = [u for u in U if u["task"] == task]; return sum(u["usd"] for u in x) / len(x), x
ga, gx = per("eval_grounded"); sa, sx = per("showcase_answer"); su, _ = per("summary")
lat = sorted(u["sec"] for u in gx + sx)
cost = dict(total_usd=sum(u["usd"] for u in U), calls=len(U), tokens_in=sum(u["pin"] for u in U), tokens_out=sum(u["pout"] for u in U),
  per_1k_answers=1000 * (sum(u["usd"] for u in gx + sx) / len(gx + sx)), per_1k_products_summary=1000 * su,
  p50_latency_s=float(np.percentile(lat, 50)), p90_latency_s=float(np.percentile(lat, 90)), n_latency=len(lat),
  models=sorted({u["model"] for u in U}))
search = json.load(open(E/"search_metrics.json"))
M = dict(answers=answers, live=live, live_threshold=live["threshold"], summaries=summ, efficiency=eff, cost=cost, search=search,
  headline=dict(grounded_unsupported=answers["grounded"]["unsupported_rate"], naive_unsupported=answers["naive"]["unsupported_rate"],
    correct_refusal=answers["grounded"]["correct_refusal"], naive_refusal=answers["naive"]["correct_refusal"],
    median_reviews_to_answer=eff["median_reviews"], cost_per_1k_answers=cost["per_1k_answers"]))
json.dump(M, open(E/"metrics.json", "w"), indent=1); print(json.dumps(M, indent=1))
