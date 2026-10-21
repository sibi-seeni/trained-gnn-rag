def retrieval_metrics(ranked_gold_flags, ks):
    # ranked_gold_flags: list of lists of bools in rank order
    out, n = {}, len(ranked_gold_flags)
    for k in ks:
        out[f"recall@{k}"] = sum(sum(r[:k]) / max(1, sum(r)) for r in ranked_gold_flags) / n
        out[f"both_gold@{k}"] = sum(sum(r[:k]) == sum(r) for r in ranked_gold_flags) / n
    out["mrr"] = sum(next((1/(i+1) for i, g in enumerate(r) if g), 0) for r in ranked_gold_flags) / n
    return out