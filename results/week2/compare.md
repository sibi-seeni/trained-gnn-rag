
## Week 2 results (val, 500 questions, numbers in points, mean ± std over seeds)

| method | seeds | hop2@2 | hop2@4 | hop2_mrr | both_gold@2 | both_gold@4 | recall@4 | mrr | hop2@4_bridge | hop2@4_comparison |
|---|---|---|---|---|---|---|---|---|---|---|
| plain_search | - | 38.0 | 63.2 | 35.5 | 31.6 | 59.6 | 78.4 | 90.0 | 58.6 | 85.1 |
| gnn_relevance | 3 | 63.7 ± 1.8 | 84.5 ± 0.5 | 55.5 ± 0.6 | 48.8 ± 3.4 | 78.4 ± 0.9 | 87.9 ± 0.5 | 89.3 ± 0.9 | 83.5 ± 0.3 | 89.3 ± 1.8 |
| mlp_relevance | 3 | 48.9 ± 2.4 | 74.7 ± 0.6 | 47.6 ± 1.2 | 33.9 ± 1.0 | 65.2 ± 0.9 | 81.1 ± 0.7 | 86.2 ± 0.3 | 71.8 ± 0.5 | 88.5 ± 1.1 |

## Paired bootstrap on hop2@4 (10k resamples over questions; points)

- gnn_relevance − plain_search: +21.3  [95% CI +17.2, +25.4]
- mlp_relevance − plain_search: +11.5  [95% CI +7.9, +15.1]
- gnn_relevance − mlp_relevance: +9.8  [95% CI +7.1, +12.7]

## Second facts rescued / lost vs plain search (hop2@4, majority over seeds)

- gnn_relevance: rescued 128, lost 15, net +113
- mlp_relevance: rescued 80, lost 24, net +56

## Seed noise

sigma_seed (largest seed std of hop2@4 among trained models) = 0.61 points
Copy this into PREREGISTRATION.md section 4 and commit it before any Week 3 trust numbers exist.
