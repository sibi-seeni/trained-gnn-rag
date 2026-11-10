# Week 3 / 5: trust in message passing (C) vs trust at final ranking (B)

Val, 500 questions, 3 fakes per question on attacked sets. Points, mean ± std over seeds 0-2. λ per seed (B): [1, 0.5, 1], (C_rank): [0, 0, 0.25], (B_adv): [0, 0, 2]. Rows below the line are exploratory and not part of the decision.


## clean (no fakes)

| method | hop2@4 | hop2@2 | both_gold@4 | recall@4 | hop2@4_bridge | hop2@4_comparison |
|---|---|---|---|---|---|---|
| plain_search | 63.2 | 38.0 | 59.6 | 78.4 | 58.6 | 85.1 |
| A | 84.5 ± 0.5 | 63.8 ± 1.7 | 78.5 ± 0.8 | 88.0 ± 0.5 | 83.5 ± 0.2 | 89.3 ± 1.8 |
| B | 84.7 ± 0.6 | 63.6 ± 1.9 | 78.3 ± 0.6 | 87.9 ± 0.4 | 83.7 ± 0.4 | 89.3 ± 1.8 |
| C | 84.1 ± 1.0 | 63.1 ± 1.5 | 77.1 ± 0.4 | 87.5 ± 0.5 | 83.0 ± 1.3 | 89.7 ± 1.1 |
| *exploratory* | | | | | | |
| A_adv | 83.3 ± 1.7 | 61.1 ± 1.8 | 76.6 ± 1.6 | 87.1 ± 0.9 | 83.1 ± 1.5 | 84.7 ± 2.9 |
| B_adv | 83.4 ± 1.7 | 61.3 ± 1.8 | 76.7 ± 1.6 | 87.1 ± 0.9 | 83.1 ± 1.5 | 84.7 ± 2.9 |
| C_rank | 84.2 ± 1.1 | 62.9 ± 1.6 | 77.2 ± 0.3 | 87.6 ± 0.5 | 83.1 ± 1.5 | 89.7 ± 1.1 |

## attack: edit

| method | hop2@4 | poison@4 | hop2@2 | both_gold@4 | recall@4 | hop2@4_bridge | hop2@4_comparison |
|---|---|---|---|---|---|---|---|
| plain_search | 42.4 | 94.0 | 10.6 | 34.8 | 65.6 | 39.2 | 57.5 |
| A | 65.9 ± 1.3 | 95.1 ± 0.8 | 29.3 ± 1.3 | 46.7 ± 1.7 | 71.6 ± 0.6 | 66.2 ± 1.4 | 64.4 ± 1.1 |
| B | 69.6 ± 1.2 | 89.1 ± 1.8 | 39.4 ± 0.9 | 54.7 ± 2.4 | 75.4 ± 1.3 | 69.8 ± 1.0 | 68.6 ± 2.4 |
| C | 81.8 ± 2.1 | 90.0 ± 2.1 | 56.8 ± 3.1 | 70.3 ± 3.6 | 84.5 ± 1.9 | 81.3 ± 1.6 | 84.3 ± 5.2 |
| *exploratory* | | | | | | | |
| A_adv | 81.8 ± 0.5 | 88.4 ± 0.3 | 53.8 ± 1.8 | 72.0 ± 1.1 | 85.0 ± 0.4 | 82.2 ± 1.1 | 79.7 ± 7.5 |
| B_adv | 81.7 ± 0.6 | 85.8 ± 4.2 | 54.3 ± 1.4 | 72.1 ± 0.9 | 85.0 ± 0.4 | 82.0 ± 0.7 | 80.5 ± 6.4 |
| C_rank | 81.8 ± 2.1 | 89.8 ± 1.7 | 56.7 ± 3.3 | 70.4 ± 3.5 | 84.6 ± 1.8 | 81.3 ± 1.6 | 84.3 ± 5.2 |

Trust AUC (real vs planted, val nodes): trust_stage 95.9 ± 0.9, C_trust_head 89.3 ± 2.5

## attack: connected

| method | hop2@4 | poison@4 | hop2@2 | both_gold@4 | recall@4 | hop2@4_bridge | hop2@4_comparison |
|---|---|---|---|---|---|---|---|
| plain_search | 20.2 | 97.4 | 6.0 | 12.2 | 52.5 | 21.5 | 13.8 |
| A | 50.4 ± 1.6 | 97.4 ± 0.7 | 24.2 ± 1.4 | 30.1 ± 1.9 | 61.6 ± 1.0 | 52.1 ± 1.3 | 42.5 ± 3.4 |
| B | 55.9 ± 1.7 | 87.9 ± 2.8 | 33.1 ± 1.2 | 40.7 ± 0.3 | 66.9 ± 0.2 | 57.3 ± 1.2 | 49.4 ± 4.1 |
| C | 73.2 ± 4.3 | 86.8 ± 4.7 | 49.4 ± 5.0 | 60.6 ± 6.3 | 78.6 ± 3.7 | 73.9 ± 4.3 | 69.7 ± 5.7 |
| *exploratory* | | | | | | | |
| A_adv | 68.1 ± 0.6 | 88.7 ± 1.4 | 45.0 ± 0.9 | 57.6 ± 1.0 | 75.9 ± 0.6 | 69.2 ± 0.7 | 62.8 ± 5.3 |
| B_adv | 68.3 ± 0.3 | 84.9 ± 5.3 | 45.5 ± 1.4 | 57.9 ± 0.6 | 76.0 ± 0.4 | 69.2 ± 0.7 | 63.6 ± 4.8 |
| C_rank | 73.0 ± 4.6 | 86.5 ± 4.3 | 49.3 ± 5.1 | 60.3 ± 6.7 | 78.4 ± 4.0 | 73.8 ± 4.4 | 69.0 ± 7.0 |

Trust AUC (real vs planted, val nodes): trust_stage 95.9 ± 1.1, C_trust_head 90.8 ± 2.7

## attack: connected_q

| method | hop2@4 | poison@4 | hop2@2 | both_gold@4 | recall@4 | hop2@4_bridge | hop2@4_comparison |
|---|---|---|---|---|---|---|---|
| plain_search | 8.2 | 99.8 | 0.2 | 0.0 | 42.7 | 9.9 | 0.0 |
| A | 34.1 ± 1.0 | 99.3 ± 0.1 | 9.7 ± 1.8 | 9.3 ± 0.5 | 45.3 ± 0.8 | 36.0 ± 1.1 | 24.9 ± 0.7 |
| B | 36.7 ± 0.5 | 96.7 ± 1.8 | 16.1 ± 0.7 | 15.4 ± 0.4 | 46.1 ± 1.0 | 39.0 ± 1.3 | 25.7 ± 3.3 |
| C | 49.1 ± 8.1 | 97.9 ± 1.4 | 25.7 ± 5.9 | 24.6 ± 7.4 | 54.7 ± 5.8 | 50.3 ± 8.0 | 43.3 ± 8.9 |
| *exploratory* | | | | | | | |
| A_adv | 43.3 ± 1.2 | 98.7 ± 0.2 | 22.0 ± 1.4 | 19.8 ± 0.5 | 48.7 ± 0.8 | 45.3 ± 1.3 | 34.1 ± 6.9 |
| B_adv | 43.2 ± 1.1 | 97.3 ± 2.7 | 23.1 ± 2.3 | 20.8 ± 1.6 | 48.8 ± 0.8 | 44.7 ± 0.6 | 36.0 ± 7.4 |
| C_rank | 49.2 ± 7.9 | 97.9 ± 1.4 | 25.5 ± 6.2 | 24.6 ± 7.4 | 54.4 ± 6.1 | 50.4 ± 7.8 | 43.3 ± 8.9 |

Trust AUC (real vs planted, val nodes): trust_stage 90.0 ± 0.6, C_trust_head 86.6 ± 2.9

## Pre-registered rule (Δ = max(3.0, 2 × 0.61) = 3.0 points)

| attack | C − B (mean) | per seed B / C | bootstrap 95% CI | poison@4 B / C | clean hop2@4 A / C | 1 size | 2 every seed | 3 CI > 0 | 4 guardrails | outcome |
|---|---|---|---|---|---|---|---|---|---|---|
| edit | +12.2 | 70.8/84.2, 68.4/80.8, 69.6/80.4 | [+9.3, +15.2] | 89.1 / 90.0 | 84.5 / 84.1 | yes | yes | yes | yes | **PASS** |
| connected | +17.3 | 54.0/77.8, 57.0/72.6, 56.8/69.2 | [+14.5, +20.1] | 87.9 / 86.8 | 84.5 / 84.1 | yes | yes | yes | yes | **PASS** |
| connected_q | +12.4 | 37.2/57.0, 36.4/49.4, 36.4/40.8 | [+9.5, +15.4] | 96.7 / 97.9 | 84.5 / 84.1 | yes | yes | yes | yes | **PASS** |

The pre-registered attack set is `edit`; `connected` and `connected_q` are the Week 5 harder attacks, judged with the same rule (PREREGISTRATION.md section 6 and change log).

### Exploratory: the same rule with B_adv in B's place (B's relevance part trained on C's data)

| attack | C − B_adv (mean) | per seed B_adv / C | bootstrap 95% CI | poison@4 B_adv / C | 1 size | 2 every seed | 3 CI > 0 | 4 guardrails | outcome |
|---|---|---|---|---|---|---|---|---|---|
| edit | +0.1 | 81.4/84.2, 82.4/80.8, 81.4/80.4 | [-2.1, +2.2] | 85.8 / 90.0 | no | no | no | no | WEAK |
| connected | +4.9 | 68.2/77.8, 68.6/72.6, 68.0/69.2 | [+2.7, +7.1] | 84.9 / 86.8 | yes | yes | yes | yes | PASS |
| connected_q | +5.9 | 43.6/57.0, 42.0/49.4, 44.0/40.8 | [+3.4, +8.3] | 97.3 / 97.9 | yes | no | yes | yes | WEAK |

### Exploratory pairs (same arithmetic, not part of the decision)

| attack | pair | diff (mean) | bootstrap 95% CI | every seed |
|---|---|---|---|---|
| edit | C − A_adv | +0.0 | [-2.1, +2.1] | no |
| edit | C − B_adv | +0.1 | [-2.1, +2.2] | no |
| edit | B_adv − B | +12.1 | [+9.4, +14.9] | yes |
| edit | A_adv − A | +15.9 | [+12.9, +19.1] | yes |
| edit | C_rank − C | +0.0 | [+0.0, +0.0] | no |
| edit | B − A | +3.7 | [+2.0, +5.5] | yes |
| connected | C − A_adv | +5.1 | [+2.9, +7.3] | yes |
| connected | C − B_adv | +4.9 | [+2.7, +7.1] | yes |
| connected | B_adv − B | +12.3 | [+9.5, +15.3] | yes |
| connected | A_adv − A | +17.7 | [+14.1, +21.3] | yes |
| connected | C_rank − C | -0.2 | [-0.6, +0.2] | no |
| connected | B − A | +5.5 | [+3.2, +7.9] | yes |
| connected_q | C − A_adv | +5.7 | [+3.2, +8.3] | no |
| connected_q | C − B_adv | +5.9 | [+3.4, +8.3] | no |
| connected_q | B_adv − B | +6.5 | [+3.7, +9.4] | yes |
| connected_q | A_adv − A | +9.3 | [+5.5, +13.0] | yes |
| connected_q | C_rank − C | +0.1 | [-0.1, +0.4] | no |
| connected_q | B − A | +2.6 | [+0.1, +5.1] | yes |
