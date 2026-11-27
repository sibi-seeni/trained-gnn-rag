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
| plain_search | 41.2 | 94.2 | 8.8 | 33.2 | 64.5 | 39.2 | 50.6 |
| A | 64.0 ± 0.4 | 96.1 ± 0.9 | 28.3 ± 1.6 | 44.5 ± 1.8 | 70.3 ± 0.7 | 63.9 ± 0.2 | 64.4 ± 3.0 |
| B | 69.7 ± 1.5 | 89.7 ± 1.3 | 37.7 ± 0.3 | 54.2 ± 2.7 | 75.2 ± 1.4 | 70.0 ± 1.1 | 68.2 ± 4.0 |
| C | 83.3 ± 1.3 | 90.4 ± 3.1 | 55.1 ± 3.8 | 72.1 ± 2.8 | 85.4 ± 1.5 | 82.2 ± 1.7 | 88.1 ± 1.8 |
| *exploratory* | | | | | | | |
| A_adv | 82.0 ± 0.2 | 89.5 ± 0.3 | 53.7 ± 1.2 | 72.9 ± 1.3 | 85.5 ± 0.4 | 81.9 ± 1.0 | 82.4 ± 4.8 |
| B_adv | 81.5 ± 0.9 | 86.5 ± 5.3 | 53.5 ± 1.3 | 72.3 ± 2.4 | 85.1 ± 1.1 | 81.4 ± 0.1 | 81.6 ± 6.1 |
| C_rank | 83.2 ± 1.3 | 90.3 ± 2.9 | 55.4 ± 3.3 | 72.1 ± 2.9 | 85.3 ± 1.7 | 82.2 ± 1.7 | 88.1 ± 1.8 |

Trust AUC (real vs planted, val nodes): trust_stage 95.8 ± 0.9, C_trust_head 89.2 ± 2.7

## attack: connected

| method | hop2@4 | poison@4 | hop2@2 | both_gold@4 | recall@4 | hop2@4_bridge | hop2@4_comparison |
|---|---|---|---|---|---|---|---|
| plain_search | 17.8 | 98.2 | 6.2 | 10.0 | 51.3 | 19.1 | 11.5 |
| A | 48.5 ± 1.0 | 97.7 ± 0.3 | 22.5 ± 1.7 | 27.4 ± 1.1 | 59.9 ± 0.8 | 51.3 ± 1.5 | 34.9 ± 1.8 |
| B | 55.1 ± 1.8 | 89.9 ± 3.0 | 30.3 ± 0.6 | 38.9 ± 1.3 | 65.2 ± 1.0 | 56.3 ± 1.8 | 49.0 ± 1.8 |
| C | 70.4 ± 4.7 | 89.0 ± 4.2 | 46.3 ± 5.6 | 56.5 ± 5.8 | 76.2 ± 3.6 | 71.3 ± 4.1 | 66.3 ± 8.9 |
| *exploratory* | | | | | | | |
| A_adv | 65.1 ± 1.9 | 91.5 ± 1.2 | 42.8 ± 1.4 | 53.2 ± 0.5 | 73.4 ± 0.8 | 66.0 ± 2.1 | 60.9 ± 3.0 |
| B_adv | 65.3 ± 1.8 | 88.2 ± 4.5 | 43.5 ± 1.3 | 53.1 ± 0.6 | 73.3 ± 0.9 | 66.1 ± 2.1 | 61.7 ± 1.8 |
| C_rank | 70.5 ± 4.6 | 88.9 ± 4.1 | 46.1 ± 5.9 | 56.7 ± 5.5 | 76.2 ± 3.6 | 71.3 ± 4.1 | 66.7 ± 8.3 |

Trust AUC (real vs planted, val nodes): trust_stage 95.3 ± 0.9, C_trust_head 90.5 ± 2.7

## attack: connected_q

| method | hop2@4 | poison@4 | hop2@2 | both_gold@4 | recall@4 | hop2@4_bridge | hop2@4_comparison |
|---|---|---|---|---|---|---|---|
| plain_search | 8.2 | 99.8 | 0.0 | 0.2 | 42.8 | 9.9 | 0.0 |
| A | 32.7 ± 0.9 | 99.3 ± 0.1 | 8.7 ± 1.5 | 6.7 ± 0.8 | 43.7 ± 1.6 | 34.9 ± 0.6 | 22.2 ± 2.9 |
| B | 34.6 ± 0.7 | 96.9 ± 1.0 | 16.0 ± 2.1 | 12.1 ± 0.5 | 43.7 ± 1.2 | 36.7 ± 1.1 | 24.5 ± 1.3 |
| C | 46.9 ± 7.8 | 97.9 ± 1.5 | 22.9 ± 5.9 | 22.9 ± 8.7 | 52.5 ± 6.4 | 50.0 ± 8.5 | 32.6 ± 4.6 |
| *exploratory* | | | | | | | |
| A_adv | 41.2 ± 0.6 | 98.9 ± 0.5 | 21.6 ± 2.8 | 18.3 ± 0.8 | 46.7 ± 0.6 | 44.3 ± 0.7 | 26.4 ± 3.4 |
| B_adv | 41.1 ± 0.5 | 97.7 ± 2.3 | 22.3 ± 3.4 | 19.2 ± 0.9 | 46.6 ± 0.5 | 44.0 ± 0.4 | 27.6 ± 4.0 |
| C_rank | 46.8 ± 8.0 | 97.9 ± 1.5 | 22.7 ± 6.4 | 22.9 ± 8.7 | 52.2 ± 6.8 | 49.9 ± 8.6 | 32.2 ± 5.3 |

Trust AUC (real vs planted, val nodes): trust_stage 89.3 ± 0.5, C_trust_head 86.1 ± 3.1

## Pre-registered rule (Δ = max(3.0, 2 × 0.61) = 3.0 points)

| attack | C − B (mean) | per seed B / C | bootstrap 95% CI | poison@4 B / C | clean hop2@4 A / C | 1 size | 2 every seed | 3 CI > 0 | 4 guardrails | outcome |
|---|---|---|---|---|---|---|---|---|---|---|
| edit | +13.6 | 70.8/84.6, 68.0/82.0, 70.2/83.2 | [+10.7, +16.5] | 89.7 / 90.4 | 84.5 / 84.1 | yes | yes | yes | yes | **PASS** |
| connected | +15.3 | 53.0/75.6, 56.4/69.2, 55.8/66.4 | [+12.5, +18.2] | 89.9 / 89.0 | 84.5 / 84.1 | yes | yes | yes | yes | **PASS** |
| connected_q | +12.3 | 35.4/54.8, 34.4/46.8, 34.0/39.2 | [+9.5, +15.2] | 96.9 / 97.9 | 84.5 / 84.1 | yes | yes | yes | yes | **PASS** |

The pre-registered attack set is `edit`; `connected` and `connected_q` are the Week 5 harder attacks, judged with the same rule (PREREGISTRATION.md section 6 and change log).

### Exploratory: the same rule with B_adv in B's place (B's relevance part trained on C's data)

| attack | C − B_adv (mean) | per seed B_adv / C | bootstrap 95% CI | poison@4 B_adv / C | 1 size | 2 every seed | 3 CI > 0 | 4 guardrails | outcome |
|---|---|---|---|---|---|---|---|---|---|
| edit | +1.8 | 82.2/84.6, 81.8/82.0, 80.4/83.2 | [-0.2, +3.9] | 86.5 / 90.4 | no | yes | no | no | WEAK |
| connected | +5.1 | 67.2/75.6, 63.6/69.2, 65.2/66.4 | [+2.8, +7.4] | 88.2 / 89.0 | yes | yes | yes | yes | PASS |
| connected_q | +5.8 | 41.2/54.8, 40.6/46.8, 41.6/39.2 | [+3.5, +8.1] | 97.7 / 97.9 | yes | no | yes | yes | WEAK |

### Exploratory pairs (same arithmetic, not part of the decision)

| attack | pair | diff (mean) | bootstrap 95% CI | every seed |
|---|---|---|---|---|
| edit | C − A_adv | +1.3 | [-0.7, +3.3] | yes |
| edit | C − B_adv | +1.8 | [-0.2, +3.9] | yes |
| edit | B_adv − B | +11.8 | [+9.1, +14.5] | yes |
| edit | A_adv − A | +18.0 | [+14.9, +21.2] | yes |
| edit | C_rank − C | -0.1 | [-0.2, +0.0] | no |
| edit | B − A | +5.7 | [+3.7, +7.7] | yes |
| connected | C − A_adv | +5.3 | [+2.9, +7.6] | yes |
| connected | C − B_adv | +5.1 | [+2.8, +7.4] | yes |
| connected | B_adv − B | +10.3 | [+7.3, +13.2] | yes |
| connected | A_adv − A | +16.7 | [+13.1, +20.3] | yes |
| connected | C_rank − C | +0.1 | [-0.1, +0.3] | no |
| connected | B − A | +6.6 | [+4.1, +9.0] | yes |
| connected_q | C − A_adv | +5.7 | [+3.3, +8.1] | no |
| connected_q | C − B_adv | +5.8 | [+3.5, +8.1] | no |
| connected_q | B_adv − B | +6.5 | [+3.5, +9.6] | yes |
| connected_q | A_adv − A | +8.5 | [+4.6, +12.4] | yes |
| connected_q | C_rank − C | -0.1 | [-0.3, +0.0] | no |
| connected_q | B − A | +1.9 | [-0.6, +4.4] | yes |
