# Notes

## Setup
- Date of graph-stats run: Oct 26, 2026. Baseline run: Oct 27, 2026.
- Compute: HiPerGator, account/QOS eee6778, node type: B200
- Env: /blue/eee6778/si.seenivasan/conda/envs/gnnrag (versions in requirements.lock.txt)
- Git commit of the graph-stats run: e917b05
- Git commit of the baseline run: d3d9dba

## Reader model
- Llama 3.2 3B not available on NaviGator (checked Oct 20, 2026).
- Main reader: granite-3.3-8b-instruct, temperature 0, max_tokens 64.
- Strong-reader check (later): llama-3.3-70b-instruct and/or gemma-4-31b-it.

## Data
- HotpotQA distractor, seed 42: 2000 train / 500 val questions.
- Gold passages per question: 2 for every question.
- Examples with fewer than 10 passages: 7 (handled with offsets, not i*10).
- Question types in val: 413 bridge (83%), 87 comparison (17%).

## Baseline: plain dense search (MiniLM, k = 4)
Command: `python run_pipeline.py configs/baseline_plain.yaml`

| n_q | EM | F1 | recall@2 | recall@4 | recall@10 | both_gold@2 | both_gold@4 | both_gold@10 | MRR | latency (ms, first run) |
|-----|----|----|----------|----------|-----------|-------------|-------------|--------------|-----|-------------------------|
| 100 |  0.43  |  0.590  |     0.615     |     0.775     |     1.0      |       0.31      |       0.56      |       1.0      |  0.897   |           276.615             |
| 500 |  0.398  |  0.554  |     0.621     |     0.784     |     1.0      |       0.316      |       0.596      |       1.0        |  0.900   |           611.609             |

Sanity checks:
- Raw reader outputs short? (look at cache/llm.jsonl): Yes, 1-3 words.
- 3 example failures (question, gold answer, prediction, were both golds retrieved?):

## Graph stats (val, 500 graphs)
Command: `python -m graph.run_graphs configs/baseline_plain.yaml [--sim 0.5]`

| edges | avg degree | isolated nodes / graph | gold pair linked | linked (bridge) | linked (comparison) | avg edges per graph |
|-------|------------|------------------------|------------------|-----------------|---------------------|---------------------|
| entity + title | 4.43 | 1.73 | 0.820 | 0.927 | 0.310 | 44.3 |
| + sim > 0.5    | 4.60 | 1.47 | 0.828 | 0.937 | 0.310 | 45.9 |

Takeaway:
- Bridge questions (83% of val) have the two gold passages directly linked 93% of the time, so the graph gives the GNN a path from the first fact to the second. This is the case the project is about.
- Comparison questions are linked only 31% of the time. That is expected: their two golds are about two separate entities (e.g. "who was born first, A or B?") that don't mention each other. The question connects them, not the passages.
- Similarity edges at 0.5 add about 1.6 edges per graph and +1 point on bridge linking. Not worth it: use entity + title edges as the main graph.

## Open issues / decisions
- Decision: Week 2 uses the entity + title graph (`graphs_val.pt`).
- Title edges are one-directional (passage i mentions j's title → i→j only). Make graphs undirected (`torch_geometric.utils.to_undirected`) before message passing, and note that "isolated" above counts out-degree only, so it slightly overstates truly isolated nodes. → Done in Week 2: `graph/dataset.py` makes edges undirected at load time (saved graphs unchanged).
- Comparison questions: consider adding a question node linked to every passage, or report bridge and comparison results separately. → Week 2 reports both splits; question node is `configs/gnn_relevance_qnode.yaml` (not run).

## Week 2: relevance-only model
Runbook: `WEEK2.md`. Command: `bash scripts/week2.sh`.
- Date of run: Oct 30, 2026. Code at commit cec4722 (week 2 code, unchanged for the run).
- Device: CPU (4 cores, cloud container, not HiPerGator). Same package versions as `requirements.lock.txt` but CPU torch and Python 3.11. Each training run took 15-45 s.
- Data and graphs were rebuilt from scratch with seed 42 and match Week 1: val graph stats are identical to the table above, and the plain_search row below gives the same recall@4 / both_gold@4 / MRR as the baseline.
- Train/dev/val: 1800 / 200 / 500 questions.
- hop2 rule counts (val): title 290, cos 210, none 0. (train: title 1179, cos 821, none 0.)
- No tuning: default configs only (`gnn_relevance.yaml`, `mlp_relevance.yaml`), seeds 0-2. Question-node variant and end-to-end QA not run.

### Results (`results/week2/compare.md`)

| method | seeds | hop2@2 | hop2@4 | hop2_mrr | both_gold@2 | both_gold@4 | recall@4 | mrr | hop2@4_bridge | hop2@4_comparison |
|---|---|---|---|---|---|---|---|---|---|---|
| plain_search | - | 38.0 | 63.2 | 35.5 | 31.6 | 59.6 | 78.4 | 90.0 | 58.6 | 85.1 |
| gnn_relevance | 3 | 63.7 ± 1.8 | 84.5 ± 0.5 | 55.5 ± 0.6 | 48.8 ± 3.4 | 78.4 ± 0.9 | 87.9 ± 0.5 | 89.3 ± 0.9 | 83.5 ± 0.3 | 89.3 ± 1.8 |
| mlp_relevance | 3 | 48.9 ± 2.4 | 74.7 ± 0.6 | 47.6 ± 1.2 | 33.9 ± 1.0 | 65.2 ± 0.9 | 81.1 ± 0.7 | 86.2 ± 0.3 | 71.8 ± 0.5 | 88.5 ± 1.1 |

Paired bootstrap on hop2@4 (10k resamples over questions, points): GNN − plain: +21.3 [+17.2, +25.4]; MLP − plain: +11.5 [+7.9, +15.1]; GNN − MLP: +9.8 [+7.1, +12.7]
Rescued / lost vs plain (hop2@4, majority over seeds): GNN 128 / 15 (net +113); MLP 80 / 24 (net +56)
Seed std of hop2@4: at most 0.61 points.

Caveat: when neither gold names the other (the "cos" rule, 210 of 500 val questions), hop2 is by definition the gold that plain search scores lower, so hop2@k is a bit harsh on plain search. both_gold@4 doesn't have that issue and tells the same story (59.6 → 65.2 MLP → 78.4 GNN).

### Takeaway
- A trained ranker finds the second fact much more often than plain search: hop2@4 goes from 63.2 to 84.5, and both golds are in the top 4 for 78% of questions instead of 60%.
- About half of that is the graph, not just training. The MLP control (same features, same training, no message passing) gets 74.7, and the GNN beats it by about 10 points with a 95% interval well clear of 0. Seed noise is small (≤ 0.6 points).
- The gain is almost all on bridge questions (58.6 → 71.8 MLP → 83.5 GNN), which is where the gold pair is linked 93% of the time. Comparison questions barely move (85.1 → 88.5 → 89.3), as expected, since their golds are rarely linked.
- Overall MRR doesn't improve (90.0 vs 89.3): plain search already puts one gold first most of the time. The GNN's gain is in pulling up the *second* passage.
- Limits: one dataset, 500 val questions, MiniLM features, no tuning, retrieval only (no reader EM/F1 with the GNN).

## Conclusion of Week 2 (written at the Oct 30 close-out; superseded by Weeks 3-5 below)
The project stopped after Week 2; the trust and attack work (Weeks 3-5) was never started. What it showed:
1. Plain dense search usually finds one gold passage but often misses the pair: recall@4 is 0.78, but both golds are in the top 4 only about 60% of the time. That gap is the multi-hop problem.
2. Simple entity + title edges directly link the two gold passages in 93% of bridge questions (31% for comparison), so the graph carries a path from the first fact to the second. Similarity edges add almost nothing.
3. A relevance-only GATv2 ranker over that graph uses the path: second-fact recall@4 goes from 63% (plain search) to 84%, about 10 points above an MLP trained the same way without message passing, with nearly all of the gain on bridge questions.

Next step would have been Week 3: plant fake passages into the val graphs and test whether trust inside message passing protects hop2@4 better than trust applied only at final ranking (`PREREGISTRATION.md`, never carried out).

## Weeks 3-5: trust in message passing (C) vs trust at final ranking (B)
Project reopened Nov 10, 2026. Pre-registration frozen first (commit 52d3dc8: sigma_seed = 0.61, so Δ = 3.0 points),
then the open design details logged in its change log together with the code (commit 26e1d8b), then the runs.
- Device: CPU, 4 cores, cloud container (same setup as Week 2: Python 3.11, CPU torch 2.11, PyG 2.8.1). Each Week 3
  training run took 1.5-3 min. Data and graphs rebuilt from scratch; `attacks.plant --check` confirms the new builder
  gives exactly the Week 1 edges (cols 0-1) on all 2000 train and 500 val graphs.
- Version A = the Week 2 GATv2, retrained with the Week 2 code: preds and logs for seeds 0-2 came out byte-identical to
  the committed Week 2 files, so A is exactly the Week 2 model.
- Commands: `bash scripts/week3.sh` (or `sbatch slurm/week3.sbatch`), then `python -m eval.week3`. Full output in
  `results/week3/compare.md`, numbers in `results/week3/summary.json`, per-question rankings in `results/week3/preds/`.
- md5 of the val attack files (first 12 chars), to check a rebuild on HPG gives the same fakes:
  val_none 2786a3c010d8, val_edit c752a5cc1a92, val_connected 87c5a9e726d6, val_connected_q 3f2dd9806285.

### Setup in short
- **Attacks** (`attacks/plant.py`): 3 fakes per question, each a copy of a gold passage (hop2, hop1, hop2) with 1-2
  entities swapped for same-type entities from other questions; title kept. `edit` is the pre-registered set.
  Week 5: `connected` (sentences shuffled + "See also: <3 other titles>", so it links into the graph) and
  `connected_q` (+ the question prepended, PoisonedRAG-style). Trust stage and C are trained on `edit` fakes only
  (train fakes seed 8, val fakes seed 7), so the Week 5 attacks are unseen.
- **New edge features**: agree / contradict. Contradict fires between near copies (same title or word Jaccard ≥ 0.5)
  whose entities differ. Clean graphs have 0.47 such pairs per question too (real look-alikes, e.g. two moth species
  pages, or "season 2" vs "season 3"), so it is not a free label.
- **B** = A − λ·(1 − trust_stage); λ per seed from dev: 1, 0.5, 1. **C** = trust-gated GATv2 (each message scaled by
  the sender's trust), relevance + trust heads, trained on clean + attacked train.
- Dev numbers (best epoch): trust stage AUC 0.98 / 0.98 / 0.97; C dev hop2@4 clean/attacked 0.885/0.92, 0.865/0.895,
  0.865/0.885 (seed 2 early-stopped at epoch 12, the others at 30-35).

### Results (val, 500 questions, hop2@4 in points, mean ± std over seeds 0-2)

| method | clean | edit | connected | connected_q | poison@4 (edit) |
|---|---|---|---|---|---|
| plain_search | 63.2 | 42.4 | 20.2 | 8.2 | 94.0 |
| A relevance only | 84.5 ± 0.5 | 65.9 ± 1.3 | 50.4 ± 1.6 | 34.1 ± 1.0 | 95.1 |
| B trust at ranking | 84.7 ± 0.6 | 69.6 ± 1.2 | 55.9 ± 1.7 | 36.7 ± 0.5 | 89.1 |
| C trust in message passing | 84.1 ± 1.0 | 81.8 ± 2.1 | 73.2 ± 4.3 | 49.1 ± 8.1 | 90.0 |
| *exploratory* | | | | | |
| A_adv (A trained on C's data, no trust) | 83.3 ± 1.7 | 81.8 ± 0.5 | 68.1 ± 0.6 | 43.3 ± 1.2 | 88.4 |
| B_adv (A_adv + trust at ranking) | 83.4 ± 1.7 | 81.7 ± 0.6 | 68.3 ± 0.3 | 43.2 ± 1.1 | 85.8 |
| C_rank (C + its own trust at ranking) | 84.2 ± 1.1 | 81.8 ± 2.1 | 73.0 ± 4.6 | 49.2 ± 7.9 | 89.8 |

Trust AUC on val nodes (real vs planted): trust stage 95.9 / 95.9 / 90.0, C's trust head 89.3 / 90.8 / 86.6
(edit / connected / connected_q).

**Pre-registered rule, C vs B** (Δ = 3.0):

| attack | C − B | every seed | bootstrap 95% CI | poison@4 B / C | clean A / C | outcome |
|---|---|---|---|---|---|---|
| edit (decision of record) | +12.2 | yes | [+9.3, +15.2] | 89.1 / 90.0 | 84.5 / 84.1 | **PASS** |
| connected | +17.3 | yes | [+14.5, +20.1] | 87.9 / 86.8 | 84.5 / 84.1 | PASS |
| connected_q | +12.4 | yes | [+9.5, +15.4] | 96.7 / 97.9 | 84.5 / 84.1 | PASS |

**Same rule, C vs B_adv** (exploratory, not the decision):

| attack | C − B_adv | every seed | bootstrap 95% CI | outcome |
|---|---|---|---|---|
| edit | +0.1 | no | [−2.1, +2.2] | WEAK (C ≈ B_adv) |
| connected | +4.9 | yes | [+2.7, +7.1] | PASS |
| connected_q | +5.9 | no (seed 2: 44.0 vs 40.8) | [+3.4, +8.3] | WEAK |

### Takeaway
- By the letter of the pre-registration, **C passes**: on the pre-registered `edit` attack it keeps the second fact in
  the top 4 for 81.8% of questions vs 69.6% for B, every seed, CI well above 0, guardrails fine. Same on both harder attacks.
- **But the comparison as pre-registered is confounded, and most of the gap isn't the gating.** B's relevance part is
  the Week 2 model, which never saw a fake; C's relevance head is trained with fakes as negatives. A_adv (same as A
  but trained on C's data, no trust at all) gets 81.8 on `edit`, exactly C's number. So on the attack it was trained
  for, all of C's gain over B comes from adversarial training data, not from putting trust inside message passing.
  This should have been caught when writing the pre-registration; it wasn't, and the rule is left as written.
- **Where the gating does add something is on the unseen, harder attacks**: C beats A_adv / B_adv by about 5 points
  on `connected` (every seed, CI [+2.7, +7.1]) and `connected_q` (CI [+3.4, +8.3], but not every seed). Under the
  same rule with B_adv in B's place: PASS on `connected`, WEAK on `connected_q`, WEAK (a tie) on `edit`. That is the
  honest version of the result: trust in message passing helps generalise to fakes that are wired into the graph,
  not on the fakes it was trained on.
- **Trust at final ranking barely matters once relevance is trained on fakes**: λ for B_adv came out 0, 0, 2 and
  B_adv ≈ A_adv everywhere. C_rank ≈ C as well. For A (never saw fakes) the trust stage is worth +3.7 points on `edit`.
- **Nobody keeps fakes out of the top 4**: poison@4 stays 85-98% for every method. A near-copy of a gold passage is,
  by relevance, as good as the gold, and with 3 fakes per question at least one gets in. The models protect the real
  second fact (hop2@4) but still hand the reader a fake. Whether that changes the answer is the Week 4 QA question.
- C is noisier across seeds (std up to 8 points on `connected_q` vs ~1 for the others), mostly seed 2 stopping early.
  Δ was set from Week 2's seed noise (0.61), which turns out to be much smaller than C's.
- Limits: one dataset, 500 val questions, CPU runs, programmatic fakes (no LLM-written ones), no tuning, C has 24% more
  parameters than A/B (1.02M vs 0.83M, the gates and trust head), retrieval only until the Week 4 QA rows exist.

### Week 4: end-to-end QA (needs NaviGator, run on HPG; not done yet)
```bash
python -m attacks.plant configs/trust_stage.yaml --split val --attack none
python -m attacks.plant configs/trust_stage.yaml --split val --attack edit
python -m generate.qa_week4 --sets none edit --methods plain_search A B C A_adv
```
Rows go to `results/week4.jsonl` (EM, F1, fake in context, answer taken from a fake).
