# Pre-registration: the make-or-break test

> **Status: frozen Nov 10, 2026, then carried out the same day.** Rule frozen in commit 2a0b180, open details logged
> in section 7 with the code in dee16b7, both before any trust numbers. **Outcome on the pre-registered `edit` attack:
> PASS** (C − B = +12.2 points, CI [+9.3, +15.2]), and PASS on both Week 5 attacks. But an exploratory control shows
> the B/C contrast as written here is confounded (B's relevance model never saw fakes, C's did): against B_adv, which
> fixes that, C ties on `edit` and is ~5 points better only on the unseen harder attacks. See NOTES.md, Weeks 3-5.
> Sections 1-6 are left exactly as frozen.

Written in Week 2 (Oct 26 – Nov 1, 2026), **before any trust or attack numbers exist**.
The git commit that adds this file is the timestamp. If anything below changes later, don't
edit it silently: add a dated line to section 7 saying what changed and why, and say so in the report.

## 1. The question

Does putting trust **inside message passing** protect retrieval better than applying the same
trust signal **only at final ranking**? If not, the extra GNN machinery isn't earning its keep,
and the capstone should be built around the simpler design.

## 2. The three versions

All three use the same graphs, node features, GATv2 backbone size, training data, seeds and
tuning budget. Trust-stage training data and features are identical for B and C. The only
difference between B and C is where trust enters.

| | Version | What it is |
|---|---|---|
| A | Relevance-only | The Week 2 GATv2 ranker, no trust signal. |
| B | Trust at final ranking | A's relevance score, then rescored with the trust stage's output: `final = relevance - λ·(1 - trust)`. λ picked on the dev split (train), never on val. |
| C | Trust in message passing | Trust gates the attention / messages inside each GATv2 layer, plus the two output heads. |

Seeds: 0, 1, 2 for every version (three runs each).

## 3. Evaluation set and metrics

- Questions: the same 500 val questions as Week 2 (`data/processed/val.jsonl`, seed 42).
- Attack set for the first pass: simple fakes made by editing real passages, planted into the val graphs.
  Fakes are made with a fixed seed and the same fakes are used for A, B and C.
- **Primary metric: attacked hop2@4** — with fakes planted, is the true second-fact passage still in the top 4?
  (Same definition as Week 2, `eval/second_hop.py`, so the seed noise measured in Week 2 applies to it.)
- Secondary: poison@4 — fraction of questions with at least one planted passage in the top 4 (lower is better).
- Guardrail: clean hop2@4 (no fakes).
- Everything is also reported split by bridge / comparison, but the decision uses all 500 questions.

## 4. Seed noise (filled in at the end of Week 2, then frozen)

`sigma_seed` = largest standard deviation of hop2@4 across seeds 0–2 among the Week 2 trained models
(printed by `python -m eval.compare_week2`).

- sigma_seed = **0.61** points (mlp_relevance; gnn_relevance was 0.50), from the Week 2 run at code commit cec4722,
  results committed in 279ee90, Oct 30, 2026 (`results/week2/summary.json`, `sigma_seed_hop2@4`).
- So Δ = max(3.0, 2 × 0.61) = **3.0 points**.

## 5. The "clearly better" rule

Let Δ = max(**3.0 points**, **2 × sigma_seed**).

C is **clearly better** than B only if **all four** hold:

1. **Size:** mean attacked hop2@4 of C minus mean attacked hop2@4 of B ≥ Δ (means over seeds 0–2).
2. **Consistency:** C beats B on attacked hop2@4 for every seed pair (C_s0 > B_s0, C_s1 > B_s1, C_s2 > B_s2).
3. **Not luck of the questions:** a paired bootstrap over the 500 questions (per-question hits averaged over seeds,
   10,000 resamples) gives a 95% interval for C − B whose lower end is above 0.
4. **No cheating the guardrails:** C's poison@4 is not worse than B's by more than 2 points, and C's clean hop2@4
   is within 2 points of A's (trust shouldn't be bought by losing ordinary relevance).

## 6. What each outcome means

- **Pass** (all four hold): trust-in-message-passing is the core of the capstone and the paper.
- **Weak** (C ≥ B on the mean but one or more conditions fail): no claim that C is better. Report it as is; carry
  both B and C into the capstone test bench as configurations, and look at the harder attacks (well-connected
  fakes, Week 5) as where a difference would show up if it exists — using this same rule.
- **Fail** (C < B on the mean): write it up as a negative result. The capstone builds the poisoning test bench
  around the simpler trust-at-ranking design.

The decision of record is the final run of this comparison (full setup, after Week 5). The Week 3 run is a first
pass and uses the same rule, so it shows early which way things are going but doesn't settle it.

## 7. Change log

- Oct 2026: first version, written before any trust numbers.
- Nov 10, 2026: project reopened. Filled in section 4 (sigma_seed = 0.61, so Δ = 3.0) and froze the rule. No other
  change to sections 1-6. No trust or attack code exists at this commit.
- Nov 10, 2026 (before any trust or attack numbers; code in the same commit): how the open details are filled in.
  Nothing in sections 1-6 changes; this only pins down what they left open.
  - **Attack set (`edit`, the pre-registered one)**: `attacks/plant.py`. 3 fakes per question, each a copy of a gold
    passage (hop2, hop1, hop2 again) with 1-2 entities swapped for same-type entities from other questions (the answer
    first if the passage contains it; never the article's own subject). Title kept. Val fakes use seed 7, train fakes
    seed 8. Fakes are ordinary scorable nodes, so top 4 is over all 13 passages.
  - **Features**: Week 2 node features. A keeps edge cols 0-1 (it is the Week 2 model, retrained with the same code).
    The trust stage and C add col 3 agree and col 4 contradict (definitions in `attacks/plant.py`).
  - **Trust stage (B)**: a GATv2 the same size as A, trained with BCE (real vs planted) on the 1800 fit questions clean
    + the same 1800 with `edit` fakes; checkpoint chosen on dev fake-detection AUC (dev = the Week 2 dev split, clean +
    attacked). λ grid {0, .25, .5, 1, 2, 4, 8, 16, 32, 64}, chosen per seed on mean(dev clean hop2@4, dev attacked
    hop2@4), ties to the smaller λ.
  - **C**: before each GATv2 layer a gate reads the node states and gives trust t_j; every message node j sends is
    multiplied by t_j. Relevance head (the ranking score) and trust head on [h_L, h_0]. Loss = listwise + 1.0 × BCE on
    the trust head and the gates. Same training data as the trust stage; checkpoint chosen on the same dev criterion as
    λ. No other tuning for any version.
  - **Guardrail 4** reads as |clean hop2@4 of C − clean hop2@4 of A| ≤ 2 points.
  - **Week 5 harder attacks**: `connected` (sentences shuffled + a "See also" sentence naming 3 other passages, so the
    fake is well connected) and `connected_q` (same + the question prepended). Neither is seen in training. Judged with
    the same rule. The decision of record is the rule on `edit` with the final code; the harder attacks are reported
    next to it, and don't override it.
  - **Exploratory rows** (reported, not part of the decision): A_adv (A's architecture with C's features and training
    data, no trust), B_adv (A_adv + trust stage at ranking), C_rank (C + its own trust head at ranking).

