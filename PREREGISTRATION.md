# Pre-registration: the make-or-break test

> **Status: planned, not carried out.** The project was stopped after Week 2 (Oct 2026), before any trust or
> attack code existed. Section 4 was never filled in and the rule below was never frozen or applied, so there are
> no results for this test. (Week 2 measured a seed std of at most 0.61 points on hop2@4; see NOTES.md.)

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

- sigma_seed = ______ points   (from commit ______ , date ______)

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
