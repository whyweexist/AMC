# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** [Your Team Name]
**Team Members:** [List all team members]
**Submission Date:** [Date]

> Numbers marked `⟨…⟩` are read from `output/metrics.json` / `output/run.log` of the final run on the
> real data and must be filled in before packaging. Numbers explicitly labelled *synthetic* come from the
> end-to-end smoke test on generated data and are only there to show the pipeline runs.

---

## 1. Executive Summary

We treat the task as what the metric says it is: a per-entity **set prediction** problem scored by macro
F₀.₅ with singletons counted. The pipeline is a three-stage funnel — (1) multi-view retrieval with
character n-gram TF-IDF, exact keys and (optionally) multilingual embeddings, (2) a *learned, calibrated
pruning* stage whose output **is** `candidate_pairs.tsv`, and (3) a LightGBM matcher with reverse
"competition" features, a one-to-one structural constraint, and a decision layer that maximises the
expected F₀.₅ of each entity's answer using an explicit singleton model. Every feature is
language-agnostic and every corpus statistic (IDF) is recomputed per country on the test files themselves,
which is what lets the same model transfer to the unseen French records.

---

## 2. Methodology

### 2.1 Problem Analysis

Facts extracted from the statement that shaped the design:

* **Per-entity F₀.₅** with F(∅,∅)=1, F(S,∅)=0 and F(∅,Y)=0. The empty answer is a real prediction with
  value P(Y=∅); it must be modelled, not left as "nothing crossed a threshold".
* **Candidate-set size is ranked**, subject to `candidate_pairs.tsv` being exactly what the model scores.
  The right objective is therefore *the smallest candidate set that does not reduce the final F₀.₅*, not
  raw recall.
* **France is unseen** → nothing may be hard-coded to US/India (no suffix lists as the only mechanism, no
  6-digit PIN assumption, no country one-hot).
* **Source 1 is deduplicated** → each S2/S3 record very likely belongs to at most one S1 entity.

Data-driven checks performed automatically by the code at start-up (`eda_decisions`, see `run.log`):

| check | value on training data | consequence |
|---|---|---|
| share of GT pairs with equal country label | ⟨country agreement⟩ | country used as a hard block key if ≥ 0.99 |
| share of S2/S3 GT records mapped to exactly one S1 entity | ⟨uniqueness⟩ | hard one-to-one assignment if ≥ 0.995 |
| singleton rate | ⟨n_single / n_S1⟩ | prior of the singleton model |
| match-count distribution (0/1/2/3+) | ⟨dist⟩ | sets `kmax` |

Noise patterns observed (fill from `output/error_samples.txt` and manual inspection):
⟨abbreviations / suffix swaps / typos / reordering / landmark phrases / missing postal codes …⟩

### 2.2 Solution Strategy

**Approach Type:** Blocking + learned pruning + gradient-boosted classifier + decision-theoretic set selection (Hybrid).

**Core Innovation:**
1. The blocking output is itself a calibrated classifier's output, so the candidate set can be made as
   small as the final metric allows, with a provable bound on the expected number of lost matches.
2. Instead of a global threshold, each entity's answer is chosen by exact expected-F₀.₅ maximisation over
   top-k prefixes, with the empty set valued by a dedicated singleton model. (A global-threshold policy is
   kept as a competitor and the better policy is selected on out-of-fold validation, so the rule can never
   hurt.)
3. Transductive per-country IDF and purely relative features give a model that transfers to an unseen
   country; this was validated with leave-one-country-out training (US→India and India→US).

---

## 3. Candidate Generation (Blocking)

**Blocking keys used**

* *View 1* — character `char_wb` 3–4-gram TF-IDF cosine on the normalised name, exact top-15 per S1 entity
  (`sparse_dot_topn`), inside the country block.
* *View 2* — the same on name + address, top-10.
* *View 3* — exact keys: (postal-like digit run ≥4 digits, 3-char prefix of the rarest core-name token),
  (name acronym, street number), and the first 8 characters of the space-free core name; blocks larger than
  100 records are skipped.
* *View 4 (optional, `--use-dense`)* — `multilingual-e5-small` embeddings of "name | address", HNSW top-10.
* Union of the views → raw candidate table with one row per (S1, S2/S3) and one score per view.

**Learned pruning (Stage 2).** A small LightGBM model on cheap features (view scores, within-entity rank
and margin, IDF-weighted Jaccard of core-name tokens, token-set ratio, 3-valued postal / street-number
agreement, source flag, number of candidates) is trained out-of-fold (GroupKFold by S1 entity) and
calibrated with isotonic regression. A candidate survives if `p₁ ≥ ε` and it is among the top-`kmax` (=10)
of its entity. ε is the largest value that keeps ≥ 99.5 % of the retrievable true pairs on the out-of-fold
predictions. **This pruned set is written verbatim to `candidate_pairs.tsv` and is exactly what the
Stage-3 model scores.**

Why this is safe: the decision layer only ever *includes* a candidate whose calibrated probability exceeds
≈ 0.8·F* (Section 4), and the expected number of true matches removed by pruning is bounded by the sum of
the pruned calibrated probabilities, Σ_{pruned} p₁ ≤ |pruned|·ε (the code logs this sum).

**Candidate pairs generated**

| stage | pairs | avg. per S1 entity | reduction ratio | pairs completeness (recall ceiling) |
|---|---|---|---|---|
| raw retrieval (train, OOF) | ⟨⟩ | ⟨⟩ | ⟨⟩ | ⟨⟩ |
| after Stage-2 pruning (train, OOF) | ⟨⟩ | ⟨⟩ | ⟨⟩ | ⟨⟩ |
| raw retrieval (test) | ⟨⟩ | ⟨⟩ | ⟨⟩ | – |
| **`candidate_pairs.tsv` (test)** | ⟨⟩ | ⟨⟩ | ⟨⟩ | – |

*(synthetic smoke test: raw 27.0/entity, completeness 1.000 → pruned 1.02/entity, completeness 0.996)*

**How true matches were not lost.** (i) three complementary views (character-level, address-aware,
exact keys) so that a typo, an abbreviation or a reordering has to defeat all of them at once;
(ii) the pruning threshold is set on out-of-fold predictions against an explicit recall constraint;
(iii) pairs completeness is measured at every stage and reported above; (iv) a scale fallback is
documented: for billion-record deployments the exact top-k is replaced by MinHash-LSH over the same
n-gram shingles, whose collision probability 1−(1−sʳ)ᵇ is an S-curve tuned to the same recall target.

---

## 4. Matching Model

**Features used** (all country- and language-agnostic; IDF tables are computed per country on the
union of the three test source files, i.e. unsupervised use of the provided data only):

* Name features: Jaro–Winkler, normalised Levenshtein, token-sort / token-set / partial ratios
  (RapidFuzz), space-free Jaro–Winkler, IDF-weighted Jaccard of core tokens, IDF of the rarest *shared*
  token vs. the rarest *unshared* token and their difference, acronym match, legal-suffix conflict,
  token-count difference, length ratio, char-n-gram cosine from the retrieval views.
* Address features: Levenshtein / token-set / partial ratios on the landmark-stripped core address,
  IDF-weighted Jaccard of address tokens and (separately) of landmark tokens, three-valued agreement of
  postal-like codes and of short street numbers, "both have codes" flag, address-present flags.
* Other: candidate source (S2/S3), Stage-2 calibrated probability `p₁`, within-entity rank and margin of
  `p₁`, and **competition features** computed on the candidate graph — the entity's rank and margin among
  all S1 entities competing for the same S2/S3 record, the number of competitors, and a mutual-best flag.
  Optional (`--use-cross-encoder`): the logit of a Ditto-style `mdeberta-v3-base` cross-encoder fine-tuned
  on serialised pairs, produced out-of-fold for training rows.

**Model type:** LightGBM (MIT) binary classifier, 800 rounds, trained out-of-fold with GroupKFold(5) by
S1 entity on the *pruned* candidate distribution (hard negatives only), then isotonic-calibrated.
Structural post-processing: each S2/S3 record's probabilities across competing S1 entities are
normalised so that Σᵢ pᵢⱼ ≤ 1 and, when the training data shows one-to-one ownership, only the argmax S1
entity keeps its probability.

**Threshold selection method:** per-entity expected-F₀.₅ maximisation, not a threshold.
For an entity with sorted calibrated probabilities p₁ ≥ … ≥ pₙ, the F-optimal prediction under
conditional independence is a top-k prefix (Lewis 1995; Jansche 2007; Nan et al. 2012), so we evaluate
k = 0…n exactly: |S∩Y| ~ PoissonBinomial(p₁..p_k) and |Y∖S| ~ PoissonBinomial(p_{k+1}..pₙ) give
E[F₀.₅(top_k)] in O(n²). The empty answer is valued by a separate entity-level LightGBM singleton model
P̂(Y=∅) (features: max / second / sum of probabilities, number of candidates, best address agreements,
mutual-best flag). The marginal rule this implies — include a candidate iff p > F/(1+β²) = 0.8·F — is the
F₀.₅ analogue of the F₁ result of Lipton, Elkan & Narayanaswamy (2014) and explains why the bar for a
second match is higher once a confident first match exists. A logit temperature and a scale on P̂(Y=∅)
are tuned on out-of-fold predictions; a global-threshold policy over τ ∈ {0.4,…,0.8} is scored on the
same predictions and the better policy is used (chosen policy for the final run: ⟨decision⟩).

---

## 5. Results & Error Analysis

* **F₀.₅ Score (macro), out-of-fold on training data:** ⟨validation_macro_f05⟩
  (per country: ⟨validation_per_country_f05⟩; global-threshold-0.5 baseline: ⟨baseline⟩)
* **Leave-one-country-out** (`--holdout-country India` and `--holdout-country US`): ⟨⟩ / ⟨⟩ — the drop
  relative to in-distribution validation is our estimate of the France risk.
* **Public leaderboard:** ⟨⟩
* *(synthetic smoke test: OOF 0.995; hidden synthetic test 0.992 with the unseen country "France" at 0.986)*

**Common false positives (wrong merges):** ⟨from `error_samples.txt`, e.g. same name stem at a different
address with the number missing on one side; same street, different business with a shared rare token⟩

**Common false negatives (missed matches):** ⟨e.g. typo inside the rarest name token plus a dropped
street number; heavy address reordering with landmark text; records where both name and address are
abbreviated⟩

Top features by gain in the final Stage-3 model: ⟨from run.log "top features"⟩.

---

## 6. Conclusion

Formulating the challenge as calibrated, per-entity set prediction let us make the blocking stage as small
as the metric allows, reason about singletons explicitly, and use the deduplicated nature of Source 1 as a
hard structural constraint. Transductive per-country statistics and relative features, validated with
leave-one-country-out training, are what carry the model to the unseen French records. Lessons learned:
⟨…⟩

---

## Appendix

### A. Code Artefacts

```
code/business_entity_resolution/
├── src/ber_pipeline.py          # the entire pipeline, STEP 0 … STEP 12 labelled in the file and the log
├── src/make_synthetic_data.py   # toy data generator used for smoke tests
├── utils/validate_submission.py # official validator, called automatically at the end of a run
├── README.md                    # local / Kaggle / Colab instructions
└── requirements.txt
```

Entry point (reproduces both output files, then runs the validator):

```
python src/ber_pipeline.py --data-dir dataset --out-dir output --validator utils/validate_submission.py
```

Additional outputs: `output/metrics.json` (all numbers quoted above), `output/run.log`,
`output/error_samples.txt`, `work/train_oof_pairs.parquet`, `work/test_scored_pairs.parquet`.
Deterministic given `--seed` (default 42). Models and libraries: LightGBM (MIT), RapidFuzz (MIT),
sparse-dot-topn (MIT), scikit-learn (BSD), optional `multilingual-e5-small` (MIT) and
`mdeberta-v3-base` (MIT) — all far below the 8B-parameter limit. No external data, API or service is
used at any point.

### B. Additional Results

* Prior work this design draws on: Fellegi & Sunter (1969) probabilistic record linkage; Christen,
  *Data Matching* (2012) for blocking metrics; Cohen, Ravikumar & Fienberg (2003) SoftTF-IDF;
  Paulsen et al., *Sparkly* (VLDB 2023) showing TF-IDF top-k blocking is competitive with deep blockers;
  Thirumuruganathan et al., *DeepBlocker* (VLDB 2021); Li et al., *Ditto* (VLDB 2021) for the
  cross-encoder serialisation; Nan et al. (ICML 2012), Jansche (2007), Dembczyński et al. (NeurIPS 2011)
  and Lipton et al. (2014) for F-measure-optimal decisions.
* ⟨Calibration plot of Stage-3 probabilities (isotonic, OOF)⟩
* ⟨Candidates-per-entity histogram before and after pruning⟩
* ⟨Per-country F₀.₅ and singleton accuracy⟩

---

**Note:** Teams can modify sections according to their approach while maintaining clarity and technical depth.
