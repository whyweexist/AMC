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
* **Source 1 is deduplicated, Sources 2 and 3 are not.** The ground truth shows one S1 entity owning several
  S2 *and* several S3 records (e.g. 2 + 3), so the matches of an entity arrive as a cluster of near-duplicates
  of each other. Each S2/S3 record belongs to at most one S1 entity, but matches of the same entity are
  strongly *positively correlated* — transitivity is real evidence and is used as a feature.

Data-driven checks performed automatically by the code at start-up (`eda_decisions`, see `run.log`):

| check | value on training data | consequence |
|---|---|---|
| share of GT pairs with equal country label | ⟨country agreement⟩ | country used as a hard block key if ≥ 0.99 |
| share of S2/S3 GT records mapped to exactly one S1 entity | ⟨uniqueness⟩ | hard one-to-one assignment if ≥ 0.995 |
| singleton rate | ⟨n_single / n_S1⟩ | prior of the singleton model |
| match-count distribution (0/1/2/…) | ⟨dist⟩ | `kmax = max(10, max matches + 5)` |

Noise patterns observed (fill from `output/error_samples.txt` and manual inspection):
⟨abbreviations / suffix swaps / typos / reordering / landmark phrases / missing postal codes …⟩

### 2.2 Solution Strategy

**Approach Type:** Blocking (exact sparse top-k, with MinHash-LSH as a scalable alternative) + learned pruning + gradient-boosted classifier + decision-theoretic set selection (Hybrid).

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

Blocking is a two-stage funnel. Stage 1 retrieves broadly with cheap, complementary views; Stage 2 is a
calibrated classifier that prunes to the smallest set the final metric allows. **The Stage-2 output is written
verbatim to `candidate_pairs.tsv` and is exactly what the Stage-3 matcher scores.**

### 3.1 Record representation (STEP 3)

Every record is converted once into sparse matrices, built in parallel worker processes:

* character 3–4-gram TF-IDF (hashed to 2²⁰ columns, sublinear tf, L2-normalised) of the name and of name+address;
* binary and IDF-weighted word-token matrices for the core name, core address, landmark phrase and all tokens,
  with **per-country transductive IDF** `idf_c(t) = log((N_c+1)/(df_c(t)+1)) + 1` computed on the records of
  country *c* in the current split — so French high-frequency tokens (*rue, sarl, de*) are down-weighted
  automatically without any hand-written list;
* binary matrices of postal-like digit runs (≥ 4 digits), short street numbers, and metaphone phonetic keys.

### 3.2 Primary blocker — exact sparse top-k (`--blocker tfidf`, default)

Inside each country block, the k most similar S2/S3 records per S1 entity are retrieved by exact cosine
(k = 15 on the name view, 10 on the name+address view) with `sparse_dot_topn` (multi-threaded C++):
`cos(a, b) = ⟨x_a, x_b⟩` for L2-normalised TF-IDF rows. Cost per block is O(Σ_t df_t²) and embarrassingly
parallel. This follows Paulsen et al., *Sparkly* (VLDB 2023), who show well-built TF-IDF top-k blocking matches
or beats learned deep blockers.

Exact keys are added with vectorised hash joins: (postal-like code, 3-char prefix of the rarest core-name token),
(name acronym, street number), and the first 8 characters of the space-free core name. Keys shared by more than
100 target records are dropped. If an S1 country label has no S2/S3 record with the same label (possible for the
unseen country), that group falls back to retrieval over all targets instead of returning nothing.

### 3.3 Alternative blocker — MinHash-LSH (`--blocker lsh`)

For each record's set of character n-grams we compute b·r = 48 MinHash values
`h_k(S) = min_{x∈S} ((a_k·x + b_k) mod p)`. For a random hash, P[h_k(S₁) = h_k(S₂)] = J(S₁, S₂), the Jaccard
similarity (Broder, 1997). The signature is split into b = 12 bands of r = 4 values; two records become
candidates when any band matches exactly:

  P(candidate | J) = 1 − (1 − Jʳ)ᵇ

an S-curve with threshold ≈ (1/b)^(1/r) ≈ 0.54 (J = 0.8 → ≈ 0.9999; J = 0.5 → ≈ 0.54; J = 0.3 → ≈ 0.09).
Buckets holding more than 200 targets are skipped (bounds the cost of very common names), and surviving pairs are
re-scored by exact cosine and capped at 20 per S1 entity. Cost is linear, O(n·b·r) hashing plus hash joins, versus
the quadratic worst case of exact top-k within a large block — this is the production choice at billion-record
scale. `--blocker both` takes the union of both blockers; since Stage-2 pruning compresses either input to a few
candidates per entity, the wider union costs almost nothing in the final candidate set.

| blocker (synthetic smoke test, 3 000 S1) | raw candidates / S1 | recall ceiling |
|---|---|---|
| tfidf + keys | 28.2 | 0.9995 |
| lsh + keys | 26.6 | 0.9959 |
| both | 34.8 | 1.0000 |

Per-view recall on the real data is reported in `metrics.json` → `blocking_raw.recall_found_*`: ⟨fill in⟩.

### 3.4 Learned pruning (Stage 2)

About 30 cheap features are extracted for every retrieved pair (exact cosines for all views, IDF-weighted name
Jaccard, rarest shared / unshared token, containment, digit agreement, phonetic overlap, missingness, and the
pair's rank and margin within its entity). A LightGBM model is trained out-of-fold (3 folds grouped by S1
entity; negatives subsampled at rate r with weight 1/r, which preserves the implied class prior) and calibrated with
isotonic regression, giving p₁.

A pair survives if p₁ ≥ ε and it is in the top k_max of its entity (k_max = largest ground-truth cluster + 5).
ε is the largest value keeping ≥ 99.5 % of retrievable true pairs on out-of-fold predictions. The expected
number of true matches lost is Σ_pruned p₁ ≤ |pruned|·ε (logged). Pruning is nearly free in F₀.₅ because the
decision layer only ever outputs a pair whose probability exceeds ≈ F*/(1+β²) = 0.8·F*.

### 3.5 Candidate pairs generated

| stage | pairs | avg. per S1 entity | reduction ratio | pairs completeness (recall ceiling) |
|---|---|---|---|---|
| raw retrieval (train, OOF) | ⟨⟩ | ⟨⟩ | ⟨⟩ | ⟨⟩ |
| after Stage-2 pruning (train, OOF) | ⟨⟩ | ⟨⟩ | ⟨⟩ | ⟨⟩ |
| raw retrieval (test) | ⟨⟩ | ⟨⟩ | ⟨⟩ | – |
| **`candidate_pairs.tsv` (test)** | ⟨⟩ | ⟨⟩ | ⟨⟩ | – |

*(synthetic smoke test, tfidf blocker: 28.2 raw candidates/entity at recall 0.9995 → 1.51/entity at recall 0.994)*

**How true matches were not lost:** complementary views (character-level, address-aware, exact keys, optionally
LSH and dense), so a typo, abbreviation or reordering must defeat all of them at once; the pruning threshold is set
against an explicit recall constraint on out-of-fold predictions; the per-entity cap is derived from the largest
ground-truth cluster; and recall is measured at every stage and per view.

---

## 4. Matching Model

**Features used.** ~60 features in 14 named groups (registry `FEATURE_GROUPS` in the code). All are country- and
language-agnostic — no country one-hot anywhere. Token, digit and phonetic features are computed as row-wise
sparse-matrix products; string features with RapidFuzz (C++); extraction runs in chunks on a thread pool.

| group | features |
|---|---|
| retrieval | exact char-TF-IDF cosine of name and of name+address for **every** pair, which view(s) found it, number of views |
| name_token | IDF-weighted Jaccard of core-name tokens; IDF of the rarest *shared* and rarest *unshared* token and their difference; containment of each name's weight inside the other record (DBA / field-swap signal) |
| name_string | token-set, token-sort and partial ratios; Jaro–Winkler; normalised Levenshtein; Jaro–Winkler and common-prefix similarity without spaces; exact core equality; length ratio; token-count difference |
| name_structure | acronym match, legal-suffix equality and conflict, first-token equality |
| phonetic | Jaccard of metaphone keys, first-key equality |
| address_token | IDF-weighted Jaccard of core-address, landmark and all tokens; rarest shared address token |
| address_string | token-set ratio, normalised Levenshtein, partial ratio of the landmark-stripped address |
| numeric | three-valued agreement (agree / missing / conflict) of postal-like codes and street numbers, shared counts, code Jaccard |
| missingness | address present on each side, number of codes on each side, name lengths, candidate source (S2/S3) |
| context | number of raw candidates, within-entity rank and margin of both cosines |
| stage2 | calibrated Stage-2 probability p₁, its rank and margin within the entity |
| competition | rank and margin of this S1 entity among all S1 entities competing for the same S2/S3 record, number of competitors, mutual-best flag |
| collective | similarity of the candidate to the entity's best candidate (runner-up for the best one), that reference's probability, their product, number of confident siblings — recovers the 2nd…5th duplicate of a cluster once the first is anchored |
| cross_encoder (optional) | out-of-fold logit of a Ditto-style `mdeberta-v3-base` cross-encoder on serialised pairs |

Gain share by group in the final model: ⟨from `metrics.json` → `feature_group_gain_share`⟩.
Top features: ⟨from `feature_importance.csv`⟩. Ablations (`--drop-groups`): ⟨⟩.

**Model type:** LightGBM (MIT) binary classifier, 800 rounds, 63 leaves, trained out-of-fold with 5 folds grouped
by S1 entity on the pruned candidate distribution (hard negatives only), then isotonic-calibrated. Structural
post-processing: each S2/S3 record's probabilities across competing S1 entities are normalised so that
Σᵢ pᵢⱼ ≤ 1, and when the training data shows one-to-one ownership only the argmax S1 entity keeps its probability.
In leave-one-country-out mode, calibration, pruning threshold and decision rule are fitted on the training countries
only and scored on the held-out one.

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
* *(synthetic smoke test: OOF 0.992; hidden synthetic test 0.997 with the unseen country "France" at 0.995)*

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
├── src/ber_pipeline.py          # the entire pipeline, STEP 0 … STEP 12 labelled in the file, the log and the progress bars
├── src/make_synthetic_data.py   # toy data generator used for smoke tests
├── ber_pipeline.ipynb           # the same pipeline as a notebook for Kaggle / Colab / Jupyter (source embedded)
├── utils/validate_submission.py # official validator, called automatically at the end of a run
├── README.md                    # local / Kaggle / Colab instructions and every flag
└── requirements.txt             # pinned versions
```

Entry point (reproduces both output files, then runs the validator):

```
python src/ber_pipeline.py --data-dir dataset --out-dir output --validator utils/validate_submission.py
```

Engineering: progress bars on every long loop; normalisation, vectorisation and expected-F decisions run in
worker processes, feature extraction on a thread pool, top-k search / LightGBM / RapidFuzz on native threads
(`--n-jobs`); a stage cache under `work/cache/` makes re-runs skip STEP 2–4. Alternative blocker:
`--blocker lsh` or `--blocker both`. Additional outputs: `output/metrics.json` (all numbers quoted above,
including per-step timings), `output/feature_importance.csv`, `output/run.log`, `output/error_samples.txt`,
`work/*.parquet`. Deterministic given `--seed` (default 42). Models and libraries: LightGBM (MIT), RapidFuzz (MIT),
sparse-dot-topn (MIT), scikit-learn (BSD), jellyfish (MIT), tqdm (MPL-2.0/MIT), optional `multilingual-e5-small`
(MIT) and `mdeberta-v3-base` (MIT) — all far below the 8B-parameter limit. No external data, API or service is used.

### B. Additional Results

* Prior work this design draws on: Fellegi & Sunter (1969) probabilistic record linkage; Christen,
  *Data Matching* (2012) for blocking metrics; Cohen, Ravikumar & Fienberg (2003) SoftTF-IDF;
  Paulsen et al., *Sparkly* (VLDB 2023) showing TF-IDF top-k blocking is competitive with deep blockers;
  Broder (1997) and Indyk & Motwani (1998) for MinHash and locality-sensitive hashing;
  Thirumuruganathan et al., *DeepBlocker* (VLDB 2021); Li et al., *Ditto* (VLDB 2021) for the
  cross-encoder serialisation; Nan et al. (ICML 2012), Jansche (2007), Dembczyński et al. (NeurIPS 2011)
  and Lipton et al. (2014) for F-measure-optimal decisions.
* ⟨Calibration plot of Stage-3 probabilities (isotonic, OOF)⟩
* ⟨Candidates-per-entity histogram before and after pruning⟩
* ⟨Per-country F₀.₅ and singleton accuracy⟩

---

**Note:** Teams can modify sections according to their approach while maintaining clarity and technical depth.
