# Business Entity Resolution — ML Challenge 2026 (pipeline v2)

Single-file pipeline `src/ber_pipeline.py` (plus the equivalent notebook `ber_pipeline.ipynb`). It reproduces
both `output/candidate_pairs.tsv` and `output/matching_results.tsv` from the raw TSV files using only the
provided data — no external data, APIs or services.

## What is new in v2

| area | change |
|---|---|
| progress | `tqdm` bars on every long loop (text in a terminal, widgets in Jupyter/Kaggle/Colab); `--no-progress` turns them off |
| multi-worker | normalisation, vocabulary building, sparse transforms and char-n-gram hashing run in **worker processes** (joblib/loky); feature chunks run on a **thread pool** (RapidFuzz and scipy.sparse release the GIL); sparse top-k, LightGBM and RapidFuzz use native threads; expected-F decisions run in worker processes |
| speed (STEP 4 → end) | every per-pair Python loop was replaced by vectorised sparse-matrix or C++ code: IDF-weighted Jaccard, rarest shared/unshared token, containment, digit agreement and phonetic overlap are row-wise sparse products; exact blocking keys are built with pandas joins; labels, decisions, macro-F, entity features and output writing are array arithmetic; Stage-2 LightGBM trains on 3 folds with weighted negative subsampling |
| feature engineering | 14 named feature groups (~60 features) in a registry; exact TF-IDF cosines for **every** candidate (v1 stored 0 for pairs a view did not retrieve); new containment, phonetic, suffix, first-token, prefix, digit-Jaccard and record-size features; `feature_importance.csv` with per-group gain; `--drop-groups` for ablation |
| extra feature | **stage cache**: STEP 2–4 results are saved under `work/cache/`, keyed by a hash of the input files and every parameter they depend on, so re-runs (new features, new decision rule, ablations) skip normalisation, record-store construction and retrieval |
| alternative blocker | `--blocker lsh` — MinHash-LSH over character n-gram sets with bucket caps, re-scored by exact cosine; `--blocker both` takes the union |
| validation | leave-one-country-out now tunes calibration, pruning and the decision rule on the training pool only, never on the held-out country |

## Layout

```
business_entity_resolution/
├── src/ber_pipeline.py            # the pipeline, STEP 0 … STEP 12
├── src/make_synthetic_data.py     # toy data generator for smoke tests
├── ber_pipeline.ipynb             # same pipeline as a notebook (source embedded)
├── utils/validate_submission.py   # official validator (copied from student_resource/utils)
├── dataset/train/…  dataset/test/…
├── README.md
└── requirements.txt
```

## 1. Local (terminal)

```bash
python3 -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# full reproduction: train -> out-of-fold validation -> test inference -> outputs -> validator
python src/ber_pipeline.py --data-dir dataset --out-dir output --validator utils/validate_submission.py

# alternative blocker (MinHash-LSH), or the union of both blockers
python src/ber_pipeline.py --data-dir dataset --out-dir output_lsh --blocker lsh
python src/ber_pipeline.py --data-dir dataset --out-dir output_both --blocker both

# leave-one-country-out (the unseen-France risk estimate); validation only
python src/ber_pipeline.py --data-dir dataset --out-dir output_loco --holdout-country India --no-test

# feature-group ablation (reuses the stage cache, so only STEP 5 onwards re-runs)
python src/ber_pipeline.py --data-dir dataset --out-dir output_abl --drop-groups phonetic,collective

# smoke test without the real data
python src/make_synthetic_data.py --out dataset_synth --n-train 3000 --n-test 2000
python src/ber_pipeline.py --data-dir dataset_synth --out-dir output_synth
```

`--sample 0.1` gives a fast development run on 10 % of the S1 entities; its outputs are **not** submittable
(the validator correctly reports missing S1 rows).

To run the notebook locally: `pip install jupyter` then `jupyter lab ber_pipeline.ipynb` and *Run All*.

## 2. Kaggle

1. New Notebook → *File → Import Notebook* → upload `ber_pipeline.ipynb`.
2. *Add Input* → the challenge dataset, plus a small dataset containing `validate_submission.py`.
3. Accelerator: CPU is enough. Choose GPU only for `--use-dense` / `--use-cross-encoder`.
4. *Run All*. Cell 2 finds the data and the validator under `/kaggle/input` automatically; outputs go to
   `/kaggle/working/output/` (download from the Output tab). The last cell builds the submission zip.

Terminal-style alternative inside a Kaggle cell:
```python
!pip install -q lightgbm rapidfuzz sparse-dot-topn jellyfish tqdm pyarrow
!python /kaggle/input/<code-dataset>/ber_pipeline.py --data-dir /kaggle/input/<data-dataset>/dataset \
    --out-dir /kaggle/working/output --work-dir /kaggle/working/work --cache-dir /kaggle/working/work/cache \
    --validator /kaggle/input/<code-dataset>/validate_submission.py
```

## 3. Google Colab

1. *File → Upload notebook* → `ber_pipeline.ipynb`.
2. Either upload the `dataset/` folder and `validate_submission.py` into `/content`, or put them in Google Drive
   and set `MOUNT_DRIVE = True` in cell 2.
3. *Runtime → Run all*. On the free tier (~12 GB RAM) the notebook automatically lowers
   `k_sparse_name / k_sparse_full / chunk_size`. The cache lives in `/content/work/cache` and is lost when the
   runtime resets; point `cache_dir` at Drive to keep it.

## 4. Performance knobs

| flag | default | effect |
|---|---|---|
| `--n-jobs N` | all cores | processes for STEP 2–3 and decisions; threads for top-k, LightGBM, RapidFuzz |
| `--feature-threads N` | = n-jobs | threads over feature chunks (each chunk runs RapidFuzz single-threaded) |
| `--chunk-size N` | 200 000 | pairs per feature chunk; lower it if RAM is tight |
| `--neg-sample r` | 0.3 | fraction of Stage-2 negatives used for training (weights `1/r` keep calibration) |
| `--k-sparse-name / --k-sparse-full` | 15 / 10 | retrieval breadth — the main memory driver |
| `--no-cache` | cache on | disable the stage cache |
| `--no-progress` | bars on | disable progress bars (e.g. when writing logs to a file) |

Memory rule of thumb: raw candidate pairs ≈ `n_S1 × (k_name + k_full)`, about 150 bytes each during STEP 5.

## 5. Other flags

| flag | default | meaning |
|---|---|---|
| `--blocker` | tfidf | `tfidf` (exact sparse top-k) · `lsh` (MinHash-LSH) · `both` |
| `--lsh-bands / --lsh-rows / --k-lsh` | 12 / 4 / 20 | LSH S-curve (threshold ≈ (1/b)^(1/r) ≈ 0.54) and per-entity cap after re-scoring |
| `--kmax` | 10 | cap on candidates per S1 after pruning (auto-raised to largest GT cluster + 5) |
| `--prune-target-recall` | 0.995 | recall of retrievable true pairs Stage-2 pruning must keep |
| `--drop-groups` | – | ablate feature groups (names in `FEATURE_GROUPS`) |
| `--country-block / --hard-assign` | auto | auto = decided from the training ground truth (see the `EDA:` line in `run.log`) |
| `--use-dense / --use-cross-encoder` | off | multilingual-e5-small dense view / mdeberta-v3-base cross-encoder (GPU; untested in CPU-only CI) |

## 6. Outputs

* `output/candidate_pairs.tsv` — exactly the pairs the Stage-3 matcher scored.
* `output/matching_results.tsv` — final matches (always a subset of the candidates).
* `output/metrics.json` — blocking diagnostics per stage and per retrieval view, OOF macro-F0.5 overall and per
  country, chosen decision rule, feature-group gain shares, per-step timings.
* `output/feature_importance.csv`, `output/error_samples.txt`, `output/run.log`.
* `work/*.parquet` — scored pairs; `work/cache/` — stage cache.

The validator runs automatically at the end (`PASS` in the log).
