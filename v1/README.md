# Business Entity Resolution — ML Challenge 2026

Single-file pipeline: `src/ber_pipeline.py`. It reproduces both
`output/candidate_pairs.tsv` and `output/matching_results.tsv` from the raw TSV files,
with no external data, APIs or services (only the provided train/test files are read).

## Expected layout

```
business_entity_resolution/
├── src/ber_pipeline.py
├── src/make_synthetic_data.py     # optional: builds a toy dataset for smoke tests
├── utils/validate_submission.py   # the official validator (copied from student_resource/utils)
├── dataset/train/train_source{1,2,3}.tsv, train_ground_truth.tsv
├── dataset/test/test_source{1,2,3}.tsv
├── output/                        # written by the pipeline
├── README.md
└── requirements.txt
```

## 1. Run locally

```bash
python3 -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# full reproduction (train -> out-of-fold validation -> test inference -> validator)
python src/ber_pipeline.py --data-dir dataset --out-dir output --validator utils/validate_submission.py

# fast development run on 10% of the S1 entities (NOT submittable: the validator will
# correctly complain that S1 rows are missing, because only a sample was processed)
python src/ber_pipeline.py --data-dir dataset --out-dir output_dev --sample 0.1

# leave-one-country-out check (simulates the unseen France domain): train on US, score India
python src/ber_pipeline.py --data-dir dataset --out-dir output_loco --holdout-country India --no-test
```

Smoke test without the real data:

```bash
python src/make_synthetic_data.py --out dataset_synth --n-train 3000 --n-test 2000
python src/ber_pipeline.py --data-dir dataset_synth --out-dir output_synth
```

## 2. Run on Kaggle

1. Create a Notebook, attach the challenge data as a Dataset (it will be mounted read-only under
   `/kaggle/input/<dataset-name>/`), and upload `ber_pipeline.py` + `validate_submission.py`
   either as a second Dataset or with the "Add data → Upload" button.
2. Turn on the GPU accelerator only if you plan to use `--use-dense` / `--use-cross-encoder`.
3. Run these cells:

```python
!pip install -q lightgbm rapidfuzz sparse-dot-topn pyarrow
# optional GPU extras:  !pip install -q sentence-transformers faiss-cpu sentencepiece

import shutil, os
os.makedirs("/kaggle/working/dataset", exist_ok=True)
shutil.copytree("/kaggle/input/<dataset-name>/dataset", "/kaggle/working/dataset", dirs_exist_ok=True)
shutil.copy("/kaggle/input/<code-dataset>/ber_pipeline.py", "/kaggle/working/")
shutil.copy("/kaggle/input/<code-dataset>/validate_submission.py", "/kaggle/working/")

!cd /kaggle/working && python ber_pipeline.py --data-dir dataset --out-dir output \
    --validator validate_submission.py
```

Outputs appear in `/kaggle/working/output/` and can be downloaded from the notebook's Output tab.
Kaggle CPU notebooks have ~30 GB RAM, which is enough for the full test set with the default
settings; if memory is tight lower `--k-sparse-name 10 --k-sparse-full 6`.

## 3. Run on Google Colab

```python
from google.colab import drive
drive.mount("/content/drive")          # put dataset/ , ber_pipeline.py, validate_submission.py in Drive

!pip install -q lightgbm rapidfuzz sparse-dot-topn pyarrow
# optional GPU extras (Runtime -> Change runtime type -> GPU):
# !pip install -q sentence-transformers faiss-cpu sentencepiece

%cd /content/drive/MyDrive/ber
!python ber_pipeline.py --data-dir dataset --out-dir output --validator validate_submission.py
```

Colab free tier has ~12 GB RAM: run with `--k-sparse-name 10 --k-sparse-full 6` on the full test set,
or develop with `--sample 0.2` and do the final run on Kaggle / a local machine.

## 4. Flags that matter

| flag | default | meaning |
|---|---|---|
| `--sample f` | 1.0 | keep a fraction of S1 entities (dev only) |
| `--holdout-country C` | – | leave-one-country-out validation instead of GroupKFold |
| `--no-test` | off | validation only |
| `--k-sparse-name / --k-sparse-full` | 15 / 10 | top-k per retrieval view |
| `--kmax` | 10 | hard cap on candidates per S1 entity after pruning |
| `--prune-target-recall` | 0.995 | recall of retrievable true pairs that Stage-2 pruning must retain |
| `--country-block auto/on/off` | auto | auto = on if ≥99% of GT pairs share a country |
| `--hard-assign auto/on/off` | auto | auto = on if ≥99.5% of S2/S3 GT records belong to one S1 entity |
| `--use-dense` | off | add multilingual-e5-small dense retrieval view (GPU recommended) |
| `--use-cross-encoder` | off | add fine-tuned mdeberta-v3-base cross-encoder feature (GPU required) |

## 5. What the run produces

* `output/candidate_pairs.tsv` — exactly the pairs the Stage-3 matcher scored.
* `output/matching_results.tsv` — final matches (always a subset of the candidates).
* `output/metrics.json` — blocking diagnostics (pairs completeness, reduction ratio, candidates per
  entity), out-of-fold macro-F0.5 overall and per country, chosen decision parameters.
* `output/error_samples.txt` — sample false merges and missed matches for the write-up.
* `output/run.log` — the full log, one block per STEP.
* `work/*.parquet` — scored pairs, for further analysis.

The validator is invoked automatically at the end (`PASS` is printed in the log).
