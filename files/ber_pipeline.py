#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ber_pipeline.py — Business Entity Resolution (ML Challenge 2026), single-file pipeline, v2.

Stages (labelled "STEP n" in the code, the log and the progress bars):
  STEP 0  Config, environment, logging, progress bars, parallel helpers, stage cache
  STEP 1  Load TSV data
  STEP 2  Normalisation (parallel over record chunks)
  STEP 3  Record store: sparse word / digit / phonetic / char-n-gram matrices + per-country IDF
  STEP 4  Candidate retrieval — blocker "tfidf" (exact sparse top-k, default), "lsh" (MinHash-LSH
          alternative) or "both"; plus exact keys and an optional dense view
  STEP 5  Vectorised feature extraction + Stage-2 learned pruning  ->  candidate_pairs.tsv
  STEP 6  Full feature extraction (14 feature groups) + competition + collective features
  STEP 7  Stage-3 matcher (LightGBM, out-of-fold, isotonic calibration)
  STEP 8  Structural constraints (each S2/S3 record -> at most one S1 entity)
  STEP 9  Singleton model + vectorised / parallel expected-F0.5 decision
  STEP 10 Evaluation (macro F0.5, per country, blocking diagnostics, feature importance)
  STEP 11 Write outputs and run the official validator
  STEP 12 CLI entry point

Quick start:
  python ber_pipeline.py --data-dir dataset --out-dir output                   # full run
  python ber_pipeline.py --data-dir dataset --out-dir output --blocker lsh     # alternative blocker
  python ber_pipeline.py --data-dir dataset --holdout-country India --no-test  # leave-one-country-out

Only MIT/BSD/Apache/MPL libraries are used. No external data or services are ever contacted.
"""

# =============================================================================
# STEP 0 — imports, configuration, environment, logging, progress, parallelism, cache
# =============================================================================
import argparse
import hashlib
import json
import logging
import math
import os
import re
import subprocess
import sys
import time
import unicodedata
import warnings
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
from joblib import Parallel, delayed, dump, load
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer, TfidfTransformer
from sklearn.isotonic import IsotonicRegression

import lightgbm as lgb
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler, Levenshtein, Prefix

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None
try:
    import jellyfish
    HAS_JF = True
except Exception:  # pragma: no cover
    HAS_JF = False
try:
    from sparse_dot_topn import sp_matmul_topn
    HAS_SDT = True
except Exception:  # pragma: no cover
    HAS_SDT = False

warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")
LOG = logging.getLogger("ber")
_PROGRESS = True
TIMINGS: Dict[str, float] = {}


def detect_environment() -> str:
    if os.path.exists("/kaggle/input") or "KAGGLE_KERNEL_RUN_TYPE" in os.environ:
        return "kaggle"
    if "google.colab" in sys.modules or os.path.exists("/content"):
        return "colab"
    return "local"


def has_gpu() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


@dataclass
class Config:
    data_dir: str = "dataset"
    out_dir: str = "output"
    work_dir: str = "work"
    seed: int = 42
    sample: float = 1.0                    # fraction of S1 entities to keep (dev runs only)
    n_folds: int = 5
    holdout_country: Optional[str] = None  # leave-one-country-out validation
    run_test: bool = True

    # parallelism / speed
    n_jobs: int = 0                        # processes & library threads; 0 = all cores
    feature_threads: int = 0               # threads over feature chunks; 0 = n_jobs
    chunk_size: int = 200_000              # pairs per feature-extraction chunk
    progress: bool = True
    use_cache: bool = True                 # the stage cache (STEP 2-4 are reused across runs)
    cache_dir: str = "work/cache"

    # STEP 3
    ngram_range: Tuple[int, int] = (3, 4)
    hash_bits: int = 20

    # STEP 4 retrieval
    blocker: str = "tfidf"                 # tfidf | lsh | both
    k_sparse_name: int = 15
    k_sparse_full: int = 10
    sparse_threshold: float = 0.15
    max_key_block: int = 100
    lsh_bands: int = 12
    lsh_rows: int = 4
    lsh_bucket_cap: int = 200
    k_lsh: int = 20
    use_dense: bool = False
    dense_model: str = "intfloat/multilingual-e5-small"   # MIT
    k_dense: int = 10
    query_chunk: int = 20000

    # STEP 5 pruning
    prune_target_recall: float = 0.995
    prune_eps_floor: float = 0.005
    kmax: int = 10
    neg_sample: float = 0.3                # negative subsampling for Stage-2 training (weights keep calibration)
    stage2_folds: int = 3
    lgb_rounds_stage2: int = 300

    # STEP 6/7 matcher
    use_cross_encoder: bool = False
    ce_model: str = "microsoft/mdeberta-v3-base"          # MIT
    ce_max_train_pairs: int = 150000
    ce_epochs: int = 1
    lgb_rounds_stage3: int = 800
    drop_groups: Tuple[str, ...] = ()      # feature-group ablation, e.g. ("phonetic", "collective")

    # STEP 8 structure (auto from EDA unless forced)
    hard_assign: Optional[bool] = None
    country_block: Optional[bool] = None

    # STEP 9 decision
    beta: float = 0.5
    temperature_grid: Tuple[float, ...] = (0.7, 0.85, 1.0, 1.2, 1.5)
    singleton_scale_grid: Tuple[float, ...] = (0.7, 0.85, 1.0, 1.15, 1.3)
    threshold_grid: Tuple[float, ...] = (0.4, 0.5, 0.6, 0.7, 0.8)
    tune_max_entities: int = 40000

    validator: str = "utils/validate_submission.py"

    def __post_init__(self):
        if self.n_jobs <= 0:
            self.n_jobs = os.cpu_count() or 1
        if self.feature_threads <= 0:
            self.feature_threads = self.n_jobs
        for d in (self.out_dir, self.work_dir, self.cache_dir):
            os.makedirs(d, exist_ok=True)


def setup_logging(out_dir: str):
    fmt = "%(asctime)s | %(levelname)s | %(message)s"
    logging.basicConfig(level=logging.INFO, format=fmt, force=True,
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(os.path.join(out_dir, "run.log"), mode="w", encoding="utf-8")])


class Timer:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.t = time.time()
        LOG.info(f"===== {self.name} =====")
        return self

    def __exit__(self, *a):
        dt = time.time() - self.t
        TIMINGS[self.name] = TIMINGS.get(self.name, 0.0) + dt
        LOG.info(f"----- {self.name} done in {dt:.1f}s")


class _NullBar:
    def __init__(self, it=None):
        self.it = it

    def __iter__(self):
        return iter(self.it if self.it is not None else [])

    def update(self, n=1):
        pass

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        pass


def pbar(it=None, total=None, desc="", unit="it"):
    """Progress bar (tqdm.auto: text bar in a terminal, widget in Jupyter/Kaggle/Colab)."""
    if tqdm is None or not _PROGRESS:
        return _NullBar(it)
    return tqdm(it, total=total, desc=desc, unit=unit, dynamic_ncols=True, leave=False, mininterval=0.5)


def split_chunks(n: int, size: int) -> List[Tuple[int, int]]:
    size = max(1, int(size))
    return [(s, min(n, s + size)) for s in range(0, n, size)]


def split_even(n: int, parts: int) -> List[Tuple[int, int]]:
    return split_chunks(n, max(1, math.ceil(n / max(1, parts))))


def parallel_map(fn, items, n_jobs: int, desc: str = ""):
    """Multi-process map (joblib/loky) with an ordered progress bar. fn must be a top-level function."""
    items = list(items)
    if n_jobs <= 1 or len(items) <= 1:
        return [fn(x) for x in pbar(items, total=len(items), desc=desc)]
    gen = Parallel(n_jobs=min(n_jobs, len(items)), backend="loky", return_as="generator")(delayed(fn)(x) for x in items)
    return list(pbar(gen, total=len(items), desc=desc))


def thread_map(fn, items, n_threads: int, desc: str = ""):
    """Multi-thread map for work that releases the GIL (RapidFuzz, scipy.sparse, numpy)."""
    items = list(items)
    if n_threads <= 1 or len(items) <= 1:
        return [fn(x) for x in pbar(items, total=len(items), desc=desc)]
    with ThreadPoolExecutor(max_workers=min(n_threads, len(items))) as ex:
        return list(pbar(ex.map(fn, items), total=len(items), desc=desc))


class StageCache:
    """The extra feature of v2: stage-level caching. Normalisation, record stores and retrieval results are
       stored under cache_dir keyed by a hash of the input files and every parameter they depend on, so
       re-running the pipeline (e.g. to try another decision rule or feature group) skips STEP 2-4."""

    def __init__(self, cfg: Config):
        self.on = cfg.use_cache
        self.dir = cfg.cache_dir

    @staticmethod
    def key(*parts) -> str:
        return hashlib.md5(json.dumps(parts, default=str, sort_keys=True).encode()).hexdigest()[:16]

    def get_or(self, name: str, key: str, fn):
        if not self.on:
            return fn()
        path = os.path.join(self.dir, f"{name}-{key}.joblib")
        if os.path.exists(path):
            LOG.info(f"cache hit: {path}")
            return load(path)
        obj = fn()
        dump(obj, path)
        LOG.info(f"cache write: {path}")
        return obj


def data_fingerprint(cfg: Config, split: str) -> List:
    d = os.path.join(cfg.data_dir, split)
    fp = []
    for f in sorted(os.listdir(d)) if os.path.isdir(d) else []:
        st = os.stat(os.path.join(d, f))
        fp.append((f, st.st_size, int(st.st_mtime)))
    return fp


# =============================================================================
# STEP 1 — load data
# =============================================================================
def read_tsv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, encoding="utf-8")
    df.columns = [c.strip() for c in df.columns]
    return df


def load_split(cfg: Config, split: str) -> Dict[str, pd.DataFrame]:
    d = os.path.join(cfg.data_dir, split)
    out = {}
    for s in (1, 2, 3):
        df = read_tsv(os.path.join(d, f"{split}_source{s}.tsv"))
        for c in ("entity_id", "business_name", "business_address", "country"):
            if c not in df.columns:
                df[c] = ""
        df["entity_id"] = df["entity_id"].str.strip()
        df["source"] = s
        out[f"s{s}"] = df
    gt_path = os.path.join(d, f"{split}_ground_truth.tsv")
    hidden = os.path.join(d, f"_hidden_{split}_ground_truth.tsv")   # synthetic smoke tests only
    gt_path = gt_path if os.path.exists(gt_path) else hidden
    out["gt"] = read_tsv(gt_path) if os.path.exists(gt_path) else None
    if cfg.sample < 1.0:
        rs = np.random.RandomState(cfg.seed)
        keep = out["s1"].sample(frac=cfg.sample, random_state=rs)["entity_id"]
        out["s1"] = out["s1"][out["s1"].entity_id.isin(keep)].reset_index(drop=True)
        if out["gt"] is not None:
            out["gt"] = out["gt"][out["gt"].source1_entity_id.str.strip().isin(keep)].reset_index(drop=True)
            matched = set()
            for v in out["gt"].matched_entity_ids:
                matched.update(x.strip() for x in v.split(",") if x.strip())
            for s in ("s2", "s3"):
                df = out[s]
                m = df.entity_id.isin(matched)
                out[s] = pd.concat([df[m], df[~m].sample(frac=cfg.sample, random_state=rs)]).reset_index(drop=True)
        else:
            for s in ("s2", "s3"):
                out[s] = out[s].sample(frac=cfg.sample, random_state=rs).reset_index(drop=True)
    LOG.info(f"[{split}] S1={len(out['s1'])} S2={len(out['s2'])} S3={len(out['s3'])} "
             f"GT={'yes' if out['gt'] is not None else 'no'}")
    return out


def gt_to_dict(gt: Optional[pd.DataFrame]) -> Dict[str, set]:
    if gt is None:
        return {}
    return {r.source1_entity_id.strip(): set(x.strip() for x in r.matched_entity_ids.split(",") if x.strip())
            for r in gt.itertuples()}


# =============================================================================
# STEP 2 — normalisation
# =============================================================================
_WS = re.compile(r"\s+")
_NONALNUM = re.compile(r"[^0-9a-z ]+")
_DIGITS = re.compile(r"\d+")
LANDMARK_CUES = ("near", "nr", "opp", "opposite", "behind", "beside", "next to", "adjacent to",
                 "pres de", "face a", "a cote de", "en face")
LEGAL_SUFFIXES = {"ltd", "limited", "pvt", "private", "inc", "incorporated", "corp", "corporation", "co",
                  "company", "llc", "llp", "plc", "sarl", "sas", "sasu", "sa", "eurl", "sci", "snc", "gmbh", "ag",
                  "bv", "nv", "the", "and", "sons", "l", "c", "p", "s", "a", "r", "e", "u"}
STATIC_ABBREV = {  # ordinary domain knowledge, not an external lookup
    "st": "street", "rd": "road", "ave": "avenue", "av": "avenue", "blvd": "boulevard", "bd": "boulevard",
    "dr": "drive", "ln": "lane", "hwy": "highway", "pl": "place", "sq": "square", "ct": "court", "pkwy": "parkway",
    "mkt": "market", "nr": "near", "opp": "opposite", "bldg": "building", "apt": "apartment", "fl": "floor",
    "ste": "suite", "pvt": "private", "ltd": "limited", "corp": "corporation", "inc": "incorporated",
    "co": "company", "intl": "international", "mfg": "manufacturing", "svc": "service", "svcs": "services",
    "ind": "industries", "ents": "enterprises", "bros": "brothers", "ctr": "center", "ctre": "centre",
    "mgmt": "management", "assoc": "associates", "tech": "technologies",
}


def norm_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.casefold().replace("&", " and ").replace("@", " at ").replace("'", "")
    s = _NONALNUM.sub(" ", s)
    return _WS.sub(" ", s).strip()


def mine_abbreviations(s1: pd.DataFrame, s23: pd.DataFrame, gt: Dict[str, set], min_count=5) -> Dict[str, str]:
    """Learn short->long token rewrites from matched pairs (train only)."""
    if not gt:
        return {}
    n1 = dict(zip(s1.entity_id, (s1.business_name + " " + s1.business_address).map(norm_text)))
    n2 = dict(zip(s23.entity_id, (s23.business_name + " " + s23.business_address).map(norm_text)))
    cnt = Counter()
    for a, ms in gt.items():
        ta = set(n1.get(a, "").split())
        for m in ms:
            tb = set(n2.get(m, "").split())
            da, db = sorted(ta - tb), sorted(tb - ta)
            if 0 < len(da) <= 2 and 0 < len(db) <= 2:
                for x in da:
                    for y in db:
                        if x.isdigit() or y.isdigit():
                            continue
                        short, long_ = (x, y) if len(x) < len(y) else (y, x)
                        if 2 <= len(short) <= 5 and len(long_) > len(short) and (
                                long_.startswith(short) or JaroWinkler.similarity(short, long_) > 0.85):
                            cnt[(short, long_)] += 1
    mined = {}
    for (s, l), c in cnt.most_common():
        if c >= min_count and s not in mined:
            mined[s] = l
    LOG.info(f"mined {len(mined)} abbreviation rewrites, e.g. {list(mined.items())[:8]}")
    return mined


def expand_abbrev(text: str, table: Dict[str, str]) -> str:
    return " ".join(table.get(t, t) for t in text.split())


def split_landmark(addr: str) -> Tuple[str, str]:
    for cue in LANDMARK_CUES:
        i = addr.find(" " + cue + " ")
        if i == -1 and addr.startswith(cue + " "):
            i = 0
        if i != -1:
            after = addr[i:].split()
            return (addr[:i] + " " + " ".join(after[4:])).strip(), " ".join(after[:4])
    return addr, ""


def phonetic(tok: str) -> str:
    if HAS_JF:
        try:
            k = jellyfish.metaphone(tok).replace(" ", "")
            if k:
                return k
        except Exception:
            pass
    return (tok[:1] + re.sub(r"[aeiouy]", "", tok[1:]))[:4]


def normalise_records(args) -> Dict[str, np.ndarray]:
    """Top-level (picklable) so it can run in worker processes. Returns canonical columns as numpy arrays."""
    df, abbrev = args
    name_n = [expand_abbrev(norm_text(x), abbrev) for x in df["business_name"].tolist()]
    addr_n = [expand_abbrev(norm_text(x), abbrev) for x in df["business_address"].tolist()]
    core, suf, acr, nospace, first, phon, phon_first = [], [], [], [], [], [], []
    for n in name_n:
        t = n.split()
        c = [x for x in t if x not in LEGAL_SUFFIXES] or t
        core.append(" ".join(c))
        suf.append(" ".join(x for x in t if x in LEGAL_SUFFIXES and len(x) > 1))
        acr.append("".join(x[0] for x in c if not x.isdigit()))
        nospace.append("".join(c))
        first.append(c[0] if c else "")
        ph = [phonetic(x) for x in c if len(x) >= 2 and not x.isdigit()]
        phon.append(" ".join(ph))
        phon_first.append(ph[0] if ph else "")
    acore, lm, codes, nums = [], [], [], []
    for a in addr_n:
        c, l = split_landmark(a)
        acore.append(c)
        lm.append(l)
        d = _DIGITS.findall(a)
        codes.append(" ".join(sorted(set(x for x in d if len(x) >= 4))))
        nums.append(" ".join(sorted(set((x.lstrip("0") or "0") for x in d if len(x) < 4))))
    obj = lambda x: np.array(x, dtype=object)  # noqa: E731
    return {
        "entity_id": df["entity_id"].to_numpy(dtype=object), "source": df["source"].to_numpy(dtype=np.int8),
        "country_n": obj([norm_text(x) or "unknown" for x in df["country"].tolist()]),
        "raw_name": df["business_name"].to_numpy(dtype=object), "raw_addr": df["business_address"].to_numpy(dtype=object),
        "name_n": obj(name_n), "addr_n": obj(addr_n), "core": obj(core), "suffix": obj(suf), "acr": obj(acr),
        "core_nospace": obj(nospace), "first_tok": obj(first), "phon": obj(phon), "phon_first": obj(phon_first),
        "addr_core": obj(acore), "landmark": obj(lm), "codes": obj(codes), "nums": obj(nums),
        "full_n": obj([f"{a} {b}".strip() for a, b in zip(name_n, addr_n)]),
    }


def normalise_frame(df: pd.DataFrame, abbrev: Dict[str, str], cfg: Config, desc: str) -> Dict[str, np.ndarray]:
    tasks = [(df.iloc[s:e], abbrev) for s, e in split_even(len(df), cfg.n_jobs * 2)] or [(df, abbrev)]
    parts = parallel_map(normalise_records, tasks, cfg.n_jobs, desc=desc)
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


# =============================================================================
# STEP 3 — record store: sparse matrices + per-country transductive IDF
# =============================================================================
@dataclass
class Store:
    n: int
    cols: Dict[str, np.ndarray]
    mats: Dict[str, sp.csr_matrix]
    vec: Dict[str, np.ndarray]


def _vocab_chunk(texts) -> set:
    s = set()
    for t in texts:
        s.update(t.split())
    return s


def _count_chunk(args) -> sp.csr_matrix:
    texts, vocab = args
    cv = CountVectorizer(vocabulary=vocab, tokenizer=str.split, token_pattern=None, lowercase=False,
                         binary=True, dtype=np.float32)
    return cv.transform(texts)


def _hash_chunk(args) -> sp.csr_matrix:
    texts, ngram, bits = args
    hv = HashingVectorizer(analyzer="char_wb", ngram_range=ngram, n_features=2 ** bits, alternate_sign=False,
                           norm=None, dtype=np.float32)
    return hv.transform(texts)


def token_matrices(texts: Dict[str, np.ndarray], vocab_keys: List[str], cfg: Config, desc: str):
    """Shared vocabulary built in parallel, then parallel sparse binary transforms."""
    tasks = [texts[k][s:e] for k in vocab_keys for s, e in split_even(len(texts[k]), cfg.n_jobs * 2)]
    sets = parallel_map(_vocab_chunk, tasks, cfg.n_jobs, desc=f"{desc}: vocabulary")
    toks = sorted(set().union(*sets)) or ["\x00"]
    vocab = {t: i for i, t in enumerate(toks)}
    mats = {}
    for k, arr in texts.items():
        tasks = [(arr[s:e], vocab) for s, e in split_even(len(arr), cfg.n_jobs * 2)]
        mats[k] = sp.vstack(parallel_map(_count_chunk, tasks, cfg.n_jobs, desc=f"{desc}: {k}")).tocsr().astype(np.float32)
    return np.array(toks, dtype=object), mats


def char_tfidf(texts: np.ndarray, cfg: Config, desc: str) -> sp.csr_matrix:
    """Stateless hashing (parallel, no fit pass) + sublinear TF-IDF + L2 normalisation."""
    tasks = [(texts[s:e], cfg.ngram_range, cfg.hash_bits) for s, e in split_even(len(texts), cfg.n_jobs * 2)]
    X = sp.vstack(parallel_map(_hash_chunk, tasks, cfg.n_jobs, desc=desc)).tocsr()
    return TfidfTransformer(sublinear_tf=True).fit_transform(X).astype(np.float32).tocsr()


def per_country_idf(all_b: sp.csr_matrix, ccode: np.ndarray, n_c: int) -> np.ndarray:
    """idf_c(t) = log((N_c+1)/(df_c(t)+1)) + 1 on ALL records of country c in the current split (transductive)."""
    idf = np.ones((n_c, all_b.shape[1]), np.float32)
    for c in range(n_c):
        rows = np.where(ccode == c)[0]
        df = np.asarray(all_b[rows].sum(axis=0)).ravel()
        idf[c] = np.log((len(rows) + 1) / (df + 1)) + 1
    return idf


def weighted(M: sp.csr_matrix, ccode: np.ndarray, idf: np.ndarray) -> sp.csr_matrix:
    W = M.copy()
    rows = np.repeat(np.arange(M.shape[0]), np.diff(M.indptr))
    W.data = idf[ccode[rows], M.indices].astype(np.float32)
    return W


def _rowsum(M) -> np.ndarray:
    return np.asarray(M.sum(axis=1), dtype=np.float32).ravel()


def build_stores(c1: Dict[str, np.ndarray], c2: Dict[str, np.ndarray], cfg: Config):
    n1, n2 = len(c1["entity_id"]), len(c2["entity_id"])
    allc = {k: np.concatenate([c1[k], c2[k]]) for k in c1}
    countries = sorted(set(allc["country_n"].tolist()))
    cmap = {c: i for i, c in enumerate(countries)}
    ccode = np.array([cmap[c] for c in allc["country_n"]], np.int32)
    with Timer("STEP 3a word-token matrices"):
        vocab, tm = token_matrices({"all": np.array([f"{a} {b}" for a, b in zip(allc["name_n"], allc["addr_n"])], dtype=object),
                                    "core": allc["core"], "addr": allc["addr_core"], "lm": allc["landmark"]},
                                   ["all"], cfg, "tokens")
        idf = per_country_idf(tm["all"], ccode, len(countries))
    with Timer("STEP 3b digit & phonetic matrices"):
        dvocab, dm = token_matrices({"codes": allc["codes"], "nums": allc["nums"]}, ["codes", "nums"], cfg, "digits")
        _, pm = token_matrices({"phon": allc["phon"]}, ["phon"], cfg, "phonetic")
    with Timer("STEP 3c char n-gram TF-IDF"):
        tf_name = char_tfidf(allc["name_n"], cfg, "char tf-idf name")
        tf_full = char_tfidf(allc["full_n"], cfg, "char tf-idf name+addr")

    mats = {"tf_name": tf_name, "tf_full": tf_full, "codes_b": dm["codes"], "nums_b": dm["nums"], "phon_b": pm["phon"]}
    for k in ("all", "core", "addr", "lm"):
        mats[f"{k}_b"] = tm[k]
        mats[f"{k}_w"] = weighted(tm[k], ccode, idf)
    vec = {f"{k}_wsum": _rowsum(mats[f"{k}_w"]) for k in ("all", "core", "addr", "lm")}
    for k in ("codes_b", "nums_b", "phon_b", "core_b"):
        vec[k.replace("_b", "_n")] = np.diff(mats[k].indptr).astype(np.float32)
    rare = np.asarray(mats["core_w"].argmax(axis=1)).ravel().astype(np.int64)
    rare[vec["core_n"] == 0] = -1
    vec["rare_core"] = rare
    for k, col in (("name_len", "core"), ("addr_len", "addr_core"), ("acr_len", "acr"), ("nospace_len", "core_nospace")):
        vec[k] = np.fromiter((len(x) for x in allc[col]), np.float32, count=n1 + n2)
    allc["country_code"] = ccode

    def part(sl):
        return Store(n=sl.stop - sl.start, cols={k: v[sl] for k, v in allc.items()},
                     mats={k: v[sl] for k, v in mats.items()}, vec={k: v[sl] for k, v in vec.items()})

    meta = {"countries": countries, "vocab": vocab, "tok3": np.array([t[:3] for t in vocab], dtype=object),
            "dvocab": dvocab}
    LOG.info(f"store: S1={n1} targets={n2} word-vocab={len(vocab)} digit-vocab={len(dvocab)} countries={countries}")
    return part(slice(0, n1)), part(slice(n1, n1 + n2)), meta


# =============================================================================
# STEP 4 — candidate retrieval (tfidf exact top-k | MinHash-LSH | both) + exact keys (+ dense)
# =============================================================================
def rowdot(MA, MB, i, j) -> np.ndarray:
    return _rowsum(MA[i].multiply(MB[j]))


def rowdot_chunked(MA, MB, i, j, cfg: Config, desc="") -> np.ndarray:
    if len(i) == 0:
        return np.zeros(0, np.float32)
    rng = split_chunks(len(i), cfg.chunk_size)
    return np.concatenate(thread_map(lambda r: rowdot(MA, MB, i[r[0]:r[1]], j[r[0]:r[1]]), rng, cfg.feature_threads, desc))


def sparse_topn(Q, T, k, thr, cfg: Config, desc=""):
    """Exact top-k cosine of L2-normalised rows (sparse_dot_topn is multi-threaded C++)."""
    if Q.shape[0] == 0 or T.shape[0] == 0:
        e = np.array([], np.int64)
        return e, e, np.array([], np.float32)
    rows, cols, vals = [], [], []
    TT = T.T.tocsr()
    for s in pbar(range(0, Q.shape[0], cfg.query_chunk), total=math.ceil(Q.shape[0] / cfg.query_chunk), desc=desc):
        q = Q[s:s + cfg.query_chunk]
        if HAS_SDT:
            C = sp_matmul_topn(q, TT, top_n=k, threshold=thr, sort=True, n_threads=cfg.n_jobs).tocsr()
        else:
            C = (q @ TT).tocsr()
            C.data[C.data < thr] = 0
            C.eliminate_zeros()
            keep_r, keep_c, keep_v = [], [], []
            for r in range(C.shape[0]):
                a, b = C.indptr[r], C.indptr[r + 1]
                if b > a:
                    top = np.argsort(-C.data[a:b])[:k]
                    keep_r.append(np.full(len(top), r))
                    keep_c.append(C.indices[a:b][top])
                    keep_v.append(C.data[a:b][top])
            C = sp.csr_matrix((np.concatenate(keep_v) if keep_v else [], (np.concatenate(keep_r) if keep_r else [],
                               np.concatenate(keep_c) if keep_c else [])), shape=C.shape)
        rows.append(np.repeat(np.arange(C.shape[0]) + s, np.diff(C.indptr)))
        cols.append(C.indices.copy())
        vals.append(C.data.astype(np.float32))
    return np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)


def minhash_band_keys(M: sp.csr_matrix, bands: int, rows_per_band: int, seed: int, desc: str,
                      max_nnz: int = 400_000) -> np.ndarray:
    """MinHash over the n-gram SET of each row (the column ids of M), then one uint64 key per band.
       P[two rows share a band key] = J^r ; P[candidate] = 1-(1-J^r)^b."""
    P = np.uint64(2 ** 31 - 1)
    num_perm = bands * rows_per_band
    rs = np.random.RandomState(seed)
    a = rs.randint(1, 2 ** 31 - 1, num_perm).astype(np.uint64)
    b = rs.randint(0, 2 ** 31 - 1, num_perm).astype(np.uint64)
    n = M.shape[0]
    sig = np.empty((n, num_perm), np.uint32)
    ip = M.indptr
    nnz = np.diff(ip)
    r0 = 0
    bar = pbar(total=n, desc=desc, unit="rows")
    while r0 < n:
        r1 = int(np.searchsorted(ip, ip[r0] + max_nnz, side="right")) - 1
        r1 = min(n, max(r0 + 1, r1))
        idx = M.indices[ip[r0]:ip[r1]].astype(np.uint64)
        nn = nnz[r0:r1]
        ne = np.where(nn > 0)[0]
        if len(ne):
            H = (idx[:, None] * a[None, :] + b[None, :]) % P
            starts = (ip[r0:r1] - ip[r0])[ne]
            sig[r0 + ne] = np.minimum.reduceat(H, starts, axis=0).astype(np.uint32)
        em = np.where(nn == 0)[0]
        if len(em):   # unique sentinel per empty row (>= P, never produced by the hash) -> never collides
            sig[r0 + em] = (np.uint64(2 ** 31) + (r0 + em).astype(np.uint64))[:, None].astype(np.uint32)
        bar.update(r1 - r0)
        r0 = r1
    bar.close()
    mult = (rs.randint(1, 2 ** 62, rows_per_band).astype(np.uint64) | np.uint64(1))
    keys = np.empty((n, bands), np.uint64)
    with np.errstate(over="ignore"):
        for bd in range(bands):
            keys[:, bd] = (sig[:, bd * rows_per_band:(bd + 1) * rows_per_band].astype(np.uint64) * mult).sum(axis=1)
    return keys


def lsh_pairs(KA, KB, qa, tb, cap: int) -> pd.DataFrame:
    out = []
    for bd in range(KA.shape[1]):
        a = pd.DataFrame({"key": KA[qa, bd], "q": qa})
        b = pd.DataFrame({"key": KB[tb, bd], "t": tb})
        vc = b["key"].value_counts()
        b = b[b["key"].map(vc).to_numpy() <= cap]
        out.append(a.merge(b, on="key")[["q", "t"]])
    if not out:
        return pd.DataFrame({"q": [], "t": []}, dtype=np.int64)
    return pd.concat(out, ignore_index=True).drop_duplicates()


def key_table(S: Store, meta: dict) -> pd.DataFrame:
    """Exact blocking keys, fully vectorised:
       pc = (postal-like code, 3-char prefix of rarest core token); an = (acronym, street number);
       cn = first 8 chars of the space-free core name."""
    parts = []
    rare = S.vec["rare_core"]
    r, c = S.mats["codes_b"].nonzero()
    m = rare[r] >= 0
    if m.any():
        parts.append(pd.DataFrame({"row": r[m], "k": "pc|" + pd.Series(meta["dvocab"][c[m]]) + "|" +
                                   pd.Series(meta["tok3"][rare[r[m]]])}))
    r, c = S.mats["nums_b"].nonzero()
    m = S.vec["acr_len"][r] >= 2
    if m.any():
        parts.append(pd.DataFrame({"row": r[m], "k": "an|" + pd.Series(S.cols["acr"][r[m]]) + "|" +
                                   pd.Series(meta["dvocab"][c[m]])}))
    m = np.where(S.vec["nospace_len"] >= 6)[0]
    if len(m):
        parts.append(pd.DataFrame({"row": m, "k": "cn|" + pd.Series(S.cols["core_nospace"][m]).str[:8]}))
    if not parts:
        return pd.DataFrame({"row": np.array([], np.int64), "key": np.array([], np.uint64)})
    kt = pd.concat(parts, ignore_index=True)
    return pd.DataFrame({"row": kt["row"].to_numpy(np.int64),
                         "key": pd.util.hash_array(kt["k"].to_numpy(dtype=object))}).drop_duplicates()


def key_pairs(ka: pd.DataFrame, kb: pd.DataFrame, qa, tb, cap: int) -> pd.DataFrame:
    a = ka[ka["row"].isin(qa)]
    b = kb[kb["row"].isin(tb)]
    vc = b["key"].value_counts()
    b = b[b["key"].map(vc).to_numpy() <= cap]
    m = a.merge(b, on="key", suffixes=("_a", "_b"))
    return pd.DataFrame({"q": m["row_a"].to_numpy(), "t": m["row_b"].to_numpy()}).drop_duplicates()


def dense_topn(q_text, t_text, k, model_name):
    """Optional dense view (sentence-transformers + FAISS HNSW if available)."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    pre = "query: " if "e5" in model_name else ""
    E1 = model.encode([pre + t for t in q_text], batch_size=256, normalize_embeddings=True, show_progress_bar=_PROGRESS).astype(np.float32)
    E2 = model.encode([pre + t for t in t_text], batch_size=256, normalize_embeddings=True, show_progress_bar=_PROGRESS).astype(np.float32)
    k = min(k, len(t_text))
    try:
        import faiss
        idx = faiss.IndexHNSWFlat(E2.shape[1], 32, faiss.METRIC_INNER_PRODUCT)
        idx.add(E2)
        D, I = idx.search(E1, k)
    except Exception:
        D = np.zeros((len(E1), k), np.float32)
        I = np.zeros((len(E1), k), np.int64)
        for s in range(0, len(E1), 4096):
            Sm = E1[s:s + 4096] @ E2.T
            ii = np.argpartition(-Sm, k - 1, axis=1)[:, :k]
            D[s:s + 4096] = np.take_along_axis(Sm, ii, 1)
            I[s:s + 4096] = ii
    return np.repeat(np.arange(len(E1)), k), I.ravel(), D.ravel()


FLAG_COLS = ["found_name", "found_full", "found_key", "found_lsh", "found_dense"]


def _view(q, t, flag, score=None) -> pd.DataFrame:
    d = pd.DataFrame({"q": np.asarray(q, np.int64), "t": np.asarray(t, np.int64), flag: np.int8(1)})
    if score is not None:
        d["sc_dense"] = np.asarray(score, np.float32)
    return d


def retrieve(cfg: Config, A: Store, B: Store, meta: dict, country_block: bool) -> pd.DataFrame:
    frames = []
    use_tfidf = cfg.blocker in ("tfidf", "both")
    use_lsh = cfg.blocker in ("lsh", "both")
    if use_lsh:
        with Timer("STEP 4a MinHash signatures"):
            ks = {}
            for v in ("tf_name", "tf_full"):
                K = minhash_band_keys(sp.vstack([A.mats[v], B.mats[v]]).tocsr(), cfg.lsh_bands, cfg.lsh_rows,
                                      cfg.seed + (v == "tf_full"), desc=f"minhash {v}")
                ks[v] = (K[:A.n], K[A.n:])
    ka, kb = key_table(A, meta), key_table(B, meta)
    ca, cb = A.cols["country_code"], B.cols["country_code"]
    groups = ([(meta["countries"][c], np.where(ca == c)[0], np.where(cb == c)[0]) for c in np.unique(ca)]
              if country_block else [("__all__", np.arange(A.n), np.arange(B.n))])
    for name, qa, tb in pbar(groups, total=len(groups), desc="STEP 4 retrieval blocks"):
        if len(qa) == 0:
            continue
        if len(tb) == 0:
            LOG.warning(f"  block '{name}': {len(qa)} S1 entities but no S2/S3 record carries that country label "
                        f"-> falling back to retrieval over ALL {B.n} targets")
            tb = np.arange(B.n)
        n_before = sum(len(f) for f in frames)
        if use_tfidf:
            for v, k, flag in (("tf_name", cfg.k_sparse_name, "found_name"), ("tf_full", cfg.k_sparse_full, "found_full")):
                r, c, _ = sparse_topn(A.mats[v][qa], B.mats[v][tb], k, cfg.sparse_threshold, cfg, desc=f"top-k {v} [{name}]")
                frames.append(_view(qa[r], tb[c], flag))
        if use_lsh:
            lp = pd.concat([lsh_pairs(*ks[v], qa, tb, cfg.lsh_bucket_cap) for v in ("tf_name", "tf_full")]).drop_duplicates()
            if len(lp):
                qi, ti = lp["q"].to_numpy(), lp["t"].to_numpy()
                sc = np.maximum(rowdot_chunked(A.mats["tf_name"], B.mats["tf_name"], qi, ti, cfg, "lsh rescoring name"),
                                rowdot_chunked(A.mats["tf_full"], B.mats["tf_full"], qi, ti, cfg, "lsh rescoring full"))
                lp = lp.assign(sc=sc).sort_values(["q", "sc"], ascending=[True, False])
                lp = lp[lp.groupby("q").cumcount() < cfg.k_lsh]
                frames.append(_view(lp["q"], lp["t"], "found_lsh"))
        kp = key_pairs(ka, kb, qa, tb, cfg.max_key_block)
        frames.append(_view(kp["q"], kp["t"], "found_key"))
        if cfg.use_dense:
            try:
                r, c, v = dense_topn([f"{x} | {y}" for x, y in zip(A.cols["raw_name"][qa], A.cols["raw_addr"][qa])],
                                     [f"{x} | {y}" for x, y in zip(B.cols["raw_name"][tb], B.cols["raw_addr"][tb])],
                                     cfg.k_dense, cfg.dense_model)
                frames.append(_view(qa[r], tb[c], "found_dense", v))
            except Exception as ex:
                LOG.warning(f"dense view failed ({ex}); continuing without it")
        LOG.info(f"  block '{name}': S1={len(qa)} targets={len(tb)} view-rows={sum(len(f) for f in frames) - n_before}")
    if not frames:
        return pd.DataFrame({c: [] for c in ["s1_idx", "t_idx"] + FLAG_COLS + ["sc_dense", "n_views"]})
    allv = pd.concat(frames, ignore_index=True)
    for c in FLAG_COLS + ["sc_dense"]:
        allv[c] = allv[c].fillna(0) if c in allv else 0
    key = allv["q"].to_numpy(np.int64) * B.n + allv["t"].to_numpy(np.int64)
    g = allv[FLAG_COLS + ["sc_dense"]].groupby(key, sort=True).max()
    k = g.index.to_numpy(np.int64)
    out = pd.DataFrame({"s1_idx": (k // B.n).astype(np.int32), "t_idx": (k % B.n).astype(np.int32)})
    for c in FLAG_COLS:
        out[c] = g[c].to_numpy(np.int8)
    out["sc_dense"] = g["sc_dense"].to_numpy(np.float32)
    out["n_views"] = out[FLAG_COLS].sum(axis=1).astype(np.int8)
    return out


# =============================================================================
# STEP 5/6 — feature engineering & extraction (vectorised, chunked, threaded)
# =============================================================================
FEATURE_GROUPS: Dict[str, List[str]] = {
    "retrieval":      ["cos_name", "cos_full", "found_name", "found_full", "found_key", "found_lsh", "found_dense",
                       "sc_dense", "n_views"],
    "name_token":     ["wjac_name", "shared_rare_name", "unshared_rare_name", "rare_diff_name", "cont_a_in_b", "cont_b_in_a"],
    "name_string":    ["tsr_name", "tsort_name", "jw_name", "lev_name", "partial_name", "jw_nospace", "core_equal",
                       "prefix_sim", "len_ratio", "ntok_diff"],
    "name_structure": ["acr_match", "suffix_equal", "suffix_conflict", "first_tok_equal"],
    "phonetic":       ["phon_jac", "phon_first_equal"],
    "address_token":  ["wjac_addr", "shared_rare_addr", "wjac_lm", "wjac_all"],
    "address_string": ["tsr_addr", "lev_addr", "partial_addr"],
    "numeric":        ["code_agree", "num_agree", "n_code_shared", "n_num_shared", "code_jac"],
    "missingness":    ["a_has_addr", "b_has_addr", "a_n_codes", "b_n_codes", "a_name_len", "b_name_len", "src_s3"],
    "context":        ["n_cand", "cos_name_rank", "cos_name_margin", "cos_full_rank", "cos_full_margin"],
    "stage2":         ["p1", "p1_rank", "p1_margin"],
    "competition":    ["rev_rank", "rev_n", "rev_margin", "mutual_best"],
    "collective":     ["sib_full_tsr", "sib_name_jw", "sib_p", "sib_evidence", "n_confident"],
    "cross_encoder":  ["ce_logit"],
}
FEATURE_TO_GROUP = {f: g for g, fs in FEATURE_GROUPS.items() for f in fs}
PAIR_FEATURES = set(FEATURE_GROUPS["name_token"] + FEATURE_GROUPS["name_string"] + FEATURE_GROUPS["name_structure"] +
                    FEATURE_GROUPS["phonetic"] + FEATURE_GROUPS["address_token"] + FEATURE_GROUPS["address_string"] +
                    FEATURE_GROUPS["numeric"] + FEATURE_GROUPS["missingness"] + ["cos_name", "cos_full"])
CHEAP_FEATURES = (FEATURE_GROUPS["retrieval"] + FEATURE_GROUPS["name_token"] + ["tsr_name", "core_equal"] +
                  FEATURE_GROUPS["numeric"] + ["phon_jac"] + FEATURE_GROUPS["missingness"] + FEATURE_GROUPS["context"])


def active(features: List[str], cfg: Config) -> List[str]:
    return [f for f in features if FEATURE_TO_GROUP.get(f) not in set(cfg.drop_groups)]


def _rowmax(M) -> np.ndarray:
    if M.shape[0] == 0:
        return np.zeros(0, np.float32)
    return M.max(axis=1).toarray().ravel().astype(np.float32)


def _div(a, b) -> np.ndarray:
    return np.where(b > 0, a / np.maximum(b, 1e-12), 0.0).astype(np.float32)


def _wjac(A: Store, B: Store, i, j, kind):
    """IDF-weighted Jaccard: sum_{t in a∩b} w(t) / sum_{t in a∪b} w(t)."""
    Wa = A.mats[f"{kind}_w"][i]
    Bb = B.mats[f"{kind}_b"][j]
    shared = Wa.multiply(Bb)
    inter = _rowsum(shared)
    union = A.vec[f"{kind}_wsum"][i] + B.vec[f"{kind}_wsum"][j] - inter
    return _div(inter, union), shared, Wa, Bb


def _cp(a, b, scorer, workers):
    return process.cpdist(a, b, scorer=scorer, workers=workers, dtype=np.float32)


def compute_pair_features(A: Store, B: Store, i: np.ndarray, j: np.ndarray, want: set, workers: int = 1) -> Dict[str, np.ndarray]:
    """All pairwise features for pairs (i, j). Everything is vectorised: sparse row-wise products for
       token/digit/phonetic features, RapidFuzz cpdist (C++) for string features."""
    f: Dict[str, np.ndarray] = {}

    def need(*names):
        return any(n in want for n in names)

    # -- retrieval: exact cosines for EVERY pair (not only the view that found it)
    if need("cos_name"):
        f["cos_name"] = rowdot(A.mats["tf_name"], B.mats["tf_name"], i, j)
    if need("cos_full"):
        f["cos_full"] = rowdot(A.mats["tf_full"], B.mats["tf_full"], i, j)
    # -- name tokens (IDF-weighted, per-country IDF)
    if need("wjac_name", "shared_rare_name", "unshared_rare_name", "rare_diff_name"):
        jac, shared, Wa, Bb = _wjac(A, B, i, j, "core")
        sr = _rowmax(shared)
        f["wjac_name"], f["shared_rare_name"] = jac, sr
        if need("unshared_rare_name", "rare_diff_name"):
            S = Wa + B.mats["core_w"][j]
            X = S - S.multiply(A.mats["core_b"][i].multiply(Bb))      # keeps only tokens present on one side
            us = _rowmax(X)
            f["unshared_rare_name"], f["rare_diff_name"] = us, sr - us
    if need("cont_a_in_b"):   # share of A's name weight found anywhere in B's record (DBA / field swaps)
        f["cont_a_in_b"] = _div(rowdot(A.mats["core_w"], B.mats["all_b"], i, j), A.vec["core_wsum"][i])
    if need("cont_b_in_a"):
        f["cont_b_in_a"] = _div(rowdot(B.mats["core_w"], A.mats["all_b"], j, i), B.vec["core_wsum"][j])
    # -- address tokens
    if need("wjac_addr", "shared_rare_addr"):
        jac, shared, _, _ = _wjac(A, B, i, j, "addr")
        f["wjac_addr"], f["shared_rare_addr"] = jac, _rowmax(shared)
    if need("wjac_lm"):
        f["wjac_lm"] = _wjac(A, B, i, j, "lm")[0]
    if need("wjac_all"):
        f["wjac_all"] = _wjac(A, B, i, j, "all")[0]
    # -- numeric: three-valued agreement (1 agree / 0 missing on a side / -1 conflict)
    if need("code_agree", "n_code_shared", "code_jac"):
        inter = rowdot(A.mats["codes_b"], B.mats["codes_b"], i, j)
        na, nb = A.vec["codes_n"][i], B.vec["codes_n"][j]
        f["code_agree"] = np.where((na == 0) | (nb == 0), 0, np.where(inter > 0, 1, -1)).astype(np.float32)
        f["n_code_shared"], f["code_jac"] = inter, _div(inter, na + nb - inter)
    if need("num_agree", "n_num_shared"):
        inter = rowdot(A.mats["nums_b"], B.mats["nums_b"], i, j)
        na, nb = A.vec["nums_n"][i], B.vec["nums_n"][j]
        f["num_agree"] = np.where((na == 0) | (nb == 0), 0, np.where(inter > 0, 1, -1)).astype(np.float32)
        f["n_num_shared"] = inter
    # -- phonetic
    if need("phon_jac"):
        inter = rowdot(A.mats["phon_b"], B.mats["phon_b"], i, j)
        f["phon_jac"] = _div(inter, A.vec["phon_n"][i] + B.vec["phon_n"][j] - inter)
    if need("phon_first_equal"):
        pa, pb = A.cols["phon_first"][i], B.cols["phon_first"][j]
        f["phon_first_equal"] = ((pa == pb) & (pa != "")).astype(np.float32)
    # -- name strings (RapidFuzz, C++)
    if need("tsr_name", "tsort_name", "jw_name", "lev_name", "partial_name"):
        ca, cb = A.cols["core"][i].tolist(), B.cols["core"][j].tolist()
        for name, scorer, scale in (("tsr_name", fuzz.token_set_ratio, 100), ("tsort_name", fuzz.token_sort_ratio, 100),
                                    ("jw_name", JaroWinkler.normalized_similarity, 1),
                                    ("lev_name", Levenshtein.normalized_similarity, 1), ("partial_name", fuzz.partial_ratio, 100)):
            if name in want:
                f[name] = _cp(ca, cb, scorer, workers) / scale
    if need("jw_nospace", "prefix_sim", "core_equal"):
        na_, nb_ = A.cols["core_nospace"][i], B.cols["core_nospace"][j]
        if "jw_nospace" in want:
            f["jw_nospace"] = _cp(na_.tolist(), nb_.tolist(), JaroWinkler.normalized_similarity, workers)
        if "prefix_sim" in want:
            f["prefix_sim"] = _cp(na_.tolist(), nb_.tolist(), Prefix.normalized_similarity, workers)
        f["core_equal"] = ((na_ == nb_) & (na_ != "")).astype(np.float32)
    if need("len_ratio"):
        la, lb = A.vec["name_len"][i], B.vec["name_len"][j]
        f["len_ratio"] = _div(np.minimum(la, lb), np.maximum(la, lb))
    if need("ntok_diff"):
        f["ntok_diff"] = np.abs(A.vec["core_n"][i] - B.vec["core_n"][j])
    # -- name structure
    if need("acr_match"):
        aa, ab = A.cols["acr"][i], B.cols["acr"][j]
        na_, nb_ = A.cols["core_nospace"][i], B.cols["core_nospace"][j]
        f["acr_match"] = ((A.vec["acr_len"][i] >= 2) & ((aa == ab) | (aa == nb_) | (ab == na_))).astype(np.float32)
    if need("suffix_equal", "suffix_conflict"):
        sa, sb = A.cols["suffix"][i], B.cols["suffix"][j]
        f["suffix_equal"] = ((sa == sb) & (sa != "")).astype(np.float32)
        f["suffix_conflict"] = ((sa != "") & (sb != "") & (sa != sb)).astype(np.float32)
    if need("first_tok_equal"):
        fa, fb = A.cols["first_tok"][i], B.cols["first_tok"][j]
        f["first_tok_equal"] = ((fa == fb) & (fa != "")).astype(np.float32)
    # -- address strings
    if need("tsr_addr", "lev_addr", "partial_addr"):
        aa, ab = A.cols["addr_core"][i].tolist(), B.cols["addr_core"][j].tolist()
        for name, scorer, scale in (("tsr_addr", fuzz.token_set_ratio, 100), ("lev_addr", Levenshtein.normalized_similarity, 1),
                                    ("partial_addr", fuzz.partial_ratio, 100)):
            if name in want:
                f[name] = _cp(aa, ab, scorer, workers) / scale
    # -- record-level missingness / size
    if need("a_has_addr"):
        f["a_has_addr"] = (A.vec["addr_len"][i] > 0).astype(np.float32)
    if need("b_has_addr"):
        f["b_has_addr"] = (B.vec["addr_len"][j] > 0).astype(np.float32)
    if need("a_n_codes"):
        f["a_n_codes"] = A.vec["codes_n"][i]
    if need("b_n_codes"):
        f["b_n_codes"] = B.vec["codes_n"][j]
    if need("a_name_len"):
        f["a_name_len"] = A.vec["name_len"][i]
    if need("b_name_len"):
        f["b_name_len"] = B.vec["name_len"][j]
    if need("src_s3"):
        f["src_s3"] = (B.cols["source"][j] == 3).astype(np.float32)
    return {k: np.asarray(v, dtype=np.float32) for k, v in f.items() if k in want}


def extract_features(pairs: pd.DataFrame, A: Store, B: Store, feats: List[str], cfg: Config, desc: str):
    """Chunked extraction; chunks run on a thread pool (the heavy work releases the GIL). Adds columns in place."""
    feats = [f for f in feats if f in PAIR_FEATURES and f not in pairs.columns]
    if not feats:
        return pairs
    if len(pairs) == 0:
        for k in feats:
            pairs[k] = np.zeros(0, np.float32)
        return pairs
    i = pairs["s1_idx"].to_numpy(np.int64)
    j = pairs["t_idx"].to_numpy(np.int64)
    want = set(feats)
    workers = 1 if cfg.feature_threads > 1 else cfg.n_jobs
    ranges = split_chunks(len(pairs), cfg.chunk_size)
    res = thread_map(lambda r: compute_pair_features(A, B, i[r[0]:r[1]], j[r[0]:r[1]], want, workers),
                     ranges, cfg.feature_threads, desc=f"{desc} ({len(feats)} feats, {len(ranges)} chunks)")
    for k in feats:
        pairs[k] = np.concatenate([r[k] for r in res])
    return pairs


def add_rank_features(pairs: pd.DataFrame, col: str, prefix: str):
    g = pairs.groupby("s1_idx")[col]
    pairs[f"{prefix}_rank"] = g.rank(ascending=False, method="first").astype(np.float32)
    pairs[f"{prefix}_margin"] = (g.transform("max") - pairs[col]).astype(np.float32)


def context_features(pairs: pd.DataFrame):
    pairs["n_cand"] = pairs.groupby("s1_idx")["t_idx"].transform("size").astype(np.float32)
    add_rank_features(pairs, "cos_name", "cos_name")
    add_rank_features(pairs, "cos_full", "cos_full")


def competition_features(pairs: pd.DataFrame, pcol: str) -> pd.DataFrame:
    """Reverse view: rank of this S1 entity among all S1 entities competing for the same S2/S3 record."""
    g = pairs.groupby("t_idx")[pcol]
    pairs["rev_rank"] = g.rank(ascending=False, method="first").astype(np.float32)
    pairs["rev_n"] = g.transform("size").astype(np.float32)
    pairs["rev_margin"] = (g.transform("max") - pairs[pcol]).astype(np.float32)
    fr = pairs.groupby("s1_idx")[pcol].rank(ascending=False, method="first")
    pairs["p1_rank"] = fr.astype(np.float32)
    pairs["p1_margin"] = (pairs.groupby("s1_idx")[pcol].transform("max") - pairs[pcol]).astype(np.float32)
    pairs["mutual_best"] = ((fr == 1) & (pairs["rev_rank"] == 1)).astype(np.float32)
    return pairs


def collective_features(pairs: pd.DataFrame, pcol: str, B: Store, cfg: Config) -> pd.DataFrame:
    """Transitivity (S2/S3 are not deduplicated): similarity of each candidate to the entity's best candidate
       (or, for the best one, to the runner-up), that reference's probability, and #confident siblings."""
    if len(pairs) == 0:
        for c in FEATURE_GROUPS["collective"]:
            pairs[c] = np.zeros(0, np.float32)
        return pairs
    s = pairs["s1_idx"].to_numpy()
    t = pairs["t_idx"].to_numpy()
    p = pairs[pcol].to_numpy(np.float32)
    order = np.lexsort((-p, s))
    ents, st, ct = np.unique(s[order], return_index=True, return_counts=True)
    best_t, best_p = t[order][st], p[order][st]
    has2 = ct > 1
    sec_t = np.where(has2, t[order][np.minimum(st + 1, len(t) - 1)], -1)
    sec_p = np.where(has2, p[order][np.minimum(st + 1, len(p) - 1)], 0.0)
    pos = np.searchsorted(ents, s)
    is_best = t == best_t[pos]
    ref_t = np.where(is_best, sec_t[pos], best_t[pos])
    ref_p = np.where(is_best, sec_p[pos], best_p[pos]).astype(np.float32)
    has = ref_t >= 0
    rt = np.where(has, ref_t, 0)
    workers = cfg.n_jobs
    full, core = B.cols["full_n"], B.cols["core_nospace"]
    pairs["sib_full_tsr"] = np.where(has, _cp(full[t].tolist(), full[rt].tolist(), fuzz.token_set_ratio, workers) / 100, 0).astype(np.float32)
    pairs["sib_name_jw"] = np.where(has, _cp(core[t].tolist(), core[rt].tolist(), JaroWinkler.normalized_similarity, workers), 0).astype(np.float32)
    pairs["sib_p"] = np.where(has, ref_p, 0).astype(np.float32)
    pairs["sib_evidence"] = (pairs["sib_full_tsr"] * pairs["sib_p"]).astype(np.float32)
    pairs["n_confident"] = np.bincount(pos, weights=(p >= 0.5), minlength=len(ents))[pos].astype(np.float32)
    return pairs


# =============================================================================
# STEP 5/7 — labels, folds, LightGBM training helpers, pruning
# =============================================================================
def gt_index(gt: Dict[str, set], A: Store, B: Store):
    """GT as integer pair keys + per-entity GT sizes (sizes count every GT id, even ones retrieval misses)."""
    a_map = pd.Series(np.arange(A.n), index=A.cols["entity_id"])
    b_map = pd.Series(np.arange(B.n), index=B.cols["entity_id"])
    rows = [(s, t) for s, ts in gt.items() for t in ts]
    df = pd.DataFrame(rows, columns=["s", "t"]) if rows else pd.DataFrame({"s": [], "t": []})
    si = a_map.reindex(df["s"]).to_numpy()
    ti = b_map.reindex(df["t"]).to_numpy()
    ok = ~(pd.isna(si) | pd.isna(ti))
    keys = np.unique(si[ok].astype(np.int64) * B.n + ti[ok].astype(np.int64))
    sizes = pd.Series({s: len(v) for s, v in gt.items()}, dtype=float).reindex(A.cols["entity_id"]).fillna(0).to_numpy()
    return keys, sizes.astype(np.int32)


def label_pairs(pairs: pd.DataFrame, gt_keys: np.ndarray, nB: int) -> np.ndarray:
    k = pairs["s1_idx"].to_numpy(np.int64) * nB + pairs["t_idx"].to_numpy(np.int64)
    return np.isin(k, gt_keys).astype(np.int8)


def entity_fold_map(n: int, k: int, seed: int) -> np.ndarray:
    return (np.random.RandomState(seed).permutation(n) % k).astype(np.int32)


def make_pair_folds(s_idx: np.ndarray, ent_fold: np.ndarray, pool_ent: np.ndarray, K: int):
    """K folds over the training pool (grouped by S1 entity) + held-out rows (leave-one-country-out mode)."""
    in_pool = pool_ent[s_idx]
    f = ent_fold[s_idx]
    folds = [(np.where(in_pool & (f != k))[0], np.where(in_pool & (f == k))[0]) for k in range(K)]
    return folds, np.where(~in_pool)[0]


def lgb_params(cfg: Config, seed: int, leaves=63) -> dict:
    return dict(objective="binary", learning_rate=0.05, num_leaves=leaves, min_child_samples=40, feature_fraction=0.8,
                bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, max_bin=127, force_col_wise=True, verbose=-1,
                seed=seed, num_threads=cfg.n_jobs)


def oof_lgb(X: np.ndarray, y: np.ndarray, folds, hold: np.ndarray, rounds: int, cfg: Config, leaves: int,
            neg_rate: float, desc: str):
    """Out-of-fold probabilities for every pool row, final model on the whole pool predicts held-out rows.
       Negative subsampling with weight 1/rate keeps the implied prior (and isotonic calibration cleans up)."""
    oof = np.full(len(y), np.nan, np.float32)

    def fit(idx, seed):
        w = None
        if neg_rate < 1.0:
            rs = np.random.RandomState(seed)
            neg = y[idx] == 0
            keep = ~neg | (rs.rand(len(idx)) < neg_rate)
            idx = idx[keep]
            w = np.where(y[idx] == 0, 1.0 / neg_rate, 1.0).astype(np.float32)
        return lgb.train(lgb_params(cfg, seed, leaves), lgb.Dataset(X[idx], y[idx], weight=w), num_boost_round=rounds)

    bar = pbar(total=len(folds) + 1, desc=desc)
    for k, (tr, va) in enumerate(folds):
        if len(tr) and len(va):
            oof[va] = fit(tr, cfg.seed + k).predict(X[va], num_threads=cfg.n_jobs)
        bar.update(1)
    pool = np.concatenate([va for _, va in folds]) if folds else np.arange(len(y))
    final = fit(np.sort(pool), cfg.seed)
    if len(hold):
        oof[hold] = final.predict(X[hold], num_threads=cfg.n_jobs)
    bar.update(1)
    bar.close()
    return oof, final


def fit_isotonic(p: np.ndarray, y: np.ndarray) -> IsotonicRegression:
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso.fit(p, y)
    return iso


def choose_eps(p1: np.ndarray, y: np.ndarray, target: float, floor: float) -> float:
    pos = p1[y == 1]
    if len(pos) == 0:
        return floor
    return max(floor, min(float(np.quantile(pos, max(0.0, 1.0 - target))), 0.5))


def prune(pairs: pd.DataFrame, pcol: str, eps: float, kmax: int) -> pd.DataFrame:
    keep = pairs[pairs[pcol] >= eps]
    r = keep.groupby("s1_idx")[pcol].rank(ascending=False, method="first")
    return keep[r.to_numpy() <= kmax]


def cross_encoder_scores(cfg: Config, train_pairs, y, folds, hold, A, B, test_pairs, At, Bt):
    """Optional Ditto-style cross-encoder (GPU). Returns (oof_logits_train, logits_test)."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(cfg.ce_model)

    def texts(p, SA, SB):
        i, j = p["s1_idx"].to_numpy(), p["t_idx"].to_numpy()
        return [f"[COL] name [VAL] {n1} [COL] address [VAL] {a1} [SEP] [COL] name [VAL] {n2} [COL] address [VAL] {a2}"
                for n1, a1, n2, a2 in zip(SA.cols["raw_name"][i], SA.cols["raw_addr"][i], SB.cols["raw_name"][j], SB.cols["raw_addr"][j])]

    def train_and_score(tr_txt, tr_y, score_sets, seed):
        torch.manual_seed(seed)
        rs = np.random.RandomState(seed)
        if len(tr_txt) > cfg.ce_max_train_pairs:
            keep = rs.choice(len(tr_txt), cfg.ce_max_train_pairs, replace=False)
            tr_txt, tr_y = [tr_txt[i] for i in keep], tr_y[keep]
        model = AutoModelForSequenceClassification.from_pretrained(cfg.ce_model, num_labels=1).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
        idx = np.arange(len(tr_txt))
        model.train()
        for _ in range(cfg.ce_epochs):
            rs.shuffle(idx)
            for s in pbar(range(0, len(idx), 32), total=math.ceil(len(idx) / 32), desc="cross-encoder train"):
                bi = idx[s:s + 32]
                enc = tok([tr_txt[i] for i in bi], truncation=True, max_length=160, padding=True, return_tensors="pt").to(dev)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    model(**enc).logits.squeeze(-1), torch.tensor(tr_y[bi], dtype=torch.float32, device=dev))
                loss.backward()
                opt.step()
                opt.zero_grad()
        model.eval()
        outs = []
        with torch.no_grad():
            for sc in score_sets:
                o = np.zeros(len(sc), np.float32)
                for s in pbar(range(0, len(sc), 128), total=math.ceil(len(sc) / 128), desc="cross-encoder score"):
                    enc = tok(sc[s:s + 128], truncation=True, max_length=160, padding=True, return_tensors="pt").to(dev)
                    o[s:s + 128] = model(**enc).logits.squeeze(-1).float().cpu().numpy()
                outs.append(o)
        return outs

    txt = texts(train_pairs, A, B)
    oof = np.zeros(len(train_pairs), np.float32)
    for k, (tr, va) in enumerate(folds):
        (oof[va],) = train_and_score([txt[i] for i in tr], y[tr], [[txt[i] for i in va]], cfg.seed + k)
    pool = np.concatenate([va for _, va in folds])
    sets = [[txt[i] for i in hold]] if len(hold) else []
    if test_pairs is not None and len(test_pairs):
        sets.append(texts(test_pairs, At, Bt))
    outs = train_and_score([txt[i] for i in pool], y[pool], sets, cfg.seed) if sets else []
    if len(hold):
        oof[hold] = outs.pop(0)
    return oof, (outs[0] if outs else None)


# =============================================================================
# STEP 8 — structural constraints
# =============================================================================
def apply_structure(pairs: pd.DataFrame, pcol: str, hard_assign: bool) -> np.ndarray:
    """Each S2/S3 record belongs to at most one S1 entity: p_ij <- p_ij / max(1, sum_i' p_i'j);
       hard mode additionally keeps only the argmax S1 entity per record."""
    raw = pairs[pcol].to_numpy(np.float64)
    g = pairs.groupby("t_idx")[pcol]
    p = raw / np.maximum(1.0, g.transform("sum").to_numpy())
    if hard_assign:
        p = np.where(raw >= g.transform("max").to_numpy() - 1e-12, p, 0.0)
    return p.astype(np.float32)


# =============================================================================
# STEP 9 — singleton model + vectorised / parallel expected-F0.5 decision
# =============================================================================
class EntityView:
    """Pairs sorted by (S1 entity, -p) with entity boundaries — everything downstream is array arithmetic."""

    def __init__(self, s, t, p, y=None, extra: Optional[Dict[str, np.ndarray]] = None):
        s, t, p = np.asarray(s), np.asarray(t), np.asarray(p, np.float32)
        order = np.lexsort((-p, s))
        self.s, self.t, self.p = s[order], t[order], p[order]
        self.y = None if y is None else np.asarray(y)[order].astype(bool)
        self.extra = {k: np.asarray(v)[order] for k, v in (extra or {}).items()}
        self.ents, self.starts, self.counts = np.unique(self.s, return_index=True, return_counts=True)
        self.rank = np.arange(len(self.s)) - np.repeat(self.starts, self.counts)

    def subset(self, ent_mask: np.ndarray) -> "EntityView":
        m = ent_mask[self.s]
        return EntityView(self.s[m], self.t[m], self.p[m], None if self.y is None else self.y[m],
                          {k: v[m] for k, v in self.extra.items()})


ENT_COLS = ["p_max", "p_2nd", "p_sum", "n_cand", "best_wjac", "best_code", "best_num", "best_mutual", "max_cos_name", "raw_pmax"]


def entity_features(view: EntityView, n_s1: int) -> np.ndarray:
    E = np.zeros((n_s1, len(ENT_COLS)), np.float32)
    if len(view.s) == 0:
        return E
    st, ct, p = view.starts, view.counts, view.p
    red = lambda col: np.maximum.reduceat(view.extra[col], st)  # noqa: E731
    cols = [p[st], np.where(ct > 1, p[np.minimum(st + 1, len(p) - 1)], 0), np.add.reduceat(p, st), ct,
            red("wjac_name"), red("code_agree"), red("num_agree"), red("mutual_best"), red("cos_name"), red("p3_raw")]
    E[view.ents] = np.column_stack(cols).astype(np.float32)
    return E


def poisson_binomial(ps: np.ndarray) -> np.ndarray:
    pmf = np.array([1.0])
    for q in ps:
        nxt = np.zeros(len(pmf) + 1)
        nxt[:-1] += pmf * (1 - q)
        nxt[1:] += pmf * q
        pmf = nxt
    return pmf


def expected_f_topk(p_sorted: np.ndarray, beta2: float, p_empty: float) -> np.ndarray:
    """E[F_beta(top_k)], k=1..n. P(Y=∅)=p_empty (singleton model); given Y≠∅ candidates ~ independent Bernoulli."""
    n = len(p_sorted)
    out = np.zeros(n)
    denom = max(1e-12, 1 - float(np.prod(1 - p_sorted)))
    for k in range(1, n + 1):
        pa, pb = poisson_binomial(p_sorted[:k]), poisson_binomial(p_sorted[k:])
        a = np.arange(len(pa))[:, None]
        b = np.arange(len(pb))[None, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            F = (1 + beta2) * a / (beta2 * (a + b) + k)
        F[0, 0] = 0.0
        out[k - 1] = (1 - p_empty) * float((pa[:, None] * pb[None, :] * F).sum()) / denom
    return out


def _ef_chunk(args) -> np.ndarray:
    """Worker (top-level, picklable): chosen k for each multi-candidate entity of a chunk (0 = empty)."""
    p_flat, starts, counts, pe, beta2, s_scale = args
    ks = np.zeros(len(starts), np.int32)
    for q in range(len(starts)):
        ps = p_flat[starts[q]:starts[q] + counts[q]]
        ef = expected_f_topk(ps, beta2, float(pe[q]))
        k = int(np.argmax(ef)) + 1
        if min(1.0, pe[q] * s_scale) < ef[k - 1]:
            ks[q] = k
    return ks


def decide(view: EntityView, p_empty: np.ndarray, beta2: float, policy: str, T: float, s_scale: float, tau: float,
           n_jobs: int) -> np.ndarray:
    """Boolean mask over view rows. 'threshold': p >= tau.
       'expected_f': single-candidate entities in closed form (vectorised), the rest in parallel worker chunks."""
    if policy == "threshold" or len(view.p) == 0:
        return view.p >= tau
    p = np.clip(view.p, 1e-6, 1 - 1e-6).astype(np.float64)
    if T != 1.0:
        p = 1 / (1 + np.exp(-np.log(p / (1 - p)) / T))
    pe = p_empty[view.ents]
    k_ent = np.zeros(len(view.ents), np.int32)
    one = view.counts == 1
    k_ent[one] = (np.minimum(1.0, pe[one] * s_scale) < (1 - pe[one])).astype(np.int32)   # E[F(top1)] = 1 - P(∅)
    multi = np.where(~one)[0]
    if len(multi):
        mst, mct, mpe = view.starts[multi], view.counts[multi], pe[multi]
        n_tasks = max(1, n_jobs * 4) if len(multi) > 4000 else 1
        tasks = [(p[mst[a]:mst[b - 1] + mct[b - 1]], mst[a:b] - mst[a], mct[a:b], mpe[a:b], beta2, s_scale)
                 for a, b in split_even(len(multi), n_tasks)]
        k_ent[multi] = np.concatenate(parallel_map(_ef_chunk, tasks, n_jobs if len(tasks) > 1 else 1, desc="expected-F decisions"))
    return view.rank < np.repeat(k_ent, view.counts)


def macro_f(view: EntityView, sel: np.ndarray, gt_size: np.ndarray, eval_mask: np.ndarray, beta2: float):
    n = len(gt_size)
    pred = np.bincount(view.s[sel], minlength=n)
    tp = np.bincount(view.s[sel & view.y], minlength=n)
    truth = gt_size
    with np.errstate(divide="ignore", invalid="ignore"):
        f = np.where((truth == 0) & (pred == 0), 1.0,
                     np.where((truth == 0) | (pred == 0), 0.0, (1 + beta2) * tp / (beta2 * truth + pred)))
    return (float(f[eval_mask].mean()) if eval_mask.any() else float("nan")), f


def singleton_models(E: np.ndarray, y_single: np.ndarray, ent_fold: np.ndarray, pool_ent: np.ndarray, cfg: Config):
    """OOF P(Y=∅) for pool entities, final model (pool) predicts held-out entities."""
    oof = np.full(len(E), np.nan)
    for k in pbar(range(cfg.n_folds), total=cfg.n_folds, desc="singleton model folds"):
        tr = np.where(pool_ent & (ent_fold != k))[0]
        va = np.where(pool_ent & (ent_fold == k))[0]
        if len(tr) and len(va):
            m = lgb.train(lgb_params(cfg, cfg.seed + k, 31), lgb.Dataset(E[tr], y_single[tr]), 300)
            oof[va] = m.predict(E[va], num_threads=cfg.n_jobs)
    final = lgb.train(lgb_params(cfg, cfg.seed, 31), lgb.Dataset(E[pool_ent], y_single[pool_ent]), 300)
    hold = np.where(~pool_ent)[0]
    if len(hold):
        oof[hold] = final.predict(E[hold], num_threads=cfg.n_jobs)
    return np.nan_to_num(oof, nan=0.5), final


def tune_decision(view: EntityView, p_empty, gt_size, tune_mask, cfg: Config) -> dict:
    """Grid search on out-of-fold probabilities of the training pool; two competing policies, best one kept."""
    rs = np.random.RandomState(cfg.seed)
    ids = np.where(tune_mask)[0]
    if len(ids) > cfg.tune_max_entities:
        ids = rs.choice(ids, cfg.tune_max_entities, replace=False)
    m = np.zeros(len(tune_mask), bool)
    m[ids] = True
    sub = view.subset(m)
    b2 = cfg.beta ** 2
    grid = [("expected_f", T, s, 0.5) for T in cfg.temperature_grid for s in cfg.singleton_scale_grid] + \
           [("threshold", 1.0, 1.0, tau) for tau in cfg.threshold_grid]
    best, best_by = None, {}
    for pol, T, s, tau in pbar(grid, total=len(grid), desc="STEP 9 decision grid"):
        f, _ = macro_f(sub, decide(sub, p_empty, b2, pol, T, s, tau, cfg.n_jobs), gt_size, m, b2)
        best_by[pol] = max(best_by.get(pol, -1), f)
        if best is None or f > best["f"] + 1e-9:
            best = {"f": f, "policy": pol, "T": T, "s_scale": s, "tau": tau}
    LOG.info(f"decision tuning on {len(ids)} entities: " + ", ".join(f"{k} best={v:.4f}" for k, v in best_by.items()) +
             f" -> chosen {best}")
    return best


# =============================================================================
# STEP 10/11 — evaluation helpers, outputs, validator
# =============================================================================
def blocking_stats(pairs: pd.DataFrame, n_s1: int, n_t: int, y: Optional[np.ndarray], n_true: int, name: str) -> dict:
    n = len(pairs)
    st = {"stage": name, "pairs": int(n), "avg_candidates_per_s1": n / max(1, n_s1),
          "reduction_ratio": 1 - n / max(1, n_s1 * n_t)}
    if y is not None:
        st["pairs_completeness(recall_ceiling)"] = float(y.sum()) / max(1, n_true)
        st["pair_quality(precision)"] = float(y.mean()) if n else 0.0
    LOG.info(f"blocking[{name}]: " + ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in st.items()))
    return st


def write_id_lists(path: str, A: Store, s_idx: np.ndarray, t_ids: np.ndarray, col: str):
    lists = pd.Series(t_ids).groupby(s_idx).agg(lambda x: ",".join(sorted(set(x)))) if len(s_idx) else pd.Series(dtype=object)
    full = pd.Series("", index=np.arange(A.n), dtype=object)
    full.loc[lists.index] = lists.to_numpy()
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(f"source1_entity_id\t{col}\n")
        fh.writelines(f"{e}\t{v}\n" for e, v in zip(A.cols["entity_id"], full.to_numpy()))
    LOG.info(f"wrote {path} ({A.n} rows, {int((full != '').sum())} non-empty)")


def run_validator(cfg: Config):
    here = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
    cands = [cfg.validator, "validate_submission.py", os.path.join(here, "validate_submission.py"),
             os.path.join(here, "..", "utils", "validate_submission.py")]
    v = next((c for c in cands if c and os.path.exists(c)), None)
    if not v:
        LOG.warning("validate_submission.py not found; skipping validation")
        return
    cmd = [sys.executable, v, "--matching", os.path.join(cfg.out_dir, "matching_results.tsv"),
           "--candidate", os.path.join(cfg.out_dir, "candidate_pairs.tsv"), "--test-dir", os.path.join(cfg.data_dir, "test")]
    LOG.info("running validator: " + " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    LOG.info(r.stdout.strip())
    if r.returncode != 0:
        LOG.error(r.stderr.strip())
        LOG.error("VALIDATION FAILED")


# =============================================================================
# Orchestration
# =============================================================================
def prepare_split(cfg: Config, cache: StageCache, split: str, abbrev: Optional[Dict[str, str]] = None):
    with Timer(f"STEP 1 load [{split}]"):
        data = load_split(cfg, split)
        gt = gt_to_dict(data["gt"])
        tg_raw = pd.concat([data["s2"], data["s3"]], ignore_index=True)
        if abbrev is None:
            abbrev = dict(STATIC_ABBREV)
            abbrev.update(mine_abbreviations(data["s1"], tg_raw, gt))
    key = cache.key("prep-v2", split, data_fingerprint(cfg, split), cfg.sample, cfg.seed, cfg.ngram_range,
                    cfg.hash_bits, sorted(abbrev.items()))

    def build():
        with Timer(f"STEP 2 normalisation [{split}]"):
            c1 = normalise_frame(data["s1"], abbrev, cfg, f"normalise S1 [{split}]")
            c2 = normalise_frame(tg_raw, abbrev, cfg, f"normalise S2+S3 [{split}]")
        with Timer(f"STEP 3 record store + per-country IDF [{split}]"):
            return build_stores(c1, c2, cfg)

    A, B, meta = cache.get_or(f"prep_{split}", key, build)
    return A, B, meta, gt, abbrev, key


def eda_decisions(cfg: Config, A: Store, B: Store, gt: Dict[str, set]):
    c1 = dict(zip(A.cols["entity_id"], A.cols["country_n"]))
    c2 = dict(zip(B.cols["entity_id"], B.cols["country_n"]))
    same = tot = 0
    owner = Counter()
    for a, ms in gt.items():
        for m in ms:
            tot += 1
            same += int(c1.get(a) == c2.get(m))
            owner[m] += 1
    agree = same / tot if tot else 1.0
    uniq = float(np.mean([v == 1 for v in owner.values()])) if owner else 1.0
    n_single = sum(1 for v in gt.values() if not v)
    LOG.info(f"EDA: GT pairs={tot}, country agreement={agree:.4f}, S2/S3 uniqueness={uniq:.4f}, "
             f"singletons={n_single}/{len(gt)} ({n_single / max(1, len(gt)):.3f}), "
             f"match-count dist={dict(sorted(Counter(len(v) for v in gt.values()).items()))}")
    if cfg.country_block is None:
        cfg.country_block = agree >= 0.99
    if cfg.hard_assign is None:
        cfg.hard_assign = uniq >= 0.995
    mx = max((len(v) for v in gt.values()), default=0)
    if mx + 5 > cfg.kmax:
        LOG.info(f"raising kmax from {cfg.kmax} to {mx + 5} (largest GT cluster = {mx})")
        cfg.kmax = mx + 5
    LOG.info(f"decisions: country_block={cfg.country_block}, hard_assign={cfg.hard_assign}, kmax={cfg.kmax}")
    return {"gt_pairs": tot, "country_agreement": agree, "s23_uniqueness": uniq, "singleton_rate": n_single / max(1, len(gt))}


def stage3_features(cfg: Config, pairs: pd.DataFrame, A: Store, B: Store, split: str) -> pd.DataFrame:
    with Timer(f"STEP 6 full features [{split}]"):
        pairs = pairs.reset_index(drop=True)
        extract_features(pairs, A, B, active(sorted(PAIR_FEATURES), cfg), cfg, f"full features [{split}]")
        competition_features(pairs, "p1")
        collective_features(pairs, "p1", B, cfg)
    return pairs


def run(cfg: Config):
    global _PROGRESS
    _PROGRESS = cfg.progress
    TIMINGS.clear()
    setup_logging(cfg.out_dir)
    t_start = time.time()
    LOG.info(f"environment={detect_environment()} cores={os.cpu_count()} n_jobs={cfg.n_jobs} "
             f"feature_threads={cfg.feature_threads} gpu={has_gpu()} sparse_dot_topn={HAS_SDT} jellyfish={HAS_JF}")
    LOG.info("config: " + json.dumps(asdict(cfg), default=str))
    cache = StageCache(cfg)
    metrics = {}
    b2 = cfg.beta ** 2
    cheap = active(CHEAP_FEATURES, cfg)

    # ------------------------------------------------------------------ TRAIN
    A, B, meta, gt, abbrev, pkey = prepare_split(cfg, cache, "train")
    metrics["eda"] = eda_decisions(cfg, A, B, gt)
    gt_keys, gt_size = gt_index(gt, A, B)
    n_true = int(sum(len(v) for v in gt.values()))
    rkey = cache.key(pkey, cfg.blocker, cfg.k_sparse_name, cfg.k_sparse_full, cfg.sparse_threshold, cfg.max_key_block,
                     cfg.lsh_bands, cfg.lsh_rows, cfg.lsh_bucket_cap, cfg.k_lsh, cfg.use_dense, cfg.country_block)
    with Timer("STEP 4 retrieval [train]"):
        pairs = cache.get_or("retrieval_train", rkey, lambda: retrieve(cfg, A, B, meta, cfg.country_block))
        y = label_pairs(pairs, gt_keys, B.n)
        metrics["blocking_raw"] = blocking_stats(pairs, A.n, B.n, y, n_true, "raw_retrieval")
        for c in FLAG_COLS:
            metrics["blocking_raw"][f"recall_{c}"] = float(y[pairs[c].to_numpy() == 1].sum()) / max(1, n_true)

    # entity-level split: pool (OOF) vs held-out country
    if cfg.holdout_country:
        pool_ent = A.cols["country_n"] != norm_text(cfg.holdout_country)
        if pool_ent.all():
            raise SystemExit(f"holdout country '{cfg.holdout_country}' not found in training data")
    else:
        pool_ent = np.ones(A.n, bool)
    eval_ent = ~pool_ent if cfg.holdout_country else pool_ent
    fold2 = entity_fold_map(A.n, cfg.stage2_folds, cfg.seed)
    fold3 = entity_fold_map(A.n, cfg.n_folds, cfg.seed)

    with Timer("STEP 5 cheap features + Stage-2 pruning [train]"):
        pairs["y"] = y
        extract_features(pairs, A, B, cheap, cfg, "cheap features [train]")
        context_features(pairs)
        folds, hold = make_pair_folds(pairs["s1_idx"].to_numpy(), fold2, pool_ent, cfg.stage2_folds)
        X = pairs[cheap].to_numpy(np.float32)
        oof1, m_stage2 = oof_lgb(X, y, folds, hold, cfg.lgb_rounds_stage2, cfg, 31, cfg.neg_sample, "Stage-2 LightGBM folds")
        del X
        in_pool = pool_ent[pairs["s1_idx"].to_numpy()]
        iso1 = fit_isotonic(oof1[in_pool], y[in_pool])
        pairs["p1"] = iso1.predict(oof1).astype(np.float32)
        eps = choose_eps(pairs["p1"].to_numpy()[in_pool], y[in_pool], cfg.prune_target_recall, cfg.prune_eps_floor)
        LOG.info(f"Stage-2 pruning threshold eps={eps:.4f}, kmax={cfg.kmax}")
        pruned = prune(pairs, "p1", eps, cfg.kmax)
        metrics["eps"] = eps
        metrics["blocking_pruned"] = blocking_stats(pruned, A.n, B.n, pruned["y"].to_numpy(), n_true, "after_stage2_pruning")
        lost = pairs.loc[~pairs.index.isin(pruned.index) & in_pool, "p1"].sum()
        LOG.info(f"expected true pairs lost by pruning (sum of pruned p1) = {lost:.1f}")
        del pairs

    pruned = stage3_features(cfg, pruned, A, B, "train")
    cols3 = active([c for g in FEATURE_GROUPS if g != "cross_encoder" for c in FEATURE_GROUPS[g]], cfg)
    y3 = pruned["y"].to_numpy()
    folds3, hold3 = make_pair_folds(pruned["s1_idx"].to_numpy(), fold3, pool_ent, cfg.n_folds)

    # ------------------------------------------------------------------ TEST candidates (before Stage 3 in case CE needs them)
    test = None
    if cfg.run_test:
        At, Bt, metat, gtt, _, pkey_t = prepare_split(cfg, cache, "test", abbrev)
        rkey_t = cache.key(pkey_t, rkey)
        with Timer("STEP 4 retrieval [test]"):
            pt = cache.get_or("retrieval_test", rkey_t, lambda: retrieve(cfg, At, Bt, metat, cfg.country_block))
            metrics["test_blocking_raw"] = blocking_stats(pt, At.n, Bt.n, None, 0, "test_raw_retrieval")
        with Timer("STEP 5 cheap features + pruning [test]"):
            extract_features(pt, At, Bt, cheap, cfg, "cheap features [test]")
            context_features(pt)
            pt["p1"] = iso1.predict(m_stage2.predict(pt[cheap].to_numpy(np.float32), num_threads=cfg.n_jobs)).astype(np.float32)
            pt = prune(pt, "p1", eps, cfg.kmax)
            metrics["test_blocking_pruned"] = blocking_stats(pt, At.n, Bt.n, None, 0, "test_after_stage2_pruning")
        pt = stage3_features(cfg, pt, At, Bt, "test")
        test = (At, Bt, gtt, pt)

    if cfg.use_cross_encoder and "cross_encoder" not in cfg.drop_groups:
        with Timer("STEP 6b cross-encoder"):
            oof_ce, ce_t = cross_encoder_scores(cfg, pruned, y3, folds3, hold3, A, B,
                                                test[3] if test else None, test[0] if test else None, test[1] if test else None)
            pruned["ce_logit"] = oof_ce
            if test is not None:
                test[3]["ce_logit"] = ce_t
            cols3.append("ce_logit")

    with Timer("STEP 7 Stage-3 matcher [train]"):
        X3 = pruned[cols3].to_numpy(np.float32)
        oof3, m_stage3 = oof_lgb(X3, y3, folds3, hold3, cfg.lgb_rounds_stage3, cfg, 63, 1.0, "Stage-3 LightGBM folds")
        in_pool3 = pool_ent[pruned["s1_idx"].to_numpy()]
        iso3 = fit_isotonic(oof3[in_pool3], y3[in_pool3])
        pruned["p3_raw"] = oof3
        pruned["p3"] = iso3.predict(oof3).astype(np.float32)
        imp = pd.DataFrame({"feature": cols3, "gain": m_stage3.feature_importance("gain")})
        imp["group"] = imp["feature"].map(FEATURE_TO_GROUP)
        imp = imp.sort_values("gain", ascending=False)
        imp.to_csv(os.path.join(cfg.out_dir, "feature_importance.csv"), index=False)
        grp = imp.groupby("group")["gain"].sum().sort_values(ascending=False)
        metrics["feature_group_gain_share"] = (grp / grp.sum()).round(4).to_dict()
        LOG.info("top features (gain): " + ", ".join(f"{r.feature}={r.gain:.0f}" for r in imp.head(12).itertuples()))
        LOG.info("gain share by group: " + json.dumps(metrics["feature_group_gain_share"]))

    with Timer("STEP 8 structural constraints [train]"):
        pruned["p"] = apply_structure(pruned, "p3", cfg.hard_assign)

    extra_cols = ["wjac_name", "code_agree", "num_agree", "mutual_best", "cos_name", "p3_raw"]
    with Timer("STEP 9 singleton model + decision tuning [train]"):
        view = EntityView(pruned["s1_idx"], pruned["t_idx"], pruned["p"], y3, {c: pruned[c].to_numpy() for c in extra_cols})
        E = entity_features(view, A.n)
        y_single = (gt_size == 0).astype(np.int8)
        p_empty, m_single = singleton_models(E, y_single, fold3, pool_ent, cfg)
        dec = tune_decision(view, p_empty, gt_size, pool_ent, cfg)
        metrics["decision"] = dec

    with Timer("STEP 10 evaluation [train]"):
        sel = decide(view, p_empty, b2, dec["policy"], dec["T"], dec["s_scale"], dec["tau"], cfg.n_jobs)
        f_all, f_ent = macro_f(view, sel, gt_size, eval_ent, b2)
        metrics["validation_macro_f05"] = f_all
        metrics["validation_per_country_f05"] = {c: float(f_ent[eval_ent & (A.cols["country_n"] == c)].mean())
                                                 for c in sorted(set(A.cols["country_n"][eval_ent]))}
        base = view.p >= 0.5
        metrics["baseline_global_threshold_0.5_f05"] = macro_f(view, base, gt_size, eval_ent, b2)[0]
        LOG.info(f"VALIDATION macro-F0.5 = {f_all:.4f} ({'held-out ' + cfg.holdout_country if cfg.holdout_country else 'out-of-fold'}) "
                 f"| per-country = {metrics['validation_per_country_f05']} | global-threshold-0.5 = {metrics['baseline_global_threshold_0.5_f05']:.4f}")
        # error samples
        ev_rows = eval_ent[view.s]
        fp = np.where(sel & ~view.y & ev_rows)[0][:200]
        sel_keys = view.s[sel].astype(np.int64) * B.n + view.t[sel]
        fn_keys = np.setdiff1d(gt_keys, sel_keys)
        fn_keys = fn_keys[eval_ent[fn_keys // B.n]][:200]
        la = lambda k: f"{A.cols['raw_name'][k]} | {A.cols['raw_addr'][k]}"  # noqa: E731
        lb = lambda k: f"{B.cols['raw_name'][k]} | {B.cols['raw_addr'][k]}"  # noqa: E731
        with open(os.path.join(cfg.out_dir, "error_samples.txt"), "w", encoding="utf-8") as fh:
            fh.write("FALSE POSITIVES (wrong merges)\n")
            fh.writelines(f"  {la(view.s[r])}  <->  {lb(view.t[r])}\n" for r in fp)
            fh.write("\nFALSE NEGATIVES (missed matches)\n")
            fh.writelines(f"  {la(k // B.n)}  <->  {lb(k % B.n)}\n" for k in fn_keys)
        metrics["n_false_positive_pairs"] = int((sel & ~view.y & ev_rows).sum())
        metrics["n_false_negative_pairs"] = int(len(np.setdiff1d(gt_keys[eval_ent[gt_keys // B.n]], sel_keys)))
        pruned[["s1_idx", "t_idx", "y", "p1", "p3", "p"]].to_parquet(os.path.join(cfg.work_dir, "train_oof_pairs.parquet"))

    # ------------------------------------------------------------------ TEST inference
    if test is not None:
        At, Bt, gtt, pt = test
        with Timer("STEP 7-9 test inference"):
            pt["p3_raw"] = m_stage3.predict(pt[cols3].to_numpy(np.float32), num_threads=cfg.n_jobs)
            pt["p3"] = iso3.predict(pt["p3_raw"].to_numpy()).astype(np.float32)
            pt["p"] = apply_structure(pt, "p3", cfg.hard_assign)
            vt = EntityView(pt["s1_idx"], pt["t_idx"], pt["p"], None, {c: pt[c].to_numpy() for c in extra_cols})
            pe_t = m_single.predict(entity_features(vt, At.n), num_threads=cfg.n_jobs)
            sel_t = decide(vt, pe_t, b2, dec["policy"], dec["T"], dec["s_scale"], dec["tau"], cfg.n_jobs)
        with Timer("STEP 11 write outputs + validator"):
            tid = Bt.cols["entity_id"]
            write_id_lists(os.path.join(cfg.out_dir, "candidate_pairs.tsv"), At, pt["s1_idx"].to_numpy(),
                           tid[pt["t_idx"].to_numpy()], "candidate_entity_ids")
            write_id_lists(os.path.join(cfg.out_dir, "matching_results.tsv"), At, vt.s[sel_t], tid[vt.t[sel_t]], "matched_entity_ids")
            pt[["s1_idx", "t_idx", "p1", "p3", "p"]].to_parquet(os.path.join(cfg.work_dir, "test_scored_pairs.parquet"))
            metrics["test_predicted_nonempty"] = int(len(np.unique(vt.s[sel_t])))
            metrics["test_avg_candidates_per_s1"] = len(pt) / max(1, At.n)
            if gtt:   # synthetic smoke tests only (hidden ground truth)
                kt, st_ = gt_index(gtt, At, Bt)
                vt.y = np.isin(vt.s.astype(np.int64) * Bt.n + vt.t, kt)
                all_e = np.ones(At.n, bool)
                fh_, fe_ = macro_f(vt, sel_t, st_, all_e, b2)
                metrics["hidden_test_macro_f05"] = fh_
                metrics["hidden_test_per_country"] = {c: float(fe_[At.cols["country_n"] == c].mean()) for c in sorted(set(At.cols["country_n"]))}
                LOG.info(f"HIDDEN TEST macro-F0.5 = {fh_:.4f} per-country={metrics['hidden_test_per_country']}")
            run_validator(cfg)

    metrics["timings_sec"] = {k: round(v, 2) for k, v in TIMINGS.items()}
    metrics["total_sec"] = round(time.time() - t_start, 2)
    with open(os.path.join(cfg.out_dir, "metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2, default=float)
    LOG.info("timings (s): " + json.dumps(metrics["timings_sec"]))
    LOG.info(f"ALL DONE in {metrics['total_sec']:.1f}s")
    return metrics


# =============================================================================
# STEP 12 — CLI
# =============================================================================
def parse_args(argv=None) -> Config:
    ap = argparse.ArgumentParser(description="Business Entity Resolution pipeline (v2)")
    ap.add_argument("--data-dir", default="dataset")
    ap.add_argument("--out-dir", default="output")
    ap.add_argument("--work-dir", default="work")
    ap.add_argument("--sample", type=float, default=1.0, help="fraction of S1 entities (dev runs only)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--holdout-country", default=None, help="leave-one-country-out validation, e.g. India")
    ap.add_argument("--no-test", action="store_true", help="validation only")
    ap.add_argument("--n-jobs", type=int, default=0, help="worker processes / library threads (0 = all cores)")
    ap.add_argument("--feature-threads", type=int, default=0, help="threads over feature chunks (0 = n-jobs)")
    ap.add_argument("--chunk-size", type=int, default=200_000, help="pairs per feature chunk (lower = less RAM)")
    ap.add_argument("--no-progress", action="store_true")
    ap.add_argument("--no-cache", action="store_true", help="disable the STEP 2-4 stage cache")
    ap.add_argument("--cache-dir", default="work/cache")
    ap.add_argument("--blocker", choices=["tfidf", "lsh", "both"], default="tfidf")
    ap.add_argument("--k-sparse-name", type=int, default=15)
    ap.add_argument("--k-sparse-full", type=int, default=10)
    ap.add_argument("--lsh-bands", type=int, default=12)
    ap.add_argument("--lsh-rows", type=int, default=4)
    ap.add_argument("--k-lsh", type=int, default=20)
    ap.add_argument("--kmax", type=int, default=10)
    ap.add_argument("--prune-target-recall", type=float, default=0.995)
    ap.add_argument("--neg-sample", type=float, default=0.3)
    ap.add_argument("--drop-groups", default="", help="comma-separated feature groups to ablate")
    ap.add_argument("--use-dense", action="store_true")
    ap.add_argument("--use-cross-encoder", action="store_true")
    ap.add_argument("--country-block", choices=["auto", "on", "off"], default="auto")
    ap.add_argument("--hard-assign", choices=["auto", "on", "off"], default="auto")
    ap.add_argument("--validator", default="utils/validate_submission.py")
    a = ap.parse_args(argv)
    tri = {"auto": None, "on": True, "off": False}
    return Config(data_dir=a.data_dir, out_dir=a.out_dir, work_dir=a.work_dir, sample=a.sample, seed=a.seed,
                  n_folds=a.n_folds, holdout_country=a.holdout_country, run_test=not a.no_test, n_jobs=a.n_jobs,
                  feature_threads=a.feature_threads, chunk_size=a.chunk_size, progress=not a.no_progress,
                  use_cache=not a.no_cache, cache_dir=a.cache_dir, blocker=a.blocker, k_sparse_name=a.k_sparse_name,
                  k_sparse_full=a.k_sparse_full, lsh_bands=a.lsh_bands, lsh_rows=a.lsh_rows, k_lsh=a.k_lsh, kmax=a.kmax,
                  prune_target_recall=a.prune_target_recall, neg_sample=a.neg_sample,
                  drop_groups=tuple(g.strip() for g in a.drop_groups.split(",") if g.strip()),
                  use_dense=a.use_dense, use_cross_encoder=a.use_cross_encoder, country_block=tri[a.country_block],
                  hard_assign=tri[a.hard_assign], validator=a.validator)


if __name__ == "__main__":
    run(parse_args())
