#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ber_pipeline.py — Business Entity Resolution (ML Challenge 2026), single-file pipeline.

Stages (each is labelled "STEP n" in the code and in the log):
  STEP 0  Config, environment detection, logging
  STEP 1  Load TSV data
  STEP 2  Normalisation (Unicode, tokens, numbers, landmarks, mined abbreviations)
  STEP 3  Per-country transductive IDF
  STEP 4  Multi-view retrieval (char TF-IDF sparse top-k, exact keys, optional dense)
  STEP 5  Cheap pair features + Stage-2 learned pruning  ->  candidate_pairs.tsv
  STEP 6  Full pair features (+ reverse/competition features, optional cross-encoder)
  STEP 7  Stage-3 matcher (LightGBM, out-of-fold, isotonic calibration)
  STEP 8  Structural constraints (each S2/S3 record -> at most one S1 entity)
  STEP 9  Expected-F0.5 set decision with explicit singleton model
  STEP 10 Evaluation (macro F0.5, per-country, blocking diagnostics)
  STEP 11 Write outputs and run the official validator
  STEP 12 CLI entry point

Quick start:
  python ber_pipeline.py --data-dir dataset --out-dir output            # full run
  python ber_pipeline.py --data-dir dataset --out-dir output --sample 0.2   # fast dev run
  python ber_pipeline.py --data-dir dataset --holdout-country India --no-test  # leave-one-country-out check

Only MIT/BSD/Apache libraries are used. No external data or services are ever contacted.
"""

# =============================================================================
# STEP 0 — imports, configuration, environment detection, logging
# =============================================================================
import argparse
import json
import logging
import math
import os
import re
import subprocess
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GroupKFold

import lightgbm as lgb
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler, Levenshtein

try:
    from sparse_dot_topn import sp_matmul_topn
    HAS_SDT = True
except Exception:  # pragma: no cover
    HAS_SDT = False

LOG = logging.getLogger("ber")


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
    sample: float = 1.0                 # fraction of S1 entities to keep (dev runs)
    n_folds: int = 5
    holdout_country: Optional[str] = None   # leave-one-country-out validation
    run_test: bool = True

    # STEP 4 retrieval
    ngram_range: Tuple[int, int] = (3, 4)
    k_sparse_name: int = 15
    k_sparse_full: int = 10
    sparse_threshold: float = 0.15
    max_key_block: int = 100            # skip exact-key blocks larger than this
    use_dense: bool = False
    dense_model: str = "intfloat/multilingual-e5-small"   # MIT licence
    k_dense: int = 10
    query_chunk: int = 20000

    # STEP 5 pruning
    prune_target_recall: float = 0.995  # retained fraction of retrievable GT pairs
    prune_eps_floor: float = 0.005
    kmax: int = 10

    # STEP 6/7 matcher
    use_cross_encoder: bool = False
    ce_model: str = "microsoft/mdeberta-v3-base"          # MIT licence
    ce_max_train_pairs: int = 150000
    ce_epochs: int = 1
    lgb_rounds_stage2: int = 400
    lgb_rounds_stage3: int = 800

    # STEP 8 structure (auto-set from EDA unless forced)
    hard_assign: Optional[bool] = None
    country_block: Optional[bool] = None

    # STEP 9 decision
    beta: float = 0.5
    temperature_grid: Tuple[float, ...] = (0.7, 0.85, 1.0, 1.2, 1.5)
    singleton_scale_grid: Tuple[float, ...] = (0.7, 0.85, 1.0, 1.15, 1.3)
    threshold_grid: Tuple[float, ...] = (0.4, 0.5, 0.6, 0.7, 0.8)   # competing "global threshold" policy
    tune_max_entities: int = 40000        # subsample of evaluated entities used for decision tuning

    validator: str = "utils/validate_submission.py"

    def __post_init__(self):
        os.makedirs(self.out_dir, exist_ok=True)
        os.makedirs(self.work_dir, exist_ok=True)


def setup_logging(out_dir: str):
    fmt = "%(asctime)s | %(levelname)s | %(message)s"
    logging.basicConfig(level=logging.INFO, format=fmt,
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
        LOG.info(f"----- {self.name} done in {time.time() - self.t:.1f}s")


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
            out["gt"] = out["gt"][out["gt"].source1_entity_id.isin(keep)].reset_index(drop=True)
            # keep only S2/S3 rows that are matches of the sampled S1 + random negatives
            matched = set()
            for v in out["gt"].matched_entity_ids:
                matched.update([x for x in v.split(",") if x])
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
    return {r.source1_entity_id: set(x for x in r.matched_entity_ids.split(",") if x) for r in gt.itertuples()}


# =============================================================================
# STEP 2 — normalisation
# =============================================================================
_WS = re.compile(r"\s+")
_NONALNUM = re.compile(r"[^0-9a-z ]+")
_DIGITS = re.compile(r"\d+")
LANDMARK_CUES = ("near", "nr", "opp", "opposite", "behind", "beside", "next to", "adjacent to",
                 "pres de", "face a", "a cote de", "en face")
LEGAL_SUFFIXES = {"ltd", "limited", "pvt", "private", "inc", "incorporated", "corp", "corporation", "co",
                  "company", "llc", "l l c", "llp", "l l p", "plc", "sarl", "s a r l", "sas", "s a s", "sasu",
                  "sa", "s a", "eurl", "e u r l", "sci", "snc", "gmbh", "ag", "bv", "nv", "the"}
STATIC_ABBREV = {  # ordinary domain knowledge, not an external lookup
    "st": "street", "rd": "road", "ave": "avenue", "av": "avenue", "blvd": "boulevard", "bd": "boulevard",
    "dr": "drive", "ln": "lane", "hwy": "highway", "pl": "place", "sq": "square", "ct": "court", "pkwy": "parkway",
    "mkt": "market", "nr": "near", "opp": "opposite", "bldg": "building", "apt": "apartment", "fl": "floor",
    "ste": "suite", "n": "north", "s": "south", "e": "east", "w": "west", "pvt": "private", "ltd": "limited",
    "corp": "corporation", "inc": "incorporated", "co": "company", "intl": "international", "mfg": "manufacturing",
    "svc": "service", "svcs": "services", "ind": "industries", "ents": "enterprises", "bros": "brothers",
    "ctr": "center", "ctre": "centre", "mgmt": "management", "assoc": "associates", "tech": "technologies",
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
    """Learn short->long token rewrites from matched pairs (train only). Generalises beyond the static list."""
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
                        if len(short) <= 5 and len(long_) > len(short) and (long_.startswith(short)
                                                                              or JaroWinkler.similarity(short, long_) > 0.85):
                            cnt[(short, long_)] += 1
    mined = {}
    for (s, l), c in cnt.most_common():
        if c >= min_count and s not in mined:
            mined[s] = l
    LOG.info(f"mined {len(mined)} abbreviation rewrites, e.g. {list(mined.items())[:8]}")
    return mined


def expand_abbrev(text: str, table: Dict[str, str]) -> str:
    return " ".join(table.get(t, t) for t in text.split())


def acronym_of(tokens: List[str]) -> str:
    return "".join(t[0] for t in tokens if t and not t.isdigit())


def split_landmark(addr: str) -> Tuple[str, str]:
    for cue in LANDMARK_CUES:
        i = addr.find(" " + cue + " ")
        if i == -1 and addr.startswith(cue + " "):
            i = 0
        if i != -1:
            # landmark = cue phrase up to next comma-equivalent (we lost commas; take next 3 tokens)
            after = addr[i:].split()
            lm = " ".join(after[:4])
            core = (addr[:i] + " " + " ".join(after[4:])).strip()
            return core, lm
    return addr, ""


def normalise_records(df: pd.DataFrame, abbrev: Dict[str, str]) -> pd.DataFrame:
    """Adds canonical columns used by every later stage (kept alongside raw text)."""
    df = df.copy()
    df["country_n"] = df["country"].map(norm_text)
    name_n = df["business_name"].map(norm_text).map(lambda t: expand_abbrev(t, abbrev))
    addr_n = df["business_address"].map(norm_text).map(lambda t: expand_abbrev(t, abbrev))
    df["name_n"] = name_n
    df["addr_n"] = addr_n

    core, suf, acr, toks = [], [], [], []
    for n in name_n:
        t = n.split()
        c = [x for x in t if x not in LEGAL_SUFFIXES]
        if not c:
            c = t
        core.append(" ".join(c))
        suf.append(" ".join(x for x in t if x in LEGAL_SUFFIXES))
        acr.append(acronym_of(c))
        toks.append(frozenset(c))
    df["core"] = core
    df["suffix"] = suf
    df["acr"] = acr
    df["core_toks"] = toks
    df["core_nospace"] = [c.replace(" ", "") for c in core]

    acore, lm, codes, nums, atoks, lmtoks = [], [], [], [], [], []
    for a in addr_n:
        c, l = split_landmark(a)
        acore.append(c)
        lm.append(l)
        d = _DIGITS.findall(a)
        codes.append(frozenset(x for x in d if len(x) >= 4))
        nums.append(frozenset(x.lstrip("0") or "0" for x in d if len(x) < 4))
        atoks.append(frozenset(x for x in c.split() if not x.isdigit()))
        lmtoks.append(frozenset(l.split()))
    df["addr_core"] = acore
    df["landmark"] = lm
    df["codes"] = codes
    df["nums"] = nums
    df["addr_toks"] = atoks
    df["lm_toks"] = lmtoks
    df["full_n"] = df["name_n"] + " " + df["addr_n"]
    return df


# =============================================================================
# STEP 3 — per-country transductive IDF
# =============================================================================
class IDF:
    """idf_c(t) = log((N_c+1)/(df_c(t)+1)) + 1, computed on ALL records of a country in the current split."""

    def __init__(self):
        self.tables: Dict[str, Dict[str, float]] = {}
        self.default: Dict[str, float] = {}

    def fit(self, frames: List[pd.DataFrame]):
        allr = pd.concat([f[["country_n", "name_n", "addr_n"]] for f in frames], ignore_index=True)
        for c, g in allr.groupby("country_n"):
            df_cnt = Counter()
            n = len(g)
            for a, b in zip(g.name_n, g.addr_n):
                df_cnt.update(set(a.split()) | set(b.split()))
            self.tables[c] = {t: math.log((n + 1) / (d + 1)) + 1.0 for t, d in df_cnt.items()}
            self.default[c] = math.log((n + 1) / 1.0) + 1.0
        return self

    def w(self, country: str, tok: str) -> float:
        tab = self.tables.get(country)
        if tab is None:
            return 1.0
        return tab.get(tok, self.default[country])


def wjaccard(a: frozenset, b: frozenset, tab: Dict[str, float], dflt: float) -> float:
    if not a or not b:
        return 0.0
    inter = sum(tab.get(t, dflt) for t in a & b)
    union = sum(tab.get(t, dflt) for t in a | b)
    return inter / union if union > 0 else 0.0


def rarest_idf(toks, tab, dflt) -> float:
    return max((tab.get(t, dflt) for t in toks), default=0.0)


# =============================================================================
# STEP 4 — multi-view retrieval
# =============================================================================
def sparse_topn(Q: sp.csr_matrix, T: sp.csr_matrix, k: int, thr: float, chunk: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact top-k cosine (rows already L2-normalised). Returns (query_row, target_row, score)."""
    rows, cols, vals = [], [], []
    TT = T.T.tocsr()
    for s in range(0, Q.shape[0], chunk):
        q = Q[s:s + chunk]
        if HAS_SDT:
            C = sp_matmul_topn(q, TT, top_n=k, threshold=thr, sort=True, n_threads=os.cpu_count() or 1).tocsr()
        else:  # fallback: dense chunk (slower, more memory)
            C = (q @ TT).tocsr()
            C.data[C.data < thr] = 0
            C.eliminate_zeros()
            out = sp.lil_matrix(C.shape, dtype=np.float32)
            for i in range(C.shape[0]):
                r = C.getrow(i)
                if r.nnz > k:
                    top = np.argpartition(-r.data, k)[:k]
                    out[i, r.indices[top]] = r.data[top]
                elif r.nnz:
                    out[i, r.indices] = r.data
            C = out.tocsr()
        rr = np.repeat(np.arange(C.shape[0]) + s, np.diff(C.indptr))
        rows.append(rr)
        cols.append(C.indices.copy())
        vals.append(C.data.astype(np.float32))
    if not rows:
        return np.array([], int), np.array([], int), np.array([], np.float32)
    return np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)


def dense_topn(s1_text: List[str], t_text: List[str], k: int, model_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Optional dense view (sentence-transformers). Uses FAISS if present, else chunked numpy."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    pre = "query: " if "e5" in model_name else ""
    E1 = model.encode([pre + t for t in s1_text], batch_size=256, normalize_embeddings=True, show_progress_bar=False)
    E2 = model.encode([pre + t for t in t_text], batch_size=256, normalize_embeddings=True, show_progress_bar=False)
    E1, E2 = E1.astype(np.float32), E2.astype(np.float32)
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
            S = E1[s:s + 4096] @ E2.T
            ii = np.argpartition(-S, k - 1, axis=1)[:, :k]
            D[s:s + 4096] = np.take_along_axis(S, ii, 1)
            I[s:s + 4096] = ii
    rows = np.repeat(np.arange(len(E1)), k)
    return rows, I.ravel(), D.ravel()


def key_blocks(s1: pd.DataFrame, tg: pd.DataFrame, idf: IDF, country: str, max_block: int) -> Tuple[np.ndarray, np.ndarray]:
    """Exact keys: (postal-like code, 3-char prefix of rarest core token) and (acronym, street number)."""
    tab, dflt = idf.tables.get(country, {}), idf.default.get(country, 1.0)

    def keys_of(row):
        ks = []
        toks = list(row.core_toks)
        if toks:
            rare = max(toks, key=lambda t: tab.get(t, dflt))
            for c in row.codes:
                ks.append(("pc", c, rare[:3]))
        if len(row.acr) >= 2:
            for n in row.nums:
                ks.append(("an", row.acr, n))
        if len(row.core_nospace) >= 6:
            ks.append(("cn", row.core_nospace[:8], ""))
        return ks

    index = defaultdict(list)
    for i, r in enumerate(tg.itertuples()):
        for k in keys_of(r):
            index[k].append(i)
    qi, ti = [], []
    for i, r in enumerate(s1.itertuples()):
        seen = set()
        for k in keys_of(r):
            lst = index.get(k)
            if lst and len(lst) <= max_block:
                for t in lst:
                    if t not in seen:
                        seen.add(t)
                        qi.append(i)
                        ti.append(t)
    return np.array(qi, int), np.array(ti, int)


def retrieve(cfg: Config, s1: pd.DataFrame, tg: pd.DataFrame, idf: IDF, country_block: bool) -> pd.DataFrame:
    """Returns a raw candidate table with one row per (S1 entity, S2/S3 record) and per-view scores."""
    groups = [("__all__", s1.index.values, tg.index.values)] if not country_block else [
        (c, s1.index[s1.country_n == c].values, tg.index[tg.country_n == c].values) for c in sorted(s1.country_n.unique())]
    parts = []
    for c, qi_all, ti_all in groups:
        if len(qi_all) == 0 or len(ti_all) == 0:
            LOG.info(f"  block '{c}': S1={len(qi_all)} targets={len(ti_all)} -> skipped")
            continue
        q = s1.loc[qi_all]
        t = tg.loc[ti_all]
        views = []

        def add_view(r, cidx, v, col):
            if len(r):
                d = pd.DataFrame({"q": r.astype(np.int64), "t": cidx.astype(np.int64)})
                for cc in ("sc_name", "sc_full", "sc_dense", "key_hit"):
                    d[cc] = np.float32(0)
                d[col] = v.astype(np.float32)
                views.append(d)

        # view 1: char n-gram TF-IDF on the name
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=cfg.ngram_range, min_df=1, sublinear_tf=True, dtype=np.float32)
        vec.fit(pd.concat([q.name_n, t.name_n]))
        r, cidx, v = sparse_topn(vec.transform(q.name_n), vec.transform(t.name_n), cfg.k_sparse_name, cfg.sparse_threshold, cfg.query_chunk)
        add_view(r, cidx, v, "sc_name")
        # view 2: char n-gram TF-IDF on name + address
        vec2 = TfidfVectorizer(analyzer="char_wb", ngram_range=cfg.ngram_range, min_df=1, sublinear_tf=True, dtype=np.float32)
        vec2.fit(pd.concat([q.full_n, t.full_n]))
        r, cidx, v = sparse_topn(vec2.transform(q.full_n), vec2.transform(t.full_n), cfg.k_sparse_full, cfg.sparse_threshold, cfg.query_chunk)
        add_view(r, cidx, v, "sc_full")
        # view 3: exact key blocks
        r, cidx = key_blocks(q, t, idf, c, cfg.max_key_block)
        add_view(r, cidx, np.ones(len(r), np.float32), "key_hit")
        # view 4 (optional): dense multilingual embeddings
        if cfg.use_dense:
            try:
                r, cidx, v = dense_topn((q.business_name + " | " + q.business_address).tolist(),
                                        (t.business_name + " | " + t.business_address).tolist(), cfg.k_dense, cfg.dense_model)
                add_view(r, cidx, v, "sc_dense")
            except Exception as ex:
                LOG.warning(f"dense view failed ({ex}); continuing without it")
        n_pairs = 0
        if views:
            merged = pd.concat(views, ignore_index=True).groupby(["q", "t"], sort=False, as_index=False).max()
            n_pairs = len(merged)
            parts.append(pd.DataFrame({"s1_idx": qi_all[merged.q.values], "t_idx": ti_all[merged.t.values],
                                       "sc_name": merged.sc_name.values, "sc_full": merged.sc_full.values,
                                       "sc_dense": merged.sc_dense.values, "key_hit": merged.key_hit.values.astype(np.int8)}))
        LOG.info(f"  block '{c}': S1={len(qi_all)} targets={len(ti_all)} raw pairs={n_pairs} "
                 f"({n_pairs / max(1, len(qi_all)):.1f}/entity)")
    if not parts:
        return pd.DataFrame(columns=["s1_idx", "t_idx", "sc_name", "sc_full", "sc_dense", "key_hit"])
    return pd.concat(parts, ignore_index=True)


# =============================================================================
# STEP 5 — cheap pair features and Stage-2 pruning
# =============================================================================
def add_rank_features(pairs: pd.DataFrame, col: str, prefix: str):
    g = pairs.groupby("s1_idx")[col]
    pairs[f"{prefix}_rank"] = g.rank(ascending=False, method="first").astype(np.float32)
    pairs[f"{prefix}_margin"] = (g.transform("max") - pairs[col]).astype(np.float32)


def tri_agree(a: frozenset, b: frozenset) -> int:
    """Three-valued numeric agreement: 1 agree, 0 missing on a side, -1 conflict."""
    if not a or not b:
        return 0
    return 1 if a & b else -1


def cheap_features(pairs: pd.DataFrame, s1: pd.DataFrame, tg: pd.DataFrame, idf: IDF) -> pd.DataFrame:
    a = s1.loc[pairs.s1_idx.values]
    b = tg.loc[pairs.t_idx.values]
    ctry = a.country_n.values
    tabs = {c: idf.tables.get(c, {}) for c in np.unique(ctry)}
    dfl = {c: idf.default.get(c, 1.0) for c in np.unique(ctry)}
    at, bt = a.core_toks.values, b.core_toks.values
    pairs["wjac_name"] = np.array([wjaccard(x, y, tabs[c], dfl[c]) for x, y, c in zip(at, bt, ctry)], np.float32)
    pairs["tsr_name"] = process.cpdist(a.core.tolist(), b.core.tolist(), scorer=fuzz.token_set_ratio, workers=-1).astype(np.float32) / 100
    pairs["code_agree"] = np.array([tri_agree(x, y) for x, y in zip(a.codes.values, b.codes.values)], np.int8)
    pairs["num_agree"] = np.array([tri_agree(x, y) for x, y in zip(a.nums.values, b.nums.values)], np.int8)
    pairs["core_equal"] = (a.core_nospace.values == b.core_nospace.values).astype(np.int8)
    pairs["src_s3"] = (b.source.values == 3).astype(np.int8)
    pairs["n_cand"] = pairs.groupby("s1_idx")["t_idx"].transform("size").astype(np.float32)
    add_rank_features(pairs, "sc_name", "name")
    add_rank_features(pairs, "sc_full", "full")
    if "sc_dense" in pairs:
        add_rank_features(pairs, "sc_dense", "dense")
    return pairs


CHEAP_COLS = ["sc_name", "sc_full", "sc_dense", "key_hit", "wjac_name", "tsr_name", "code_agree", "num_agree",
              "core_equal", "src_s3", "n_cand", "name_rank", "name_margin", "full_rank", "full_margin",
              "dense_rank", "dense_margin"]


def lgb_params(seed: int, leaves=63) -> dict:
    return dict(objective="binary", learning_rate=0.05, num_leaves=leaves, min_child_samples=40,
                feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1,
                seed=seed, num_threads=os.cpu_count() or 1)


def make_folds(pairs: pd.DataFrame, s1: pd.DataFrame, cfg: Config) -> List[Tuple[np.ndarray, np.ndarray]]:
    """GroupKFold by S1 entity, or a single leave-one-country-out fold."""
    if cfg.holdout_country:
        ctry = s1.loc[pairs.s1_idx.values, "country_n"].values
        h = norm_text(cfg.holdout_country)
        tr, va = np.where(ctry != h)[0], np.where(ctry == h)[0]
        if len(va) == 0:
            raise SystemExit(f"holdout country '{cfg.holdout_country}' has no pairs")
        return [(tr, va)]
    gkf = GroupKFold(n_splits=cfg.n_folds)
    return list(gkf.split(pairs, groups=pairs.s1_idx.values))


def oof_lgb(pairs: pd.DataFrame, cols: List[str], y: np.ndarray, folds, rounds: int, seed: int, leaves=63):
    """Out-of-fold LightGBM probabilities + a model fitted on the union of fold-train sets."""
    oof = np.full(len(pairs), np.nan, np.float32)
    X = pairs[cols].astype(np.float32)
    for k, (tr, va) in enumerate(folds):
        m = lgb.train(lgb_params(seed + k, leaves), lgb.Dataset(X.iloc[tr], y[tr]), num_boost_round=rounds)
        oof[va] = m.predict(X.iloc[va])
    all_tr = np.unique(np.concatenate([tr for tr, _ in folds]))
    final = lgb.train(lgb_params(seed, leaves), lgb.Dataset(X.iloc[all_tr], y[all_tr]), num_boost_round=rounds)
    return oof, final


def fit_isotonic(p: np.ndarray, y: np.ndarray) -> IsotonicRegression:
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso.fit(p, y)
    return iso


def label_pairs(pairs: pd.DataFrame, s1: pd.DataFrame, tg: pd.DataFrame, gt: Dict[str, set]) -> np.ndarray:
    a = s1.loc[pairs.s1_idx.values, "entity_id"].values
    b = tg.loc[pairs.t_idx.values, "entity_id"].values
    return np.array([1 if (x in gt and yv in gt[x]) else 0 for x, yv in zip(a, b)], np.int8)


def choose_eps(p1: np.ndarray, y: np.ndarray, target: float, floor: float) -> float:
    """Largest eps keeping >= target of the retrievable true pairs (Stage-2 recall constraint)."""
    pos = np.sort(p1[y == 1])
    if len(pos) == 0:
        return floor
    q = max(0.0, 1.0 - target)
    eps = float(np.quantile(pos, q))
    return max(floor, min(eps, 0.5))


def prune(pairs: pd.DataFrame, pcol: str, eps: float, kmax: int) -> pd.DataFrame:
    keep = pairs[pairs[pcol] >= eps].copy()
    keep["_r"] = keep.groupby("s1_idx")[pcol].rank(ascending=False, method="first")
    keep = keep[keep._r <= kmax].drop(columns="_r")   # original index kept on purpose (pruning diagnostics)
    return keep


# =============================================================================
# STEP 6 — full pair features (+ competition features, optional cross-encoder)
# =============================================================================
def full_features(pairs: pd.DataFrame, s1: pd.DataFrame, tg: pd.DataFrame, idf: IDF) -> pd.DataFrame:
    a = s1.loc[pairs.s1_idx.values]
    b = tg.loc[pairs.t_idx.values]
    ctry = a.country_n.values
    tabs = {c: idf.tables.get(c, {}) for c in np.unique(ctry)}
    dfl = {c: idf.default.get(c, 1.0) for c in np.unique(ctry)}
    an, bn = a.core.tolist(), b.core.tolist()
    aa, ba = a.addr_core.tolist(), b.addr_core.tolist()

    def cp(x, y, scorer):
        return process.cpdist(x, y, scorer=scorer, workers=-1).astype(np.float32)

    pairs["jw_name"] = cp(an, bn, JaroWinkler.normalized_similarity)
    pairs["lev_name"] = cp(an, bn, Levenshtein.normalized_similarity)
    pairs["tsort_name"] = cp(an, bn, fuzz.token_sort_ratio) / 100
    pairs["partial_name"] = cp(an, bn, fuzz.partial_ratio) / 100
    pairs["jw_nospace"] = cp(a.core_nospace.tolist(), b.core_nospace.tolist(), JaroWinkler.normalized_similarity)
    pairs["lev_addr"] = cp(aa, ba, Levenshtein.normalized_similarity)
    pairs["tsr_addr"] = cp(aa, ba, fuzz.token_set_ratio) / 100
    pairs["partial_addr"] = cp(aa, ba, fuzz.partial_ratio) / 100
    pairs["wjac_addr"] = np.array([wjaccard(x, y, tabs[c], dfl[c]) for x, y, c in zip(a.addr_toks.values, b.addr_toks.values, ctry)], np.float32)
    pairs["wjac_lm"] = np.array([wjaccard(x, y, tabs[c], dfl[c]) for x, y, c in zip(a.lm_toks.values, b.lm_toks.values, ctry)], np.float32)
    pairs["shared_rare"] = np.array([rarest_idf(x & y, tabs[c], dfl[c]) for x, y, c in zip(a.core_toks.values, b.core_toks.values, ctry)], np.float32)
    pairs["unshared_rare"] = np.array([rarest_idf(x ^ y, tabs[c], dfl[c]) for x, y, c in zip(a.core_toks.values, b.core_toks.values, ctry)], np.float32)
    pairs["rare_diff"] = pairs["shared_rare"] - pairs["unshared_rare"]
    aacr, bacr = a.acr.values, b.acr.values
    pairs["acr_match"] = np.array([1 if (len(x) >= 2 and (x == y or x == yb or y == xb)) else 0
                                   for x, y, xb, yb in zip(aacr, bacr, a.core_nospace.values, b.core_nospace.values)], np.int8)
    pairs["suffix_conflict"] = np.array([1 if (x and y and x != y) else 0 for x, y in zip(a.suffix.values, b.suffix.values)], np.int8)
    pairs["both_codes"] = np.array([int(bool(x) and bool(y)) for x, y in zip(a.codes.values, b.codes.values)], np.int8)
    pairs["ntok_diff"] = np.abs(np.array([len(x) for x in a.core_toks.values]) - np.array([len(x) for x in b.core_toks.values])).astype(np.float32)
    pairs["len_ratio"] = np.array([min(len(x), len(y)) / max(1, max(len(x), len(y))) for x, y in zip(an, bn)], np.float32)
    pairs["a_has_addr"] = np.array([int(len(x) > 0) for x in aa], np.int8)
    pairs["b_has_addr"] = np.array([int(len(x) > 0) for x in ba], np.int8)
    return pairs


def competition_features(pairs: pd.DataFrame, pcol: str) -> pd.DataFrame:
    """Reverse view: how does this S1 entity rank among all S1 entities competing for the same S2/S3 record?"""
    g = pairs.groupby("t_idx")[pcol]
    pairs["rev_rank"] = g.rank(ascending=False, method="first").astype(np.float32)
    pairs["rev_n"] = g.transform("size").astype(np.float32)
    pairs["rev_margin"] = (g.transform("max") - pairs[pcol]).astype(np.float32)
    fr = pairs.groupby("s1_idx")[pcol].rank(ascending=False, method="first")
    pairs["p1_rank"] = fr.astype(np.float32)
    pairs["p1_margin"] = (pairs.groupby("s1_idx")[pcol].transform("max") - pairs[pcol]).astype(np.float32)
    pairs["mutual_best"] = ((fr == 1) & (pairs.rev_rank == 1)).astype(np.int8)
    return pairs


FULL_COLS = CHEAP_COLS + ["p1", "jw_name", "lev_name", "tsort_name", "partial_name", "jw_nospace", "lev_addr", "tsr_addr",
                          "partial_addr", "wjac_addr", "wjac_lm", "shared_rare", "unshared_rare", "rare_diff", "acr_match",
                          "suffix_conflict", "both_codes", "ntok_diff", "len_ratio", "a_has_addr", "b_has_addr",
                          "rev_rank", "rev_n", "rev_margin", "p1_rank", "p1_margin", "mutual_best"]


def cross_encoder_scores(cfg: Config, train_pairs: pd.DataFrame, y: np.ndarray, folds, s1: pd.DataFrame, tg: pd.DataFrame,
                         test_pairs: Optional[pd.DataFrame], s1t: Optional[pd.DataFrame], tgt: Optional[pd.DataFrame]):
    """Optional Ditto-style cross-encoder. Returns (oof_logits_train, logits_test). Requires GPU in practice."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(cfg.ce_model)

    def texts(p, A, B):
        aa = A.loc[p.s1_idx.values]
        bb = B.loc[p.t_idx.values]
        return [f"[COL] name [VAL] {n1} [COL] address [VAL] {a1} [SEP] [COL] name [VAL] {n2} [COL] address [VAL] {a2}"
                for n1, a1, n2, a2 in zip(aa.business_name, aa.business_address, bb.business_name, bb.business_address)]

    def train_and_score(tr_txt, tr_y, sc_txt_list, seed):
        torch.manual_seed(seed)
        rs = np.random.RandomState(seed)
        if len(tr_txt) > cfg.ce_max_train_pairs:
            keep = rs.choice(len(tr_txt), cfg.ce_max_train_pairs, replace=False)
            tr_txt = [tr_txt[i] for i in keep]
            tr_y = tr_y[keep]
        model = AutoModelForSequenceClassification.from_pretrained(cfg.ce_model, num_labels=1).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
        idx = np.arange(len(tr_txt))
        model.train()
        bs = 32
        for ep in range(cfg.ce_epochs):
            rs.shuffle(idx)
            for s in range(0, len(idx), bs):
                bi = idx[s:s + bs]
                enc = tok([tr_txt[i] for i in bi], truncation=True, max_length=160, padding=True, return_tensors="pt").to(dev)
                logits = model(**enc).logits.squeeze(-1)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.tensor(tr_y[bi], dtype=torch.float32, device=dev))
                loss.backward()
                opt.step()
                opt.zero_grad()
        model.eval()
        outs = []
        with torch.no_grad():
            for sc_txt in sc_txt_list:
                o = np.zeros(len(sc_txt), np.float32)
                for s in range(0, len(sc_txt), 128):
                    enc = tok(sc_txt[s:s + 128], truncation=True, max_length=160, padding=True, return_tensors="pt").to(dev)
                    o[s:s + 128] = model(**enc).logits.squeeze(-1).float().cpu().numpy()
                outs.append(o)
        del model
        torch.cuda.empty_cache() if dev == "cuda" else None
        return outs

    tr_txt_all = texts(train_pairs, s1, tg)
    oof = np.zeros(len(train_pairs), np.float32)
    for k, (tr, va) in enumerate(folds):
        (o,) = train_and_score([tr_txt_all[i] for i in tr], y[tr], [[tr_txt_all[i] for i in va]], cfg.seed + k)
        oof[va] = o
    test_logits = None
    if test_pairs is not None and len(test_pairs):
        all_tr = np.unique(np.concatenate([tr for tr, _ in folds]))
        (test_logits,) = train_and_score([tr_txt_all[i] for i in all_tr], y[all_tr], [texts(test_pairs, s1t, tgt)], cfg.seed)
    return oof, test_logits


# =============================================================================
# STEP 8 — structural constraints
# =============================================================================
def apply_structure(pairs: pd.DataFrame, pcol: str, hard_assign: bool) -> np.ndarray:
    """Each S2/S3 record belongs to at most one S1 entity:
       soft: p_ij <- p_ij / max(1, sum_i' p_i'j);  hard: keep only the argmax S1 per record."""
    p = pairs[pcol].values.astype(np.float64)
    g = pairs.groupby("t_idx")[pcol]
    s = g.transform("sum").values
    p = p / np.maximum(1.0, s)
    if hard_assign:
        best = g.transform("max").values
        p = np.where(pairs[pcol].values >= best - 1e-12, p, 0.0)
    return p.astype(np.float32)


# =============================================================================
# STEP 9 — expected-F0.5 decision with explicit singleton model
# =============================================================================
def poisson_binomial(ps: np.ndarray) -> np.ndarray:
    pmf = np.array([1.0])
    for q in ps:
        nxt = np.zeros(len(pmf) + 1)
        nxt[:-1] += pmf * (1 - q)
        nxt[1:] += pmf * q
        pmf = nxt
    return pmf


def expected_f_topk(p_sorted: np.ndarray, beta2: float, p_empty: float) -> np.ndarray:
    """E[F_beta(top_k)] for k=1..n. Truth-set distribution: P(Y=empty)=p_empty (from the singleton model);
       conditional on Y non-empty the candidates are independent Bernoulli(p_j)."""
    n = len(p_sorted)
    out = np.zeros(n)
    pmfs = [poisson_binomial(p_sorted[:k]) for k in range(n + 1)]
    suf = [poisson_binomial(p_sorted[k:]) for k in range(n + 1)]
    p_all_zero = float(np.prod(1 - p_sorted)) if n else 1.0
    denom_cond = max(1e-12, 1 - p_all_zero)
    for k in range(1, n + 1):
        pa, pb = pmfs[k], suf[k]
        a = np.arange(len(pa))[:, None]
        b = np.arange(len(pb))[None, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            F = (1 + beta2) * a / (beta2 * (a + b) + k)
        F[0, 0] = 0.0
        e_joint = float((pa[:, None] * pb[None, :] * F).sum())      # includes the a+b=0 term as 0
        e_cond = e_joint / denom_cond                                # E[F | Y non-empty]
        out[k - 1] = (1 - p_empty) * e_cond
    return out


def decide_entity(probs: np.ndarray, ids: np.ndarray, p_empty: float, beta2: float, T: float, s_scale: float):
    """Returns the chosen subset of ids (possibly empty). T = temperature on logits, s_scale multiplies P(empty)."""
    if len(probs) == 0:
        return []
    p = np.clip(probs, 1e-6, 1 - 1e-6)
    if T != 1.0:
        p = 1 / (1 + np.exp(-np.log(p / (1 - p)) / T))
    order = np.argsort(-p)
    ps = p[order]
    ef = expected_f_topk(ps, beta2, p_empty)
    k = int(np.argmax(ef)) + 1
    if min(1.0, p_empty * s_scale) >= ef[k - 1]:
        return []
    return list(ids[order[:k]])


ENT_COLS = ["p_max", "p_2nd", "p_sum", "n_cand", "best_wjac", "best_code", "best_num", "best_mutual", "max_sc_name", "raw_pmax"]


def entity_table(pairs: pd.DataFrame, pcol: str, s1: pd.DataFrame) -> pd.DataFrame:
    """Entity-level features for the singleton model + sorted candidates per entity."""
    d = pairs.sort_values(["s1_idx", pcol], ascending=[True, False])
    g = d.groupby("s1_idx")
    ent = pd.DataFrame({
        "p_max": g[pcol].max(), "p_2nd": g[pcol].apply(lambda x: x.iloc[1] if len(x) > 1 else 0.0),
        "p_sum": g[pcol].sum(), "n_cand": g[pcol].size().astype(float),
        "best_wjac": g["wjac_name"].max(), "best_code": g["code_agree"].max(), "best_num": g["num_agree"].max(),
        "best_mutual": g["mutual_best"].max(), "max_sc_name": g["sc_name"].max(),
        "raw_pmax": g["p3_raw"].max() if "p3_raw" in d else g[pcol].max(),
    })
    ent = ent.reindex(s1.index).fillna(0.0)
    ent["n_cand"] = ent["n_cand"].fillna(0.0)
    return ent


def singleton_oof(ent: pd.DataFrame, y_single: np.ndarray, eval_mask: np.ndarray, s1: pd.DataFrame, cfg: Config):
    """OOF P(Y = empty) on evaluated entities + final model."""
    X = ent[ENT_COLS].astype(np.float32)
    oof = np.full(len(ent), np.nan)
    idx = np.where(eval_mask)[0]
    if cfg.holdout_country:
        tr = np.where(~eval_mask)[0]
        m = lgb.train(lgb_params(cfg.seed, 31), lgb.Dataset(X.iloc[tr], y_single[tr]), 300)
        oof[idx] = m.predict(X.iloc[idx])
    else:
        gkf = GroupKFold(n_splits=cfg.n_folds)
        for k, (tr, va) in enumerate(gkf.split(idx, groups=idx)):
            m = lgb.train(lgb_params(cfg.seed + k, 31), lgb.Dataset(X.iloc[idx[tr]], y_single[idx[tr]]), 300)
            oof[idx[va]] = m.predict(X.iloc[idx[va]])
    train_idx = np.where(~eval_mask)[0] if cfg.holdout_country else idx
    final = lgb.train(lgb_params(cfg.seed, 31), lgb.Dataset(X.iloc[train_idx], y_single[train_idx]), 300)
    return oof, final


def decide_all(pairs: pd.DataFrame, pcol: str, s1: pd.DataFrame, tg: pd.DataFrame, p_empty: np.ndarray,
               beta2: float, T: float, s_scale: float, policy: str = "expected_f", tau: float = 0.5,
               only_idx: Optional[np.ndarray] = None) -> Dict[str, List[str]]:
    """policy='expected_f': per-entity expected-F0.5 maximisation with the singleton model (default);
       policy='threshold': keep every candidate with p >= tau (baseline, kept as a competing policy)."""
    out = {eid: [] for eid in s1.entity_id}
    tid = tg.entity_id.values
    s1id = s1.entity_id.values
    sub = pairs if only_idx is None else pairs[pairs.s1_idx.isin(only_idx)]
    if policy == "threshold":
        for r in sub[sub[pcol] >= tau].itertuples():
            out[s1id[r.s1_idx]].append(tid[r.t_idx])
        return out
    for si, g in sub.groupby("s1_idx"):
        out[s1id[si]] = decide_entity(g[pcol].values, tid[g.t_idx.values], float(p_empty[si]), beta2, T, s_scale)
    return out


# =============================================================================
# STEP 10 — evaluation
# =============================================================================
def f_beta_entity(pred: set, truth: set, beta2: float) -> float:
    if not truth and not pred:
        return 1.0
    if not truth or not pred:
        return 0.0
    tp = len(pred & truth)
    return (1 + beta2) * tp / (beta2 * len(truth) + len(pred))


def macro_f(pred: Dict[str, List[str]], gt: Dict[str, set], ids: List[str], beta2: float) -> float:
    return float(np.mean([f_beta_entity(set(pred.get(i, [])), gt.get(i, set()), beta2) for i in ids])) if ids else float("nan")


def tune_decision(pairs, pcol, s1, tg, p_empty, gt, eval_ids, cfg) -> dict:
    """Grid-search the decision layer on out-of-fold probabilities. Two competing policies are scored and the
       better one is kept (selection by validation, so the fancier rule can never hurt)."""
    rs = np.random.RandomState(cfg.seed)
    ids = list(eval_ids)
    if len(ids) > cfg.tune_max_entities:
        ids = list(rs.choice(ids, cfg.tune_max_entities, replace=False))
    id2idx = dict(zip(s1.entity_id, s1.index))
    only = np.array([id2idx[i] for i in ids])
    best = {"f": -1.0, "policy": "expected_f", "T": 1.0, "s_scale": 1.0, "tau": 0.5}
    best_ef = -1.0
    for T in cfg.temperature_grid:
        for s in cfg.singleton_scale_grid:
            pred = decide_all(pairs, pcol, s1, tg, p_empty, cfg.beta ** 2, T, s, "expected_f", only_idx=only)
            f = macro_f(pred, gt, ids, cfg.beta ** 2)
            best_ef = max(best_ef, f)
            if f > best["f"]:
                best = {"f": f, "policy": "expected_f", "T": T, "s_scale": s, "tau": 0.5}
    best_thr = None
    for tau in cfg.threshold_grid:
        pred = decide_all(pairs, pcol, s1, tg, p_empty, cfg.beta ** 2, 1.0, 1.0, "threshold", tau, only_idx=only)
        f = macro_f(pred, gt, ids, cfg.beta ** 2)
        if best_thr is None or f > best_thr[0]:
            best_thr = (f, tau)
        if f > best["f"] + 1e-9:
            best = {"f": f, "policy": "threshold", "T": 1.0, "s_scale": 1.0, "tau": tau}
    LOG.info(f"decision tuning on {len(ids)} entities: expected-F policy best={best_ef:.4f}, "
             f"threshold policy best={best_thr[0]:.4f} (tau={best_thr[1]}); chosen={best}")
    return best


def blocking_stats(pairs: pd.DataFrame, s1: pd.DataFrame, tg: pd.DataFrame, gt: Dict[str, set], name: str) -> dict:
    n_pairs = len(pairs)
    n_s1 = len(s1)
    stats = {"stage": name, "pairs": int(n_pairs), "avg_candidates_per_s1": n_pairs / max(1, n_s1),
             "reduction_ratio": 1 - n_pairs / max(1, n_s1 * len(tg))}
    if gt:
        y = label_pairs(pairs, s1, tg, gt)
        n_true = sum(len(v) for v in gt.values() if v)
        stats["pairs_completeness(recall_ceiling)"] = float(y.sum()) / max(1, n_true)
        stats["pair_quality(precision)"] = float(y.mean()) if n_pairs else 0.0
    LOG.info(f"blocking[{name}]: " + ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in stats.items()))
    return stats


# =============================================================================
# STEP 11 — writing outputs, validator
# =============================================================================
def write_id_lists(path: str, s1: pd.DataFrame, mapping: Dict[str, List[str]], col: str):
    rows = []
    for eid in s1.entity_id:
        ids = mapping.get(eid, [])
        ids = sorted(set(i for i in ids if i.startswith(("S2-", "S3-"))))
        rows.append((eid, ",".join(ids)))
    pd.DataFrame(rows, columns=["source1_entity_id", col]).to_csv(path, sep="\t", index=False, encoding="utf-8")
    LOG.info(f"wrote {path} ({len(rows)} rows, {sum(1 for r in rows if r[1])} non-empty)")


def run_validator(cfg: Config):
    cands = [cfg.validator, "validate_submission.py", os.path.join(os.path.dirname(__file__), "validate_submission.py")]
    v = next((c for c in cands if os.path.exists(c)), None)
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
# Orchestration: train (with out-of-fold evaluation) then test inference
# =============================================================================
def prepare_split(cfg: Config, split: str, abbrev: Optional[Dict[str, str]] = None):
    data = load_split(cfg, split)
    gt = gt_to_dict(data["gt"])
    tg_raw = pd.concat([data["s2"], data["s3"]], ignore_index=True)
    if abbrev is None:
        abbrev = dict(STATIC_ABBREV)
        abbrev.update(mine_abbreviations(data["s1"], tg_raw, gt))
    with Timer(f"STEP 2 normalisation [{split}]"):
        s1 = normalise_records(data["s1"], abbrev).reset_index(drop=True)
        tg = normalise_records(tg_raw, abbrev).reset_index(drop=True)
    with Timer(f"STEP 3 per-country IDF [{split}]"):
        idf = IDF().fit([s1, tg])
        LOG.info(f"countries seen: {sorted(idf.tables)}")
    return s1, tg, gt, idf, abbrev


def eda_decisions(cfg: Config, s1: pd.DataFrame, tg: pd.DataFrame, gt: Dict[str, set]):
    """Data-driven switches: country blocking and one-to-one assignment."""
    c1 = dict(zip(s1.entity_id, s1.country_n))
    c2 = dict(zip(tg.entity_id, tg.country_n))
    same = tot = 0
    owner = Counter()
    for a, ms in gt.items():
        for m in ms:
            tot += 1
            same += int(c1.get(a) == c2.get(m))
            owner[m] += 1
    agree = same / tot if tot else 1.0
    uniq = np.mean([v == 1 for v in owner.values()]) if owner else 1.0
    n_single = sum(1 for v in gt.values() if not v)
    LOG.info(f"EDA: GT pairs={tot}, country agreement={agree:.4f}, S2/S3 uniqueness={uniq:.4f}, "
             f"singletons={n_single}/{len(gt)} ({n_single / max(1, len(gt)):.3f}), "
             f"match-count dist={dict(Counter(len(v) for v in gt.values()))}")
    if cfg.country_block is None:
        cfg.country_block = agree >= 0.99
    if cfg.hard_assign is None:
        cfg.hard_assign = uniq >= 0.995
    LOG.info(f"decisions: country_block={cfg.country_block}, hard_assign={cfg.hard_assign}")


def run(cfg: Config):
    setup_logging(cfg.out_dir)
    LOG.info(f"environment={detect_environment()} gpu={has_gpu()} sparse_dot_topn={HAS_SDT}")
    LOG.info("config: " + json.dumps(asdict(cfg), default=str))
    np.random.seed(cfg.seed)
    metrics = {}

    # ---------------------------------------------------------------- TRAIN
    with Timer("STEP 1 load train"):
        s1, tg, gt, idf, abbrev = prepare_split(cfg, "train")
        eda_decisions(cfg, s1, tg, gt)

    with Timer("STEP 4 retrieval [train]"):
        pairs = retrieve(cfg, s1, tg, idf, cfg.country_block)
        metrics["blocking_raw"] = blocking_stats(pairs, s1, tg, gt, "raw_retrieval")

    with Timer("STEP 5 cheap features + Stage-2 pruning [train]"):
        pairs = cheap_features(pairs, s1, tg, idf)
        y = label_pairs(pairs, s1, tg, gt)
        folds = make_folds(pairs, s1, cfg)
        eval_pair_mask = np.zeros(len(pairs), bool)
        for _, va in folds:
            eval_pair_mask[va] = True
        oof1, m_stage2 = oof_lgb(pairs, CHEAP_COLS, y, folds, cfg.lgb_rounds_stage2, cfg.seed, leaves=31)
        iso1 = fit_isotonic(oof1[eval_pair_mask], y[eval_pair_mask])
        pairs["p1"] = np.where(eval_pair_mask, iso1.predict(np.nan_to_num(oof1)), iso1.predict(m_stage2.predict(pairs[CHEAP_COLS].astype(np.float32)))).astype(np.float32)
        eps = choose_eps(pairs.p1.values[eval_pair_mask], y[eval_pair_mask], cfg.prune_target_recall, cfg.prune_eps_floor)
        LOG.info(f"Stage-2 pruning threshold eps={eps:.4f}, kmax={cfg.kmax}")
        pairs["y"] = y
        pairs["is_eval"] = eval_pair_mask
        pruned = prune(pairs, "p1", eps, cfg.kmax)
        metrics["blocking_pruned"] = blocking_stats(pruned, s1, tg, gt, "after_stage2_pruning")
        metrics["eps"] = eps
        expected_lost = float(pairs.loc[~pairs.index.isin(pruned.index) & pairs["is_eval"], "p1"].sum())
        LOG.info(f"expected true pairs lost by pruning (sum of pruned p1) = {expected_lost:.1f}")

    with Timer("STEP 6 full features [train]"):
        pruned = full_features(pruned.reset_index(drop=True), s1, tg, idf)
        pruned = competition_features(pruned, "p1")
        y3 = pruned["y"].values
        folds3 = make_folds(pruned, s1, cfg)
        eval3 = np.zeros(len(pruned), bool)
        for _, va in folds3:
            eval3[va] = True
        cols3 = list(FULL_COLS)
        ce_test_holder = {}
        if cfg.use_cross_encoder:
            LOG.info("training optional cross-encoder (this needs a GPU to be practical)")
            pruned["ce_logit"] = 0.0
            ce_test_holder["needed"] = True
            cols3.append("ce_logit")

    # ---------------------------------------------------------------- TEST retrieval (needed now if cross-encoder is on)
    test = None
    if cfg.run_test:
        with Timer("STEP 1/2/3/4/5 test preparation + retrieval + pruning"):
            s1t, tgt, gtt, idft, _ = prepare_split(cfg, "test", abbrev)
            pt = retrieve(cfg, s1t, tgt, idft, cfg.country_block)
            metrics["test_blocking_raw"] = blocking_stats(pt, s1t, tgt, {}, "test_raw_retrieval")
            pt = cheap_features(pt, s1t, tgt, idft)
            pt["p1"] = iso1.predict(m_stage2.predict(pt[CHEAP_COLS].astype(np.float32))).astype(np.float32)
            pt = prune(pt, "p1", eps, cfg.kmax)
            metrics["test_blocking_pruned"] = blocking_stats(pt, s1t, tgt, {}, "test_after_stage2_pruning")
            pt = full_features(pt.reset_index(drop=True), s1t, tgt, idft)
            pt = competition_features(pt, "p1")
            test = (s1t, tgt, gtt, idft, pt)

    if cfg.use_cross_encoder:
        with Timer("STEP 6b cross-encoder"):
            oof_ce, ce_test = cross_encoder_scores(cfg, pruned, y3, folds3, s1, tg,
                                                   test[4] if test else None, test[0] if test else None, test[1] if test else None)
            pruned["ce_logit"] = oof_ce
            if test is not None:
                test[4]["ce_logit"] = ce_test

    with Timer("STEP 7 Stage-3 matcher [train, out-of-fold]"):
        oof3, m_stage3 = oof_lgb(pruned, cols3, y3, folds3, cfg.lgb_rounds_stage3, cfg.seed)
        iso3 = fit_isotonic(oof3[eval3], y3[eval3])
        pruned["p3_raw"] = np.nan_to_num(oof3)
        pruned["p3"] = iso3.predict(pruned["p3_raw"].values).astype(np.float32)
        imp = sorted(zip(cols3, m_stage3.feature_importance("gain")), key=lambda x: -x[1])[:15]
        LOG.info("top features (gain): " + ", ".join(f"{c}={g:.0f}" for c, g in imp))

    with Timer("STEP 8 structural constraints [train]"):
        pruned["p"] = apply_structure(pruned, "p3", cfg.hard_assign)

    with Timer("STEP 9 singleton model + decision tuning [train]"):
        ev = pruned[eval3].reset_index(drop=True)
        eval_entity_mask = np.zeros(len(s1), bool)
        if cfg.holdout_country:
            eval_entity_mask[:] = s1.country_n.values == norm_text(cfg.holdout_country)
        else:
            eval_entity_mask[:] = True
        ent = entity_table(ev, "p", s1)
        y_single = np.array([1 if not gt.get(e, set()) else 0 for e in s1.entity_id], np.int8)
        p_empty_oof, m_single = singleton_oof(ent, y_single, eval_entity_mask, s1, cfg)
        p_empty_oof = np.nan_to_num(p_empty_oof, nan=0.5)
        eval_ids = list(s1.entity_id[eval_entity_mask])
        dec = tune_decision(ev, "p", s1, tg, p_empty_oof, gt, eval_ids, cfg)
        best_f = dec["f"]
        metrics["validation_macro_f05_tuning_subset"] = best_f
        metrics["decision"] = dec

    with Timer("STEP 10 evaluation [train, out-of-fold]"):
        pred = decide_all(ev, "p", s1, tg, p_empty_oof, cfg.beta ** 2, dec["T"], dec["s_scale"], dec["policy"], dec["tau"])
        metrics["validation_macro_f05"] = macro_f(pred, gt, eval_ids, cfg.beta ** 2)
        best_f = metrics["validation_macro_f05"]
        per_country = {}
        for c in sorted(s1.country_n.unique()):
            ids = list(s1.entity_id[(s1.country_n == c) & eval_entity_mask])
            if ids:
                per_country[c] = macro_f(pred, gt, ids, cfg.beta ** 2)
        metrics["validation_per_country_f05"] = per_country
        # simple baseline for reference: global threshold 0.5 on p
        base = {e: [] for e in s1.entity_id}
        for r in ev[ev.p >= 0.5].itertuples():
            base[s1.entity_id.values[r.s1_idx]].append(tg.entity_id.values[r.t_idx])
        metrics["baseline_global_threshold_0.5_f05"] = macro_f(base, gt, eval_ids, cfg.beta ** 2)
        LOG.info(f"VALIDATION macro-F0.5 = {best_f:.4f} | per-country = {per_country} | "
                 f"global-threshold baseline = {metrics['baseline_global_threshold_0.5_f05']:.4f}")
        # error analysis samples
        fp, fn = [], []
        s1n = dict(zip(s1.entity_id, s1.business_name + " | " + s1.business_address))
        tgn = dict(zip(tg.entity_id, tg.business_name + " | " + tg.business_address))
        for e in eval_ids:
            P, Tt = set(pred.get(e, [])), gt.get(e, set())
            for x in P - Tt:
                fp.append((s1n[e], tgn.get(x, "")))
            for x in Tt - P:
                fn.append((s1n[e], tgn.get(x, "")))
        metrics["n_false_positive_pairs"] = len(fp)
        metrics["n_false_negative_pairs"] = len(fn)
        with open(os.path.join(cfg.out_dir, "error_samples.txt"), "w", encoding="utf-8") as f:
            f.write("FALSE POSITIVES (wrong merges)\n")
            f.writelines(f"  {a}  <->  {b}\n" for a, b in fp[:200])
            f.write("\nFALSE NEGATIVES (missed matches)\n")
            f.writelines(f"  {a}  <->  {b}\n" for a, b in fn[:200])
        ev[["s1_idx", "t_idx", "y", "p1", "p3", "p"]].to_parquet(os.path.join(cfg.work_dir, "train_oof_pairs.parquet"))

    # ---------------------------------------------------------------- TEST inference
    if test is not None:
        s1t, tgt, gtt, idft, pt = test
        with Timer("STEP 7/8/9 test inference"):
            pt["p3_raw"] = m_stage3.predict(pt[cols3].astype(np.float32))
            pt["p3"] = iso3.predict(pt["p3_raw"].values).astype(np.float32)
            pt["p"] = apply_structure(pt, "p3", cfg.hard_assign)
            entt = entity_table(pt, "p", s1t)
            p_empty_t = m_single.predict(entt[ENT_COLS].astype(np.float32))
            predt = decide_all(pt, "p", s1t, tgt, p_empty_t, cfg.beta ** 2, dec["T"], dec["s_scale"], dec["policy"], dec["tau"])
            candt = defaultdict(list)
            for r in pt[["s1_idx", "t_idx"]].itertuples():
                candt[s1t.entity_id.values[r.s1_idx]].append(tgt.entity_id.values[r.t_idx])
        with Timer("STEP 11 write outputs"):
            write_id_lists(os.path.join(cfg.out_dir, "candidate_pairs.tsv"), s1t, candt, "candidate_entity_ids")
            write_id_lists(os.path.join(cfg.out_dir, "matching_results.tsv"), s1t, predt, "matched_entity_ids")
            pt[["s1_idx", "t_idx", "p1", "p3", "p"]].to_parquet(os.path.join(cfg.work_dir, "test_scored_pairs.parquet"))
            metrics["test_predicted_nonempty"] = int(sum(1 for v in predt.values() if v))
            metrics["test_avg_candidates_per_s1"] = len(pt) / max(1, len(s1t))
            if gtt:  # only for synthetic smoke tests with a hidden test ground truth
                metrics["hidden_test_macro_f05"] = macro_f(predt, gtt, list(s1t.entity_id), cfg.beta ** 2)
                metrics["hidden_test_per_country"] = {c: macro_f(predt, gtt, list(s1t.entity_id[s1t.country_n == c]), cfg.beta ** 2)
                                                      for c in sorted(s1t.country_n.unique())}
                LOG.info(f"HIDDEN TEST macro-F0.5 = {metrics['hidden_test_macro_f05']:.4f} per-country={metrics['hidden_test_per_country']}")
            run_validator(cfg)

    with open(os.path.join(cfg.out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, default=float)
    LOG.info("metrics: " + json.dumps(metrics, indent=1, default=float))
    LOG.info("ALL DONE")


# =============================================================================
# STEP 12 — CLI
# =============================================================================
def parse_args(argv=None) -> Config:
    ap = argparse.ArgumentParser(description="Business Entity Resolution pipeline")
    ap.add_argument("--data-dir", default="dataset")
    ap.add_argument("--out-dir", default="output")
    ap.add_argument("--work-dir", default="work")
    ap.add_argument("--sample", type=float, default=1.0, help="fraction of S1 entities (dev runs)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--holdout-country", default=None, help="leave-one-country-out validation, e.g. India")
    ap.add_argument("--no-test", action="store_true", help="skip test inference (validation only)")
    ap.add_argument("--use-dense", action="store_true", help="add multilingual dense retrieval view (needs GPU)")
    ap.add_argument("--use-cross-encoder", action="store_true", help="add fine-tuned cross-encoder feature (needs GPU)")
    ap.add_argument("--k-sparse-name", type=int, default=15)
    ap.add_argument("--k-sparse-full", type=int, default=10)
    ap.add_argument("--kmax", type=int, default=10)
    ap.add_argument("--prune-target-recall", type=float, default=0.995)
    ap.add_argument("--country-block", choices=["auto", "on", "off"], default="auto")
    ap.add_argument("--hard-assign", choices=["auto", "on", "off"], default="auto")
    ap.add_argument("--validator", default="utils/validate_submission.py")
    a = ap.parse_args(argv)
    tri = {"auto": None, "on": True, "off": False}
    return Config(data_dir=a.data_dir, out_dir=a.out_dir, work_dir=a.work_dir, sample=a.sample, seed=a.seed,
                  n_folds=a.n_folds, holdout_country=a.holdout_country, run_test=not a.no_test,
                  use_dense=a.use_dense, use_cross_encoder=a.use_cross_encoder, k_sparse_name=a.k_sparse_name,
                  k_sparse_full=a.k_sparse_full, kmax=a.kmax, prune_target_recall=a.prune_target_recall,
                  country_block=tri[a.country_block], hard_assign=tri[a.hard_assign], validator=a.validator)


if __name__ == "__main__":
    run(parse_args())
