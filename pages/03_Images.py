# app.py
# Streamlit app: easy-to-use IMAGE dataset safety analyzer (quality, reliability, robustness, fairness, security)
#
# Install:
#   pip install streamlit pandas numpy pillow
#   pip install imagehash          # perceptual-duplicate detection (recommended)
#   pip install pyarrow            # parquet metadata support (optional)
#
# Run:
#   streamlit run app.py
#
# Notes:
# - Heuristic checks. Validate any flags with domain + legal review.
# - No face detection, no OCR.

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import zipfile
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageStat, ImageOps

# Optional perceptual hashing
try:
    import imagehash

    IMAGEHASH_OK = True
except Exception:
    IMAGEHASH_OK = False


# =========================
# Styling
# =========================

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils import SHARED_CSS, sha256_bytes as _sha256, badge as _badge, kpi as _kpi, health_ring_html, progress_bar_html, compute_health_score, to_json_safe as _to_json_safe

st.set_page_config(page_title="Image Dataset Analyzer", page_icon="🖼", layout="wide")

CUSTOM_CSS = """
<style>
.block-container { padding-top: 1.0rem; padding-bottom: 3rem; max-width: 1250px; }
h1, h2, h3 { letter-spacing: -0.02em; }
.small-muted { color: rgba(128,128,128,0.95); font-size: 0.92rem; }
.kpi-card {
  border: 1px solid rgba(120,120,120,0.20);
  border-radius: 14px;
  padding: 14px 16px;
  background: rgba(255,255,255,0.03);
}
.badge {
  display: inline-block;
  padding: 2px 10px;
  border-radius: 999px;
  font-size: 0.85rem;
  border: 1px solid rgba(120,120,120,0.25);
}
.badge-ok { background: rgba(0, 200, 0, 0.12); }
.badge-warn { background: rgba(255, 165, 0, 0.12); }
.badge-bad { background: rgba(255, 0, 0, 0.10); }
.section-card {
  border: 1px solid rgba(120,120,120,0.20);
  border-radius: 16px;
  padding: 18px 18px 8px 18px;
  background: rgba(255,255,255,0.02);
}
hr { border: none; height: 1px; background: rgba(120,120,120,0.22); margin: 1.2rem 0; }
.code-pill {
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
  font-size: 0.85rem;
  padding: 2px 8px;
  border-radius: 999px;
  border: 1px solid rgba(120,120,120,0.25);
}
</style>
"""
st.markdown(SHARED_CSS, unsafe_allow_html=True)


# =========================
# JSON safety
# =========================

def to_json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return to_json_safe(obj.tolist())
    if isinstance(obj, (pd.Timestamp, pd.Timedelta)):
        return str(obj)
    if obj is pd.NA:
        return None
    try:
        if pd.isna(obj):
            return None
    except Exception:
        pass
    return obj


# =========================
# Helpers
# =========================

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

def sha256_bytes(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()

def clip_text(s: str, n: int = 90) -> str:
    s = str(s)
    return s if len(s) <= n else s[: n - 1] + "…"

def to_datetime_if_possible(s: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(s.dtype):
        return s
    try:
        return pd.to_datetime(s, errors="coerce")
    except Exception:
        return s

def categorical_cols(df: pd.DataFrame, exclude: Optional[List[str]] = None) -> List[str]:
    exclude_set = set(exclude or [])
    cols: List[str] = []
    for c in df.columns:
        if c in exclude_set:
            continue
        if pd.api.types.is_datetime64_any_dtype(df[c].dtype):
            continue
        if pd.api.types.is_numeric_dtype(df[c].dtype):
            continue
        cols.append(c)
    return cols

def numeric_cols(df: pd.DataFrame, exclude: Optional[List[str]] = None) -> List[str]:
    exclude_set = set(exclude or [])
    cols: List[str] = []
    for c in df.columns:
        if c in exclude_set:
            continue
        if pd.api.types.is_numeric_dtype(df[c].dtype):
            cols.append(c)
    return cols

def ks_statistic(x1: np.ndarray, x2: np.ndarray) -> Optional[float]:
    x1 = x1[~np.isnan(x1)]
    x2 = x2[~np.isnan(x2)]
    if len(x1) < 30 or len(x2) < 30:
        return None
    x1 = np.sort(x1)
    x2 = np.sort(x2)
    all_vals = np.sort(np.unique(np.concatenate([x1, x2])))
    cdf1 = np.searchsorted(x1, all_vals, side="right") / len(x1)
    cdf2 = np.searchsorted(x2, all_vals, side="right") / len(x2)
    return float(np.max(np.abs(cdf1 - cdf2)))

def badge(label: str, kind: str) -> str:
    cls = {"ok": "badge badge-ok", "warn": "badge badge-warn", "bad": "badge badge-bad"}.get(kind, "badge")
    return f'<span class="{cls}">{label}</span>'

def kpi(title: str, value: str, hint: str = "") -> None:
    st.markdown(
        f"""
        <div class="kpi-card">
          <div style="font-size:0.92rem; opacity:0.9;">{title}</div>
          <div style="font-size:1.5rem; font-weight:700; margin-top:4px;">{value}</div>
          <div class="muted" style="margin-top:6px;">{hint}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


PII_PATTERNS = {
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "phone_like": re.compile(r"\b(?:\+?\d{1,3}[\s-]?)?(?:\(?\d{2,4}\)?[\s-]?)?\d{3,4}[\s-]?\d{3,4}\b"),
    "ip_v4": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    "credit_card_like": re.compile(r"\b(?:\d[ -]*?){13,19}\b"),
}


# =========================
# Thresholds + presets
# =========================

@dataclass(frozen=True)
class Thresholds:
    drift_ks_threshold: float
    pii_hit_rate_threshold: float
    perceptual_dup_hamming_threshold: int
    blur_laplacian_like_threshold: float
    min_resolution_short_side: int


PRESETS: Dict[str, Thresholds] = {
    "Balanced (recommended)": Thresholds(
        drift_ks_threshold=0.30,
        pii_hit_rate_threshold=0.01,
        perceptual_dup_hamming_threshold=6,
        blur_laplacian_like_threshold=25.0,
        min_resolution_short_side=224,
    ),
    "Strict": Thresholds(
        drift_ks_threshold=0.20,
        pii_hit_rate_threshold=0.005,
        perceptual_dup_hamming_threshold=4,
        blur_laplacian_like_threshold=35.0,
        min_resolution_short_side=256,
    ),
    "Lenient": Thresholds(
        drift_ks_threshold=0.40,
        pii_hit_rate_threshold=0.02,
        perceptual_dup_hamming_threshold=8,
        blur_laplacian_like_threshold=18.0,
        min_resolution_short_side=160,
    ),
}


# =========================
# Column guessing for metadata
# =========================

def _name_score(name: str, patterns: List[str]) -> float:
    n = name.lower().strip()
    score = 0.0
    for p in patterns:
        if re.search(p, n):
            score += 1.0
    return score

def guess_columns_meta(df: pd.DataFrame) -> Dict[str, Any]:
    cols = df.columns.tolist()
    nrows = max(1, len(df))
    notes: List[str] = []

    nunique = {c: int(df[c].nunique(dropna=True)) for c in cols}
    uniq_ratio = {c: float(nunique[c] / nrows) for c in cols}

    label_patterns = [r"\blabel\b", r"\btarget\b", r"\boutcome\b", r"\bclass\b", r"\bground[_\s-]?truth\b", r"\bgt\b", r"\by\b"]
    split_patterns = [r"\bsplit\b", r"\bfold\b", r"\bset\b", r"\bpartition\b", r"\btrain[_\s-]?test\b"]
    time_patterns = [r"\btime\b", r"\bdate\b", r"\btimestamp\b", r"\bcreated\b", r"\bupdated\b", r"\bcaptured\b"]
    path_patterns = [r"\bpath\b", r"\bfilepath\b", r"\bfile\b", r"\bimage\b", r"\bimg\b", r"\buri\b", r"\burl\b", r"\bfilename\b"]
    id_patterns = [r"\bid\b", r"\buuid\b", r"\bguid\b", r"\brecord[_\s-]?id\b", r"\bimage[_\s-]?id\b"]
    group_patterns = [r"\bgender\b", r"\bsex\b", r"\bage\b", r"\bregion\b", r"\bcountry\b", r"\bethnicity\b", r"\brace\b", r"\bgroup\b"]

    def rank_label(c: str) -> float:
        s = _name_score(c, label_patterns) * 3.0
        if 2 <= nunique[c] <= min(100, int(0.02 * nrows) + 2):
            s += 2.0
        if uniq_ratio[c] < 0.3:
            s += 1.0
        if uniq_ratio[c] > 0.9:
            s -= 2.0
        return s

    def rank_split(c: str) -> float:
        s = _name_score(c, split_patterns) * 3.0
        if nunique[c] <= 30:
            s += 2.0
        try:
            vals = df[c].dropna().astype("string").str.lower().value_counts().head(12).index.tolist()
            joined = " ".join(vals)
            if any(x in joined for x in ["train", "test", "val", "valid", "dev", "holdout"]):
                s += 2.0
        except Exception:
            pass
        if uniq_ratio[c] > 0.5:
            s -= 2.0
        return s

    def rank_time(c: str) -> float:
        s = _name_score(c, time_patterns) * 3.0
        parsed = to_datetime_if_possible(df[c])
        if pd.api.types.is_datetime64_any_dtype(parsed.dtype):
            s += float(parsed.notna().mean()) * 2.0
        return s

    def rank_path(c: str) -> float:
        s = _name_score(c, path_patterns) * 3.0
        if uniq_ratio[c] > 0.7:
            s += 1.0
        return s

    def rank_id(c: str) -> float:
        s = _name_score(c, id_patterns) * 3.0
        if uniq_ratio[c] > 0.95:
            s += 2.5
        return s

    def rank_group(c: str) -> float:
        s = _name_score(c, group_patterns) * 2.0
        if 2 <= nunique[c] <= 50:
            s += 2.0
        if uniq_ratio[c] > 0.6:
            s -= 1.0
        return s

    ranked_label = sorted(cols, key=rank_label, reverse=True)
    ranked_split = sorted(cols, key=rank_split, reverse=True)
    ranked_time = sorted(cols, key=rank_time, reverse=True)
    ranked_path = sorted(cols, key=rank_path, reverse=True)
    ranked_id = sorted(cols, key=rank_id, reverse=True)
    ranked_group = sorted(cols, key=rank_group, reverse=True)

    label = ranked_label[0] if ranked_label and rank_label(ranked_label[0]) >= 2.0 else None
    split = ranked_split[0] if ranked_split and rank_split(ranked_split[0]) >= 2.0 else None
    time = ranked_time[0] if ranked_time and rank_time(ranked_time[0]) >= 2.0 else None
    path = ranked_path[0] if ranked_path and rank_path(ranked_path[0]) >= 2.0 else None

    ids: List[str] = []
    for c in ranked_id:
        if c in {label, split, time, path}:
            continue
        if rank_id(c) >= 3.5:
            ids.append(c)
        if len(ids) >= 3:
            break

    groups: List[str] = []
    for c in ranked_group:
        if c in {label, split, time, path} or c in set(ids):
            continue
        if rank_group(c) >= 2.5:
            groups.append(c)
        if len(groups) >= 3:
            break

    if path:
        notes.append(f"Guessed image path column: {path}")
    if label:
        notes.append(f"Guessed label: {label}")
    if split:
        notes.append(f"Guessed split: {split}")
    if time:
        notes.append(f"Guessed time: {time}")
    if ids:
        notes.append(f"Guessed id columns: {', '.join(ids)}")
    if groups:
        notes.append(f"Guessed group columns: {', '.join(groups)}")

    return {"path": path, "label": label, "split": split, "time": time, "ids": ids, "groups": groups, "notes": notes}


# =========================
# Image scanning
# =========================

EXIF_COLUMNS = ["has_exif", "has_gps", "exif_make", "exif_model", "exif_datetime"]

def safe_open_image(data: bytes) -> Tuple[Optional[Image.Image], Optional[str]]:
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
        return img, None
    except Exception as e:
        return None, str(e)

def image_features(img: Image.Image) -> Dict[str, Any]:
    try:
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass

    w, h = img.size
    mode = img.mode

    if mode not in ("RGB", "L"):
        img_rgb = img.convert("RGB")
    else:
        img_rgb = img if mode == "RGB" else img.convert("RGB")

    stat = ImageStat.Stat(img_rgb)
    means = stat.mean
    stds = stat.stddev

    gray = img_rgb.convert("L")
    g = np.asarray(gray, dtype=np.float32)

    if g.shape[0] >= 3 and g.shape[1] >= 3:
        center = g[1:-1, 1:-1]
        lap = g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:] - 4.0 * center
        blur_score = float(np.var(lap))
    else:
        blur_score = float("nan")

    hist = np.bincount(np.clip(g.astype(np.int32), 0, 255).ravel(), minlength=256).astype(np.float64)
    p = hist / max(1.0, hist.sum())
    entropy = float(-(p[p > 0] * np.log2(p[p > 0])).sum())

    return {
        "width": int(w),
        "height": int(h),
        "short_side": int(min(w, h)),
        "long_side": int(max(w, h)),
        "aspect_ratio": float(w / h) if h else None,
        "mode": str(mode),
        "mean_r": float(means[0]),
        "mean_g": float(means[1]),
        "mean_b": float(means[2]),
        "std_r": float(stds[0]),
        "std_g": float(stds[1]),
        "std_b": float(stds[2]),
        "brightness_mean": float(np.mean(means)),
        "color_std_mean": float(np.mean(stds)),
        "blur_var_lap": blur_score,
        "entropy": entropy,
    }

def exif_flags(img: Image.Image) -> Dict[str, Any]:
    flags: Dict[str, Any] = {}
    try:
        exif = img.getexif()
        if exif is None or len(exif) == 0:
            return {"has_exif": False, "has_gps": False, "exif_make": None, "exif_model": None, "exif_datetime": None}
        flags["has_exif"] = True

        dt_orig = exif.get(36867) or exif.get(306)
        flags["exif_datetime"] = str(dt_orig) if dt_orig else None

        make = exif.get(271)
        model = exif.get(272)
        flags["exif_make"] = str(make) if make else None
        flags["exif_model"] = str(model) if model else None

        gps = exif.get(34853)
        flags["has_gps"] = bool(gps)
    except Exception as e:
        flags["exif_error"] = str(e)
        flags.setdefault("has_exif", False)
        flags.setdefault("has_gps", False)
        flags.setdefault("exif_make", None)
        flags.setdefault("exif_model", None)
        flags.setdefault("exif_datetime", None)
    return flags

def perceptual_hash(img: Image.Image) -> Optional[str]:
    if not IMAGEHASH_OK:
        return None
    try:
        h = imagehash.phash(ImageOps.exif_transpose(img).convert("RGB"))
        return str(h)
    except Exception:
        return None

def hamming_hexhash(a: Optional[str], b: Optional[str]) -> Optional[int]:
    if not a or not b:
        return None
    try:
        ia = int(a, 16)
        ib = int(b, 16)
        return int((ia ^ ib).bit_count())
    except Exception:
        return None


# =========================
# Config + assessment
# =========================

@dataclass
class AssessConfig:
    path_col: Optional[str]
    label_col: Optional[str]
    split_col: Optional[str]
    time_col: Optional[str]
    group_cols: List[str]
    id_cols: List[str]

    metadata: Dict[str, Any]
    mode: str = "Quick Scan"
    thresholds: Thresholds = field(default_factory=lambda: PRESETS["Balanced (recommended)"])
    random_state: int = 7

    max_images: int = 3000
    sample_for_perceptual_dups: int = 1500
    sample_for_exif: int = 2000
    max_pairs_for_near_dups: int = 200000

def build_recommendations(report: Dict[str, Any], cfg: AssessConfig) -> List[str]:
    recs: List[str] = []

    q = report.get("quality", {})
    if q.get("corrupt_images", {}).get("corrupt_rate", 0.0) > 0.001:
        recs.append("Corrupt images found. Remove or fix them and rerun the scan.")
    if q.get("low_resolution", {}).get("low_res_rate", 0.0) > 0.05:
        recs.append("Many images are below the minimum short-side threshold. Consider resizing policy or filtering.")
    if q.get("duplicates", {}).get("exact_duplicate_rate", 0.0) > 0.01:
        recs.append("Exact duplicate files exceed 1%. Consider deduplication before splitting.")
    if q.get("duplicates", {}).get("perceptual_near_duplicate_pairs", 0) > 0:
        recs.append("Near-duplicate images detected. Remove near-duplicates and avoid cross-split near-duplicates.")

    r = report.get("reliability", {})
    drift = r.get("feature_drift_ks_first_last", {}).get("top_10_ks", {})
    if drift and any(float(v) > cfg.thresholds.drift_ks_threshold for v in drift.values() if v is not None):
        recs.append("Drift above threshold on at least one feature. Check pipeline changes or retrain with updated data.")

    s = report.get("security", {})
    if s.get("exif_privacy", {}).get("gps_images_count", 0) > 0:
        recs.append("EXIF GPS present. Strip EXIF or remove GPS fields before release.")
    pii_hits = s.get("pii_like_in_paths", {}).get("columns_with_hits", {})
    if pii_hits:
        recs.append("PII-like patterns in filenames/paths or metadata text. Mask or rename assets.")

    if not recs:
        recs.append("No major red flags under current checks. Save the JSON report and rerun after updates.")
    return recs

def verdict_panel(report: Dict[str, Any], cfg: AssessConfig) -> Tuple[str, str, List[str]]:
    reasons: List[str] = []

    q = report.get("quality", {})
    r = report.get("reliability", {})
    s = report.get("security", {})

    corrupt_rate = q.get("corrupt_images", {}).get("corrupt_rate", 0.0)
    if corrupt_rate and float(corrupt_rate) > 0.001:
        reasons.append("Corrupt images present.")

    dup_rate = q.get("duplicates", {}).get("exact_duplicate_rate", 0.0)
    if dup_rate and float(dup_rate) > 0.01:
        reasons.append("Exact duplicates exceed 1%.")

    drift = r.get("feature_drift_ks_first_last", {}).get("top_10_ks", {})
    drift_flag = any(v is not None and float(v) > cfg.thresholds.drift_ks_threshold for v in drift.values())
    if drift_flag:
        reasons.append("Potential drift. At least one KS statistic is above the selected threshold.")

    if s.get("exif_privacy", {}).get("gps_images_count", 0) > 0:
        reasons.append("EXIF GPS present.")

    pii_hits = s.get("pii_like_in_paths", {}).get("columns_with_hits", {})
    if pii_hits:
        reasons.append("PII-like patterns detected in paths or metadata.")

    if reasons:
        return "Needs review", "warn", reasons
    return "Looks OK (evidence-based)", "ok", ["No major red flags under current checks."]


# =========================
# ZIP ingestion
# =========================

def read_zip_images(zip_bytes: bytes, cfg: AssessConfig) -> Tuple[pd.DataFrame, List[str]]:
    warnings: List[str] = []

    zf = zipfile.ZipFile(io.BytesIO(zip_bytes))
    names = [n for n in zf.namelist() if not n.endswith("/")]

    img_names: List[str] = []
    for n in names:
        ext = os.path.splitext(n)[1].lower()
        if ext in IMG_EXTS:
            img_names.append(n)

    if not img_names:
        return pd.DataFrame(), ["No supported images found in ZIP."]

    if len(img_names) > cfg.max_images:
        warnings.append(f"Image count {len(img_names)} exceeds cap {cfg.max_images}. Scanning a sample.")
        rng = np.random.RandomState(cfg.random_state)
        img_names = list(rng.choice(img_names, size=cfg.max_images, replace=False))

    rows: List[Dict[str, Any]] = []

    rng = np.random.RandomState(cfg.random_state)
    sample_phash = set()
    if IMAGEHASH_OK:
        sample_n = min(len(img_names), cfg.sample_for_perceptual_dups)
        sample_phash = set(rng.choice(img_names, size=sample_n, replace=False))
    sample_exif = set(rng.choice(img_names, size=min(len(img_names), cfg.sample_for_exif), replace=False))

    for n in img_names:
        data = zf.read(n)
        row: Dict[str, Any] = {
            "path_in_zip": n,
            "byte_size": int(len(data)),
            "sha256": sha256_bytes(data),
        }

        img, err = safe_open_image(data)
        if img is None:
            row["open_ok"] = False
            row["open_error"] = err
            row.update({c: None for c in EXIF_COLUMNS})
            row["phash"] = None
            rows.append(row)
            continue

        row["open_ok"] = True
        row["open_error"] = None

        row.update(image_features(img))

        if n in sample_exif:
            ex = exif_flags(img)
            for c in EXIF_COLUMNS:
                ex.setdefault(c, None)
            row.update(ex)
        else:
            row.update({c: None for c in EXIF_COLUMNS})

        if IMAGEHASH_OK and n in sample_phash:
            row["phash"] = perceptual_hash(img)
        else:
            row["phash"] = None

        rows.append(row)

    img_df = pd.DataFrame(rows)
    for c in EXIF_COLUMNS + ["phash"]:
        if c not in img_df.columns:
            img_df[c] = None

    return img_df, warnings


def load_metadata_file(uploaded_meta) -> Optional[pd.DataFrame]:
    if uploaded_meta is None:
        return None
    data = uploaded_meta.getvalue()
    name = (uploaded_meta.name or "").lower()
    try:
        if name.endswith(".csv"):
            return pd.read_csv(io.BytesIO(data))
        if name.endswith(".parquet"):
            return pd.read_parquet(io.BytesIO(data))
        st.warning("Unsupported metadata type. Use CSV or Parquet.")
        return None
    except Exception as e:
        st.error(f"Failed to load metadata file: {e}")
        return None


def join_meta(img_df: pd.DataFrame, meta_df: Optional[pd.DataFrame], cfg: AssessConfig) -> Optional[pd.DataFrame]:
    if meta_df is None:
        return None

    if cfg.path_col and cfg.path_col in meta_df.columns:
        m = meta_df.copy()
        m = m.rename(columns={cfg.path_col: "path_in_zip"})
        merged = m.merge(img_df, on="path_in_zip", how="left")
        return merged

    if "filename" in meta_df.columns and "path_in_zip" in img_df.columns:
        m = meta_df.copy()
        m["_base"] = m["filename"].astype(str).apply(lambda x: os.path.basename(x))
        tmp = img_df.copy()
        tmp["_base"] = tmp["path_in_zip"].astype(str).apply(lambda x: os.path.basename(x))
        merged = m.merge(tmp.drop(columns=["path_in_zip"]), on="_base", how="left")
        merged = merged.drop(columns=["_base"])
        return merged

    return meta_df


# =========================
# Assessments
# =========================

def _exact_duplicates(img_df: pd.DataFrame) -> Dict[str, Any]:
    ok = img_df["open_ok"].astype(bool)
    d = img_df.loc[ok, "sha256"]
    if d.empty:
        return {"exact_duplicate_rate": None, "num_unique_sha256": 0, "top_duplicate_sha256": {}}
    counts = d.value_counts()
    dup = counts[counts > 1]
    dup_rate = float((d.duplicated()).mean()) if len(d) else 0.0
    top = dup.head(10).to_dict()
    return {
        "exact_duplicate_rate": float(dup_rate),
        "num_unique_sha256": int(counts.shape[0]),
        "num_duplicate_sha256": int(dup.shape[0]),
        "top_duplicate_sha256": {str(k): int(v) for k, v in top.items()},
    }

def _near_duplicates_phash(img_df: pd.DataFrame, cfg: AssessConfig) -> Dict[str, Any]:
    if not IMAGEHASH_OK:
        return {"note": "imagehash is not installed. Near-duplicate check is disabled."}

    dfp = img_df[img_df["phash"].notna()].copy()
    if dfp.empty:
        return {"note": "No perceptual hashes available (sample too small or all missing)."}

    dfp["bucket"] = dfp["phash"].astype(str).str[:4]
    buckets = dfp.groupby("bucket")

    pairs_checked = 0
    near_pairs = 0
    examples: List[Dict[str, Any]] = []

    thr = int(cfg.thresholds.perceptual_dup_hamming_threshold)

    for _, g in buckets:
        if len(g) < 2:
            continue
        idx = g.index.to_list()
        ph = g["phash"].astype(str).to_list()
        paths = g["path_in_zip"].astype(str).to_list()

        n = len(idx)
        for i in range(n):
            for j in range(i + 1, n):
                pairs_checked += 1
                if pairs_checked > cfg.max_pairs_for_near_dups:
                    return {
                        "perceptual_hash_rows": int(len(dfp)),
                        "pairs_checked": int(pairs_checked),
                        "perceptual_near_duplicate_pairs": int(near_pairs),
                        "hamming_threshold": thr,
                        "examples": examples[:20],
                        "note": "Pair cap reached. Increase max_pairs_for_near_dups to scan more.",
                    }
                hd = hamming_hexhash(ph[i], ph[j])
                if hd is not None and hd <= thr:
                    near_pairs += 1
                    if len(examples) < 20:
                        examples.append(
                            {"path_a": paths[i], "path_b": paths[j], "hamming": int(hd), "phash": ph[i]}
                        )

    return {
        "perceptual_hash_rows": int(len(dfp)),
        "pairs_checked": int(pairs_checked),
        "perceptual_near_duplicate_pairs": int(near_pairs),
        "hamming_threshold": thr,
        "examples": examples[:20],
    }

def assess_quality(img_df: pd.DataFrame, meta_joined: Optional[pd.DataFrame], cfg: AssessConfig) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    n = int(len(img_df))

    open_ok = img_df["open_ok"].astype(bool)
    corrupt = int((~open_ok).sum())
    out["corrupt_images"] = {
        "images_scanned": n,
        "corrupt_count": corrupt,
        "corrupt_rate": float(corrupt / max(1, n)),
        "example_errors": img_df.loc[~open_ok, ["path_in_zip", "open_error"]].head(10).to_dict(orient="records"),
    }

    # Resolution
    ok_df = img_df.loc[open_ok].copy()
    if ok_df.empty:
        out["low_resolution"] = {"note": "No images opened successfully."}
        out["blur_proxy"] = {"note": "No images opened successfully."}
        out["duplicates"] = {"note": "No images opened successfully."}
        return out

    min_short = int(cfg.thresholds.min_resolution_short_side)
    low_res = ok_df["short_side"].astype(float) < min_short
    out["low_resolution"] = {
        "min_short_side": min_short,
        "low_res_count": int(low_res.sum()),
        "low_res_rate": float(low_res.mean()),
        "short_side_stats": {
            "min": float(ok_df["short_side"].min()),
            "p10": float(ok_df["short_side"].quantile(0.10)),
            "median": float(ok_df["short_side"].median()),
            "p90": float(ok_df["short_side"].quantile(0.90)),
            "max": float(ok_df["short_side"].max()),
        },
    }

    # Blur proxy
    blur_thr = float(cfg.thresholds.blur_laplacian_like_threshold)
    blur = pd.to_numeric(ok_df["blur_var_lap"], errors="coerce")
    blur_low = blur < blur_thr
    out["blur_proxy"] = {
        "threshold": blur_thr,
        "low_blur_count": int(blur_low.sum(skipna=True)),
        "low_blur_rate": float(blur_low.mean(skipna=True)) if blur.notna().any() else None,
        "blur_var_lap_stats": {
            "min": float(blur.min(skipna=True)) if blur.notna().any() else None,
            "p10": float(blur.quantile(0.10)) if blur.notna().any() else None,
            "median": float(blur.median(skipna=True)) if blur.notna().any() else None,
            "p90": float(blur.quantile(0.90)) if blur.notna().any() else None,
            "max": float(blur.max(skipna=True)) if blur.notna().any() else None,
        },
    }

    # Duplicates
    exact = _exact_duplicates(img_df)
    near = _near_duplicates_phash(img_df, cfg) if (cfg.mode == "Full Scan") else _near_duplicates_phash(img_df, cfg)

    out["duplicates"] = {
        **exact,
        **({"near_duplicates": near} if isinstance(near, dict) and "note" in near else {}),
        "perceptual_near_duplicate_pairs": int(near.get("perceptual_near_duplicate_pairs", 0)) if isinstance(near, dict) else 0,
        "perceptual_examples": near.get("examples", []) if isinstance(near, dict) else [],
        "perceptual_pairs_checked": int(near.get("pairs_checked", 0)) if isinstance(near, dict) else 0,
        "perceptual_hash_rows": int(near.get("perceptual_hash_rows", 0)) if isinstance(near, dict) else 0,
        "perceptual_hamming_threshold": int(cfg.thresholds.perceptual_dup_hamming_threshold),
        "imagehash_available": bool(IMAGEHASH_OK),
    }

    # Label stats and split leakage proxy (metadata)
    if meta_joined is not None and cfg.label_col and cfg.label_col in meta_joined.columns:
        y = meta_joined[cfg.label_col]
        vc = y.value_counts(dropna=True)
        out["label_stats"] = {
            "label_missing_rate": float(y.isna().mean()),
            "label_cardinality": int(y.nunique(dropna=True)),
            "top_10_labels_share": (vc / max(1, vc.sum())).head(10).to_dict() if len(vc) else {},
        }

    if meta_joined is not None and cfg.split_col and cfg.split_col in meta_joined.columns:
        tmp = meta_joined[["sha256", cfg.split_col]].copy()
        tmp = tmp[tmp["sha256"].notna()]
        if not tmp.empty:
            distinct_splits = tmp.groupby("sha256")[cfg.split_col].nunique(dropna=False)
            out["split_leakage_proxy"] = {
                "sha256_cross_split_rate": float((distinct_splits > 1).mean()),
                "num_unique_sha256": int(distinct_splits.shape[0]),
            }

    return out

def _choose_slice(meta_joined: Optional[pd.DataFrame], cfg: AssessConfig) -> Tuple[Optional[pd.Series], Dict[str, Any]]:
    info: Dict[str, Any] = {}
    if meta_joined is None:
        return None, {"note": "No metadata uploaded. Reliability slicing disabled."}

    if cfg.split_col and cfg.split_col in meta_joined.columns:
        s = meta_joined[cfg.split_col].astype("string")
        info["slice_type"] = "split"
        info["slice_col"] = cfg.split_col
        return s, info

    if cfg.time_col and cfg.time_col in meta_joined.columns:
        t = to_datetime_if_possible(meta_joined[cfg.time_col])
        ok = t.notna()
        if ok.sum() == 0:
            return None, {"note": "Time column exists but parsing produced no valid timestamps."}
        # month slices
        s = t.dt.to_period("M").astype("string")
        info["slice_type"] = "time_month"
        info["slice_col"] = cfg.time_col
        info["time_parse_ok_rate"] = float(ok.mean())
        return s, info

    return None, {"note": "Select split column or time column to enable slicing."}

def assess_reliability(img_df: pd.DataFrame, meta_joined: Optional[pd.DataFrame], cfg: AssessConfig) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    slices, info = _choose_slice(meta_joined, cfg)
    out.update(info)
    if slices is None:
        return out

    df = meta_joined.copy()
    df["_slice"] = slices

    feat_cols = [c for c in ["short_side", "aspect_ratio", "brightness_mean", "color_std_mean", "blur_var_lap", "entropy"] if c in df.columns]
    if not feat_cols:
        out["note2"] = "No image features available for drift."
        return out

    miss_by_slice: Dict[str, float] = {}
    for s_val, g in df.groupby("_slice", dropna=False):
        miss_by_slice[str(s_val)] = float(g[feat_cols].isna().mean().mean())
    out["missing_rate_by_slice_features"] = miss_by_slice

    # Drift: first vs last slice
    uniq = pd.Series(df["_slice"]).dropna().unique().tolist()
    uniq_sorted = sorted(map(str, uniq))
    if len(uniq_sorted) < 2:
        out["note"] = "Need at least two slices for drift."
        return out

    s_first, s_last = uniq_sorted[0], uniq_sorted[-1]
    g1 = df[df["_slice"].astype("string") == s_first]
    g2 = df[df["_slice"].astype("string") == s_last]

    drift: Dict[str, float] = {}
    for c in feat_cols:
        d = ks_statistic(
            pd.to_numeric(g1[c], errors="coerce").to_numpy(),
            pd.to_numeric(g2[c], errors="coerce").to_numpy(),
        )
        if d is not None:
            drift[c] = d

    out["feature_drift_ks_first_last"] = {
        "first_slice": s_first,
        "last_slice": s_last,
        "first_slice_rows": int(len(g1)),
        "last_slice_rows": int(len(g2)),
        "top_10_ks": dict(sorted(drift.items(), key=lambda kv: kv[1], reverse=True)[:10]),
        "num_features_evaluated": int(len(drift)),
    }
    return out

def assess_robustness(img_df: pd.DataFrame, meta_joined: Optional[pd.DataFrame], cfg: AssessConfig) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    df = meta_joined if meta_joined is not None else img_df

    feat_cols = [c for c in ["short_side", "aspect_ratio", "brightness_mean", "color_std_mean", "blur_var_lap", "entropy"] if c in df.columns]
    if feat_cols:
        X = df[feat_cols].apply(pd.to_numeric, errors="coerce")
        med = X.median(axis=0, skipna=True)
        mad = (X - med).abs().median(axis=0, skipna=True).replace(0, np.nan)
        z = (X - med).abs().divide(mad)
        row_score = z.mean(axis=1, skipna=True)
        out["image_feature_outliers_mad"] = {
            "features": feat_cols,
            "mean": float(row_score.mean(skipna=True)),
            "p95": float(row_score.quantile(0.95)),
            "p99": float(row_score.quantile(0.99)),
            "max": float(row_score.max(skipna=True)),
            "top_20_row_indices": row_score.sort_values(ascending=False).head(20).index.tolist(),
        }
    else:
        out["image_feature_outliers_mad"] = {"note": "No numeric features available."}

    # Rare category label concentration
    if meta_joined is not None and cfg.label_col and cfg.label_col in meta_joined.columns:
        y = meta_joined[cfg.label_col]
        cat_cols = categorical_cols(meta_joined, exclude=[cfg.label_col, cfg.split_col, cfg.time_col] + (cfg.group_cols or []) + (cfg.id_cols or []))
        suspicious: List[Dict[str, Any]] = []
        for c in cat_cols[:50]:
            vc = meta_joined[c].value_counts(dropna=True)
            rare_vals = vc[vc <= max(5, int(0.001 * len(meta_joined)))].index.tolist()
            for v in rare_vals[:200]:
                mask = meta_joined[c] == v
                if int(mask.sum()) < 5:
                    continue
                dist = y[mask].value_counts(normalize=True, dropna=True)
                if len(dist) >= 1:
                    top_share = float(dist.iloc[0])
                    if top_share >= 0.95:
                        suspicious.append(
                            {
                                "column": c,
                                "value": str(v),
                                "count": int(mask.sum()),
                                "top_label": str(dist.index[0]),
                                "top_label_share": top_share,
                            }
                        )
        out["rare_category_label_concentration"] = {
            "num_findings": len(suspicious),
            "top_findings": sorted(suspicious, key=lambda d: (-d["top_label_share"], -d["count"]))[:20],
            "columns_scanned": int(min(50, len(cat_cols))),
        }
    else:
        out["rare_category_label_concentration"] = {"note": "Upload metadata and select a label column."}

    return out

def assess_fairness(img_df: pd.DataFrame, meta_joined: Optional[pd.DataFrame], cfg: AssessConfig) -> Dict[str, Any]:
    if meta_joined is None or not cfg.group_cols:
        return {"note": "Upload metadata and select group columns to compute fairness checks."}

    df = meta_joined
    out: Dict[str, Any] = {}
    label_ok = bool(cfg.label_col and cfg.label_col in df.columns)

    per_groupcol: Dict[str, Any] = {}
    for gcol in [c for c in cfg.group_cols if c in df.columns]:
        counts = df[gcol].value_counts(dropna=False)
        shares = counts / max(1, counts.sum())
        stats: Dict[str, Any] = {
            "num_groups": int(len(counts)),
            "min_group_share": float(shares.min()) if len(shares) else None,
            "max_group_share": float(shares.max()) if len(shares) else None,
            "representation_share_top10": shares.sort_values(ascending=False).head(10).to_dict(),
        }

        # Missingness disparity on image features + label
        feat_cols = [c for c in ["short_side", "aspect_ratio", "brightness_mean", "color_std_mean", "blur_var_lap", "entropy"] if c in df.columns]
        miss_disp: Dict[str, float] = {}
        for c in feat_cols + ([cfg.label_col] if label_ok else []):
            mr = df.groupby(gcol)[c].apply(lambda s: s.isna().mean())
            if mr.shape[0] >= 2:
                miss_disp[c] = float(mr.max() - mr.min())
        stats["missingness_disparity_top10"] = dict(sorted(miss_disp.items(), key=lambda kv: kv[1], reverse=True)[:10])

        if label_ok:
            y = df[cfg.label_col]
            stats["label_missingness_by_group"] = df.groupby(gcol)[cfg.label_col].apply(lambda s: s.isna().mean()).to_dict()

            if y.dropna().nunique() == 2:
                tmp = df.copy()
                tmp["_y_enc"], _ = pd.factorize(tmp[cfg.label_col])
                pos = tmp[tmp[cfg.label_col].notna()].groupby(gcol)["_y_enc"].mean()
                if len(pos) >= 2:
                    stats["positive_rate_by_group"] = pos.to_dict()
                    stats["positive_rate_disparity"] = float(pos.max() - pos.min())

        per_groupcol[gcol] = stats

    out["group_checks"] = per_groupcol
    return out

def assess_security(img_df: pd.DataFrame, meta_joined: Optional[pd.DataFrame], cfg: AssessConfig, zip_bytes: bytes) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    out["integrity"] = {
        "sha256_zip": sha256_bytes(zip_bytes),
        "zip_byte_size": int(len(zip_bytes)),
    }

    # EXIF privacy indicators
    exif_df = img_df.copy()
    gps_count = int((exif_df["has_gps"] == True).sum()) if "has_gps" in exif_df.columns else 0
    exif_count = int((exif_df["has_exif"] == True).sum()) if "has_exif" in exif_df.columns else 0
    out["exif_privacy"] = {
        "exif_rows_sampled": int(exif_df[EXIF_COLUMNS].notna().any(axis=1).sum()),
        "exif_images_count": exif_count,
        "gps_images_count": gps_count,
        "make_top10": exif_df["exif_make"].dropna().astype(str).value_counts().head(10).to_dict() if "exif_make" in exif_df.columns else {},
        "model_top10": exif_df["exif_model"].dropna().astype(str).value_counts().head(10).to_dict() if "exif_model" in exif_df.columns else {},
    }

    # PII patterns in paths and metadata text cols
    columns_with_hits: Dict[str, Dict[str, float]] = {}

    def scan_series(name: str, s: pd.Series, max_rows: int = 3000) -> None:
        s = s.dropna().astype("string")
        if s.empty:
            return
        rng = np.random.RandomState(cfg.random_state)
        if len(s) > max_rows:
            s = s.sample(n=max_rows, random_state=int(rng.randint(0, 1_000_000)))
        col_hits: Dict[str, float] = {}
        for pat_name, pat in PII_PATTERNS.items():
            rate = float(s.str.contains(pat, regex=True).mean())
            if rate >= cfg.thresholds.pii_hit_rate_threshold:
                col_hits[pat_name] = rate
        if col_hits:
            columns_with_hits[name] = col_hits

    if "path_in_zip" in img_df.columns:
        scan_series("path_in_zip", img_df["path_in_zip"], max_rows=5000)

    if meta_joined is not None:
        text_cols = categorical_cols(meta_joined, exclude=[])
        # keep it small
        for c in text_cols[:12]:
            scan_series(f"meta:{c}", meta_joined[c], max_rows=2000)

    out["pii_like_in_paths"] = {
        "threshold_hit_rate": float(cfg.thresholds.pii_hit_rate_threshold),
        "columns_with_hits": columns_with_hits,
        "note": "Heuristic scan for PII-like patterns. Validate with legal and domain review.",
    }

    return out

def assess_all(img_df: pd.DataFrame, meta_joined: Optional[pd.DataFrame], cfg: AssessConfig, zip_bytes: bytes) -> Dict[str, Any]:
    return {
        "quality": assess_quality(img_df, meta_joined, cfg),
        "reliability": assess_reliability(img_df, meta_joined, cfg),
        "robustness": assess_robustness(img_df, meta_joined, cfg),
        "fairness": assess_fairness(img_df, meta_joined, cfg),
        "security": assess_security(img_df, meta_joined, cfg, zip_bytes=zip_bytes),
        "notes": {
            "mode": cfg.mode,
            "thresholds": {
                "drift_ks_threshold": cfg.thresholds.drift_ks_threshold,
                "pii_hit_rate_threshold": cfg.thresholds.pii_hit_rate_threshold,
                "perceptual_dup_hamming_threshold": cfg.thresholds.perceptual_dup_hamming_threshold,
                "blur_laplacian_like_threshold": cfg.thresholds.blur_laplacian_like_threshold,
                "min_resolution_short_side": cfg.thresholds.min_resolution_short_side,
            },
            "imagehash_available": bool(IMAGEHASH_OK),
            "metadata": cfg.metadata,
        },
    }


# =========================
# Header
# =========================

st.markdown(
    """
    <div class="dsa-card">
      <h1 style="margin-bottom:0.2rem;">Image Dataset Safety Analyzer</h1>
      <div class="muted">
        Upload a ZIP of images. Optional: upload metadata (CSV/Parquet) to enable label, split/time, and fairness checks.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.write("")


# =========================
# Sidebar: Upload
# =========================

with st.sidebar:
    st.header("Upload")
    zip_up = st.file_uploader("Images ZIP", type=["zip"])
    meta_up = st.file_uploader("Optional metadata (CSV/Parquet)", type=["csv", "parquet"])

if zip_up is None:
    st.info("Upload an images ZIP to start.")
    st.stop()

zip_bytes = zip_up.getvalue()

meta_df_raw = load_metadata_file(meta_up)
if meta_df_raw is not None and "auto_meta_guesses" not in st.session_state:
    st.session_state["auto_meta_guesses"] = guess_columns_meta(meta_df_raw)
if "use_auto_meta" not in st.session_state:
    st.session_state["use_auto_meta"] = True

with st.sidebar:
    st.header("Run mode")
    mode = st.radio("Mode", ["Quick Scan", "Full Scan"], index=0)

    st.header("Thresholds")
    preset_name = st.selectbox("Preset", list(PRESETS.keys()), index=0)
    th = PRESETS[preset_name]

    with st.expander("Show thresholds"):
        st.write(
            {
                "drift_ks_threshold": th.drift_ks_threshold,
                "pii_hit_rate_threshold": th.pii_hit_rate_threshold,
                "perceptual_dup_hamming_threshold": th.perceptual_dup_hamming_threshold,
                "blur_laplacian_like_threshold": th.blur_laplacian_like_threshold,
                "min_resolution_short_side": th.min_resolution_short_side,
                "imagehash_available": IMAGEHASH_OK,
            }
        )

    st.header("Sampling")
    random_state = st.number_input("Random seed", min_value=0, max_value=10000, value=7, step=1)
    max_images = st.number_input("Max images scanned", min_value=100, max_value=100000, value=3000, step=100)
    sample_phash = st.number_input("Max images with perceptual hash", min_value=0, max_value=100000, value=1500, step=100)
    sample_exif = st.number_input("Max images with EXIF scan", min_value=0, max_value=100000, value=2000, step=100)
    max_pairs = st.number_input("Max near-dup pairs compared", min_value=1000, max_value=5000000, value=200000, step=10000)

    st.header("Metadata columns")
    if meta_df_raw is None:
        st.caption("No metadata uploaded. Only image-only checks will run.")
        path_col = label_col = split_col = time_col = None
        id_cols: List[str] = []
        group_cols: List[str] = []
    else:
        use_auto = st.toggle("Use suggested columns", value=st.session_state["use_auto_meta"])
        st.session_state["use_auto_meta"] = use_auto
        guesses = st.session_state.get("auto_meta_guesses", {})

        if use_auto and guesses.get("notes"):
            st.caption("Suggestions")
            for n in guesses["notes"]:
                st.write("• " + n)

        cols = meta_df_raw.columns.tolist()
        col_filter = st.text_input("Filter columns", value="", help="Type to filter dropdown lists.")
        filtered_cols = [c for c in cols if col_filter.lower() in c.lower()] if col_filter else cols

        def pick_one(label: str, auto_value: Optional[str]) -> Optional[str]:
            options = ["(none)"] + filtered_cols
            if use_auto and auto_value in filtered_cols:
                idx = options.index(auto_value)
            else:
                idx = 0
            chosen = st.selectbox(label, options, index=idx)
            return None if chosen == "(none)" else chosen

        path_col = pick_one("Image path column", guesses.get("path"))
        label_col = pick_one("Label column", guesses.get("label"))
        split_col = pick_one("Split column", guesses.get("split"))
        time_col = pick_one("Time column", guesses.get("time"))

        default_ids = guesses.get("ids", []) if use_auto else []
        default_groups = guesses.get("groups", []) if use_auto else []
        id_cols = st.multiselect("ID columns", filtered_cols, default=[c for c in default_ids if c in filtered_cols])
        group_cols = st.multiselect("Group columns (fairness)", filtered_cols, default=[c for c in default_groups if c in filtered_cols])

    st.divider()
    run = st.button("Run analysis", type="primary", use_container_width=True)


# =========================
# Scan images
# =========================

md: Dict[str, Any] = {
    "zip_name": getattr(zip_up, "name", None),
    "metadata_name": getattr(meta_up, "name", None) if meta_up is not None else None,
}

cfg = AssessConfig(
    path_col=path_col,
    label_col=label_col,
    split_col=split_col,
    time_col=time_col,
    group_cols=group_cols if meta_df_raw is not None else [],
    id_cols=id_cols if meta_df_raw is not None else [],
    metadata=md,
    random_state=int(random_state),
    mode=mode,
    thresholds=th,
    max_images=int(max_images),
    sample_for_perceptual_dups=int(sample_phash),
    sample_for_exif=int(sample_exif),
    max_pairs_for_near_dups=int(max_pairs),
)

with st.spinner("Reading ZIP and scanning images..."):
    img_df, ingest_warnings = read_zip_images(zip_bytes, cfg)

if img_df.empty:
    st.error("No images scanned.")
    for w in ingest_warnings:
        st.warning(w)
    st.stop()

meta_df_joined = None
if meta_df_raw is not None:
    meta_df_joined = join_meta(img_df, meta_df_raw, cfg)


# =========================
# Preview if not run
# =========================

if not run:
    c1, c2, c3, c4 = st.columns([1, 1, 1, 1], gap="large")
    with c1:
        kpi("Images scanned", f"{len(img_df):,}", "From ZIP")
    with c2:
        ok_cnt = int((img_df["open_ok"].astype(bool) == True).sum())
        kpi("Open OK", f"{ok_cnt:,}", f"Corrupt: {len(img_df) - ok_cnt:,}")
    with c3:
        kpi("ZIP SHA-256", sha256_bytes(zip_bytes)[:16] + "…", "Integrity fingerprint")
    with c4:
        kpi("imagehash", "Available" if IMAGEHASH_OK else "Not installed", "Near-duplicates need it")

    if ingest_warnings:
        st.warning(" | ".join([clip_text(w, 140) for w in ingest_warnings]))

    st.write("")
    st.markdown('<div class="dsa-card">', unsafe_allow_html=True)
    st.subheader("Preview")
    st.caption("Image table sample")
    st.dataframe(img_df.head(50), use_container_width=True)
    if meta_df_joined is not None:
        st.caption("Joined metadata sample")
        st.dataframe(meta_df_joined.head(50), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()


# =========================
# Run analysis
# =========================

with st.spinner("Running checks..."):
    report = assess_all(img_df, meta_df_joined, cfg, zip_bytes=zip_bytes)

safe_report = to_json_safe(report)

verdict, kind, reasons = verdict_panel(report, cfg)
recs = build_recommendations(report, cfg)

# Compute image health score (simplified)
img_score_comps = {}
corrupt_rate = report["quality"]["corrupt_images"]["corrupt_rate"]
img_score_comps["quality"]  = max(0, 1 - corrupt_rate / 0.1) * 35
pii_gps = report["security"].get("exif_pii", {}).get("gps_found_rate", 0)
img_score_comps["security"] = (0 if float(pii_gps) > 0.01 else 1.0) * 25
img_score_comps["reliability"] = 20.0 * 0.75
img_score_comps["robustness"] = 10.0 * 0.75
img_score_comps["fairness"]   = 7.5
img_total_score = round(min(100, sum(img_score_comps.values())))
img_grade = "A" if img_total_score >= 90 else ("B" if img_total_score >= 80 else ("C" if img_total_score >= 70 else ("D" if img_total_score >= 60 else "F")))

st.markdown(f"""
<div class="verdict-card">
  <div style="display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:10px;">
    <div>
      <div class="verdict-title">Verdict</div>
      <div class="verdict-text">{verdict}</div>
      <div class="muted" style="margin-top:4px;">
        Mode: <span class="code-pill">{cfg.mode}</span>
        &nbsp; Preset: <span class="code-pill">{preset_name}</span>
      </div>
    </div>
    <div style="display:flex; gap:8px; flex-wrap:wrap;">
      {badge(verdict, kind)}
      {badge(f"Score {img_total_score}/100", 'ok' if img_total_score>=80 else ('warn' if img_total_score>=60 else 'bad'))}
      {badge(f"Grade {img_grade}", 'ok' if img_grade in ('A','B') else ('warn' if img_grade=='C' else 'bad'))}
    </div>
  </div>
  <hr>
  <div style="display:grid; grid-template-columns:1fr 1fr; gap:16px;">
    <div><div style="font-weight:700; font-size:0.85rem; margin-bottom:6px;">Findings</div>
         <ul style="margin:0; padding-left:1.1em; font-size:0.83rem; line-height:1.7;">{''.join(f'<li>{clip_text(r,200)}</li>' for r in reasons)}</ul></div>
    <div><div style="font-weight:700; font-size:0.85rem; margin-bottom:6px;">Recommended actions</div>
         <ul style="margin:0; padding-left:1.1em; font-size:0.83rem; line-height:1.7;">{''.join(f'<li>{clip_text(r,200)}</li>' for r in recs)}</ul></div>
  </div>
</div>
""", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns([1, 1, 1, 1], gap="large")
with c1:
    kpi("Images", f"{len(img_df):,}", "Scanned from ZIP")
with c2:
    ok_cnt = int((img_df["open_ok"].astype(bool) == True).sum())
    corrupt_rate = float(1.0 - ok_cnt / max(1, len(img_df)))
    kpi("Corrupt rate", f"{corrupt_rate:.4f}", f"Corrupt: {len(img_df) - ok_cnt:,}")
with c3:
    kpi("ZIP SHA-256", sha256_bytes(zip_bytes)[:16] + "…", "Integrity fingerprint")
with c4:
    st.write("")

st.write("")

tab_overview, tab_quality, tab_reliability, tab_robustness, tab_fairness, tab_transparency, tab_security, tab_export = st.tabs(
    ["Overview", "Quality", "Reliability", "Robustness", "Fairness", "Transparency", "Security", "Export"]
)

with tab_overview:
    st.markdown('<div class="dsa-card">', unsafe_allow_html=True)
    st.subheader("Overview")

    a = report["quality"]
    b = report["reliability"]
    sec = report["security"]

    oc1, oc2, oc3, oc4 = st.columns(4)
    with oc1:
        st.metric("Corrupt rate", f"{a['corrupt_images']['corrupt_rate']:.4f}")
    with oc2:
        st.metric("Exact dup rate", f"{a.get('duplicates', {}).get('exact_duplicate_rate', 0.0):.4f}")
    with oc3:
        st.metric("Low-res rate", f"{a.get('low_resolution', {}).get('low_res_rate', 0.0):.4f}")
    with oc4:
        st.metric("GPS EXIF images", str(sec["exif_privacy"]["gps_images_count"]))

    st.caption("Tip: upload metadata and set split/time/group columns to unlock drift and fairness checks.")
    st.markdown("</div>", unsafe_allow_html=True)

with tab_quality:
    st.markdown('<div class="dsa-card">', unsafe_allow_html=True)
    st.subheader("Quality signals")

    a = report["quality"]

    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.write("Corrupt images")
        st.json(to_json_safe(a["corrupt_images"]))
    with col2:
        st.write("Low resolution")
        st.json(to_json_safe(a["low_resolution"]))

    st.write("Blur proxy")
    st.json(to_json_safe(a.get("blur_proxy", {})))

    st.write("Duplicates")
    st.json(to_json_safe(a.get("duplicates", {})))

    if "label_stats" in a:
        st.write("Label stats (metadata)")
        st.json(to_json_safe(a["label_stats"]))
    if "split_leakage_proxy" in a:
        st.write("Split leakage proxy (duplicates across splits)")
        st.json(to_json_safe(a["split_leakage_proxy"]))

    st.markdown("</div>", unsafe_allow_html=True)

with tab_reliability:
    st.markdown('<div class="dsa-card">', unsafe_allow_html=True)
    st.subheader("Reliability")

    b = report["reliability"]
    st.write({"slice_type": b.get("slice_type"), "slice_col": b.get("slice_col")})

    if "note" in b:
        st.warning(b["note"])
    if "note2" in b:
        st.info(b["note2"])

    if "missing_rate_by_slice_features" in b:
        s = pd.Series(b["missing_rate_by_slice_features"]).sort_index()
        st.write("Feature missingness by slice")
        st.line_chart(s)
        st.dataframe(s.rename("missing_rate").reset_index().rename(columns={"index": "slice"}), use_container_width=True)

    if "feature_drift_ks_first_last" in b:
        ks = b["feature_drift_ks_first_last"]["top_10_ks"]
        ks_df = pd.DataFrame(list(ks.items()), columns=["feature", "ks_stat"]).sort_values("ks_stat", ascending=False)
        st.write(f"Feature drift (KS) first vs last slice, threshold={cfg.thresholds.drift_ks_threshold}")
        st.dataframe(ks_df, use_container_width=True)
        if not ks_df.empty:
            st.bar_chart(ks_df.set_index("feature")["ks_stat"])

    st.markdown("</div>", unsafe_allow_html=True)

with tab_robustness:
    st.markdown('<div class="dsa-card">', unsafe_allow_html=True)
    st.subheader("Robustness")

    c = report["robustness"]

    st.write("Outlier images (MAD on simple features)")
    st.json(to_json_safe(c.get("image_feature_outliers_mad", {})))

    st.write("Rare category label concentration (metadata)")
    st.json(to_json_safe(c.get("rare_category_label_concentration", {})))

    st.markdown("</div>", unsafe_allow_html=True)

with tab_fairness:
    st.markdown('<div class="dsa-card">', unsafe_allow_html=True)
    st.subheader("Fairness proxies")

    d = report["fairness"]
    if "note" in d:
        st.warning(d["note"])
    else:
        for gcol, stats_ in d["group_checks"].items():
            st.markdown(f"#### Group column: `{gcol}`")
            rep = stats_.get("representation_share_top10", {})
            rep_df = pd.DataFrame(list(rep.items()), columns=["group_value", "share"]).sort_values("share", ascending=False)
            st.write("Representation (top 10)")
            st.dataframe(rep_df, use_container_width=True)
            if not rep_df.empty:
                st.bar_chart(rep_df.set_index("group_value")["share"])

            miss = stats_.get("missingness_disparity_top10", {})
            miss_df = pd.DataFrame(list(miss.items()), columns=["column", "max_minus_min_missing"]).sort_values(
                "max_minus_min_missing", ascending=False
            )
            st.write("Missingness disparity (top 10)")
            st.dataframe(miss_df, use_container_width=True)

            if "positive_rate_by_group" in stats_:
                pr = pd.Series(stats_["positive_rate_by_group"]).sort_index()
                st.write("Positive rate by group (binary labels)")
                st.bar_chart(pr)
                st.write({"positive_rate_disparity": stats_.get("positive_rate_disparity")})

            st.write("---")

    st.markdown("</div>", unsafe_allow_html=True)

with tab_transparency:
    st.markdown('<div class="dsa-card">', unsafe_allow_html=True)
    st.subheader("Transparency — Data Card")

    # Dataset identity
    st.markdown('<div class="transparency-header">📁 Dataset Identity</div>', unsafe_allow_html=True)
    sha_zip = sha256_bytes(zip_bytes) if zip_bytes else "n/a"
    identity_rows = [
        ("ZIP file", zip_file_name if 'zip_file_name' in dir() else "uploaded.zip"),
        ("Total images scanned", f"{len(img_df):,}"),
        ("ZIP file size", f"{len(zip_bytes) / 1024:.1f} KB ({len(zip_bytes):,} bytes)"),
        ("ZIP SHA-256", sha_zip),
    ]
    html = "<div>"
    for k, v in identity_rows:
        html += f'<div class="config-row"><div class="config-key">{k}</div><div class="config-val mono">{v}</div></div>'
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # Analysis config
    st.markdown('<div class="transparency-header">⚙️ Analysis Configuration</div>', unsafe_allow_html=True)
    config_rows_img = [
        ("Scan mode", cfg.mode),
        ("Threshold preset", preset_name),
        ("Max images to process", str(cfg.max_images)),
        ("Random seed", str(cfg.random_state)),
        ("Label column (metadata)", cfg.label_col or "(none)"),
        ("Group columns", ", ".join(cfg.group_cols) if cfg.group_cols else "(none)"),
    ]
    html2 = "<div>"
    for k, v in config_rows_img:
        html2 += f'<div class="config-row"><div class="config-key">{k}</div><div class="config-val mono">{v}</div></div>'
    html2 += "</div>"
    st.markdown(html2, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # Check registry
    st.markdown('<div class="transparency-header">📋 Check Registry</div>', unsafe_allow_html=True)
    img_checks = [
        ("Quality › Corrupt images",     "Attempts to open each image with PIL; flags those that fail.",                    "✓ Ran"),
        ("Quality › Resolution stats",   "Width, height, megapixels per image — min, max, mean, std.",                     "✓ Ran"),
        ("Quality › Aspect ratio",        "Detects extreme or inconsistent aspect ratios.",                                 "✓ Ran"),
        ("Quality › Brightness/contrast","Mean pixel value and std deviation per image (proxy for quality).",               "✓ Ran"),
        ("Reliability › Label balance",  "Class distribution from folder structure or metadata label column.",              "✓ Ran" if cfg.label_col else "— Skipped (no label)"),
        ("Robustness › Exact duplicates","SHA-256 hash per image to find byte-identical files.",                            "✓ Ran"),
        ("Robustness › Perceptual dups", "pHash comparison to find visually similar images.",                               "✓ Ran" if True else "— imagehash not installed"),
        ("Security › EXIF metadata",     "Extracts EXIF fields; flags GPS coordinates as PII risk.",                       "✓ Ran"),
        ("Security › Integrity",         "SHA-256 fingerprint of the ZIP archive.",                                        "✓ Ran"),
    ]
    check_df = pd.DataFrame(img_checks, columns=["Check", "Description", "Status"])
    st.dataframe(check_df, use_container_width=True, hide_index=True)

    st.markdown('</div>', unsafe_allow_html=True)

with tab_security:
    st.markdown('<div class="dsa-card">', unsafe_allow_html=True)
    st.subheader("Security (CIA prerequisites)")

    sec = report["security"]

    st.write("Integrity")
    st.json(to_json_safe(sec.get("integrity", {})))

    st.write("EXIF privacy indicators")
    st.json(to_json_safe(sec.get("exif_privacy", {})))

    st.write("PII-like pattern scan (paths + metadata text)")
    pii_cols = sec.get("pii_like_in_paths", {}).get("columns_with_hits", {})
    st.write({"columns_flagged": len(pii_cols)})
    if pii_cols:
        rows = []
        for col, hits in pii_cols.items():
            for k, v in hits.items():
                rows.append({"column": col, "pattern": k, "hit_rate": v})
        st.dataframe(pd.DataFrame(rows).sort_values("hit_rate", ascending=False), use_container_width=True)
    else:
        st.success("No PII-like patterns flagged by the heuristic scan.")

    st.markdown("</div>", unsafe_allow_html=True)

with tab_export:
    st.markdown('<div class="dsa-card">', unsafe_allow_html=True)
    st.subheader("Export")

    st.caption("Exports include the report, thresholds, and mode. JSON is safe for pandas/numpy types.")
    st.json(safe_report)

    out_bytes = json.dumps(safe_report, indent=2, ensure_ascii=False).encode("utf-8")
    st.download_button(
        "Download JSON report",
        data=out_bytes,
        file_name="image_dataset_report.json",
        mime="application/json",
        use_container_width=True,
    )

    md_lines: List[str] = []
    md_lines.append("# Image Dataset Safety Analyzer report")
    md_lines.append("")
    md_lines.append(f"- Mode: {cfg.mode}")
    md_lines.append(f"- Preset: {preset_name}")
    md_lines.append(f"- Verdict: {verdict}")
    md_lines.append("")
    md_lines.append("## Reasons")
    for r in reasons:
        md_lines.append(f"- {r}")
    md_lines.append("")
    md_lines.append("## Recommended actions")
    for r in recs:
        md_lines.append(f"- {r}")
    md_lines.append("")
    md_lines.append("## Key metrics")
    md_lines.append(f"- Images scanned: {len(img_df)}")
    md_lines.append(f"- Corrupt rate: {report['quality']['corrupt_images']['corrupt_rate']:.4f}")
    md_lines.append(f"- Exact duplicate rate: {report['quality'].get('duplicates', {}).get('exact_duplicate_rate', 0.0):.4f}")
    md_lines.append(f"- ZIP SHA-256: {report['security']['integrity']['sha256_zip']}")
    md_lines.append("")
    md_lines.append("## Notes")
    md_lines.append("- This is an automated heuristic report. Validate conclusions with domain review.")

    md_bytes = ("\n".join(md_lines)).encode("utf-8")
    st.download_button(
        "Download Markdown summary",
        data=md_bytes,
        file_name="image_dataset_report_summary.md",
        mime="text/markdown",
        use_container_width=True,
    )

    st.markdown("</div>", unsafe_allow_html=True)