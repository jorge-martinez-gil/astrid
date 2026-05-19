"""Headless ASTRID analysis API.

This module contains reusable analysis functions that do not depend on
Streamlit page execution. The initial scope is tabular data so experiments can
run reproducibly from scripts and CI; the Streamlit app can keep its current UI
while the research workflow grows around this stable API.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from io import BytesIO
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from audit_history import build_audit_record, evaluate_policy
from utils import (
    DEFAULT_WEIGHTS,
    PII_PATTERNS,
    approx_iqr_outlier_rate,
    categorical_cols,
    compute_health_score,
    infer_column_types,
    ks_statistic,
    numeric_cols,
    sha256_bytes,
    to_datetime_if_possible,
    to_json_safe,
)

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split

    SKLEARN_OK = True
except Exception:  # pragma: no cover - optional dependency
    SKLEARN_OK = False


@dataclass(frozen=True)
class TabularThresholds:
    drift_ks_threshold: float
    pii_hit_rate_threshold: float


TABULAR_PRESETS: Dict[str, TabularThresholds] = {
    "Balanced (recommended)": TabularThresholds(
        drift_ks_threshold=0.30,
        pii_hit_rate_threshold=0.01,
    ),
    "Strict": TabularThresholds(drift_ks_threshold=0.20, pii_hit_rate_threshold=0.005),
    "Lenient": TabularThresholds(drift_ks_threshold=0.40, pii_hit_rate_threshold=0.02),
}


@dataclass
class TabularAssessConfig:
    label_col: Optional[str] = None
    split_col: Optional[str] = None
    time_col: Optional[str] = None
    group_cols: List[str] = field(default_factory=list)
    annotator_label_cols: List[str] = field(default_factory=list)
    id_cols: List[str] = field(default_factory=list)
    random_state: int = 7
    max_categories_for_stats: int = 50
    mode: str = "Quick Scan"
    thresholds: TabularThresholds = field(
        default_factory=lambda: TABULAR_PRESETS["Balanced (recommended)"]
    )
    pii_max_rows: int = 2000
    pii_max_text_cols: int = 10
    rare_max_cat_cols: int = 50
    drift_max_num_cols: int = 50


def _name_score(name: str, patterns: List[str]) -> float:
    n = name.lower().strip()
    return sum(1.0 for p in patterns if re.search(p, n))


def guess_tabular_columns(df: pd.DataFrame) -> Dict[str, Any]:
    """Suggest common column roles using the same heuristics as the UI analyzer."""

    cols = df.columns.tolist()
    nrows = max(1, len(df))
    nunique = {c: int(df[c].nunique(dropna=True)) for c in cols}
    uniq_ratio = {c: float(nunique[c] / nrows) for c in cols}

    dt_success: Dict[str, float] = {}
    for c in cols[:250]:
        if pd.api.types.is_datetime64_any_dtype(df[c].dtype):
            dt_success[c] = 1.0
        elif pd.api.types.is_numeric_dtype(df[c].dtype):
            dt_success[c] = 0.0
        else:
            parsed = to_datetime_if_possible(df[c])
            dt_success[c] = (
                float(parsed.notna().mean())
                if pd.api.types.is_datetime64_any_dtype(parsed.dtype)
                else 0.0
            )

    label_pats = [r"\blabel\b", r"\btarget\b", r"\boutcome\b", r"\bclass\b", r"\bgt\b", r"\by\b"]
    split_pats = [r"\bsplit\b", r"\bfold\b", r"\bset\b", r"\bpartition\b"]
    time_pats = [r"\btime\b", r"\bdate\b", r"\btimestamp\b", r"\bcreated\b", r"\bupdated\b"]
    id_pats = [r"\bid\b", r"\buuid\b", r"\bguid\b", r"\buser[_\s-]?id\b", r"\bserial\b"]
    grp_pats = [r"\bgender\b", r"\bsex\b", r"\bage\b", r"\bregion\b", r"\bcountry\b", r"\bgroup\b", r"\bsite\b"]

    def rank_label(c: str) -> float:
        score = _name_score(c, label_pats) * 3
        if 2 <= nunique[c] <= min(50, int(0.01 * nrows) + 2):
            score += 2
        if uniq_ratio[c] < 0.2:
            score += 1
        if uniq_ratio[c] > 0.9:
            score -= 2
        if dt_success.get(c, 0) > 0.8:
            score -= 2
        return score

    def rank_split(c: str) -> float:
        score = _name_score(c, split_pats) * 3
        if nunique[c] <= 20:
            score += 2
        try:
            values = " ".join(df[c].dropna().astype("string").str.lower().value_counts().head(12).index)
            if any(x in values for x in ["train", "test", "val", "valid", "dev"]):
                score += 2
        except Exception:
            pass
        if uniq_ratio[c] > 0.5:
            score -= 2
        return score

    def rank_time(c: str) -> float:
        score = _name_score(c, time_pats) * 3 + dt_success.get(c, 0) * 2
        if dt_success.get(c, 0) < 0.3:
            score -= 1.5
        return score

    def rank_id(c: str) -> float:
        score = _name_score(c, id_pats) * 3
        if uniq_ratio[c] > 0.95:
            score += 2.5
        if dt_success.get(c, 0) > 0.8:
            score -= 2
        return score

    def rank_group(c: str) -> float:
        score = _name_score(c, grp_pats) * 2
        if 2 <= nunique[c] <= 50:
            score += 2
        if uniq_ratio[c] > 0.5:
            score -= 1
        if dt_success.get(c, 0) > 0.8:
            score -= 2
        return score

    sorted_label = sorted(cols, key=rank_label, reverse=True)
    sorted_split = sorted(cols, key=rank_split, reverse=True)
    sorted_time = sorted(cols, key=rank_time, reverse=True)
    sorted_id = sorted(cols, key=rank_id, reverse=True)
    sorted_group = sorted(cols, key=rank_group, reverse=True)

    label = sorted_label[0] if sorted_label and rank_label(sorted_label[0]) >= 2 else None
    split = sorted_split[0] if sorted_split and rank_split(sorted_split[0]) >= 2 else None
    time = sorted_time[0] if sorted_time and rank_time(sorted_time[0]) >= 2 else None

    ids: List[str] = []
    groups: List[str] = []
    for c in sorted_id:
        if c in {label, split, time}:
            continue
        if rank_id(c) >= 3.5:
            ids.append(c)
        if len(ids) >= 3:
            break
    for c in sorted_group:
        if c in {label, split, time} or c in set(ids):
            continue
        if rank_group(c) >= 2.5:
            groups.append(c)
        if len(groups) >= 3:
            break

    notes = (
        ([f"Guessed label: {label}"] if label else [])
        + ([f"Guessed split: {split}"] if split else [])
        + ([f"Guessed time: {time}"] if time else [])
        + ([f"Guessed IDs: {', '.join(ids)}"] if ids else [])
        + ([f"Guessed groups: {', '.join(groups)}"] if groups else [])
    )
    return {"label": label, "split": split, "time": time, "ids": ids, "groups": groups, "notes": notes}


def make_tabular_config(
    df: pd.DataFrame,
    *,
    preset: str = "Balanced (recommended)",
    mode: str = "Quick Scan",
    use_auto_columns: bool = True,
    label_col: Optional[str] = None,
    split_col: Optional[str] = None,
    time_col: Optional[str] = None,
    group_cols: Optional[List[str]] = None,
    annotator_label_cols: Optional[List[str]] = None,
    id_cols: Optional[List[str]] = None,
    random_state: int = 7,
) -> TabularAssessConfig:
    """Build a tabular config from explicit roles plus optional auto-detected roles."""

    guesses = guess_tabular_columns(df) if use_auto_columns else {}
    thresholds = TABULAR_PRESETS.get(preset, TABULAR_PRESETS["Balanced (recommended)"])
    return TabularAssessConfig(
        label_col=label_col or guesses.get("label"),
        split_col=split_col or guesses.get("split"),
        time_col=time_col or guesses.get("time"),
        group_cols=group_cols if group_cols is not None else list(guesses.get("groups", [])),
        annotator_label_cols=annotator_label_cols or [],
        id_cols=id_cols if id_cols is not None else list(guesses.get("ids", [])),
        random_state=random_state,
        mode=mode,
        thresholds=thresholds,
    )


def detect_tabular_task_type(df: pd.DataFrame, label_col: Optional[str]) -> str:
    if not label_col or label_col not in df.columns:
        return "No label selected"
    y = df[label_col]
    if pd.api.types.is_numeric_dtype(y.dtype):
        unique = int(y.nunique(dropna=True))
        if unique > 20 and unique > 0.05 * max(1, len(y)):
            return "Regression"
    unique = int(y.nunique(dropna=True))
    if unique == 2:
        return "Binary classification"
    if 2 < unique <= 50:
        return "Multi-class classification"
    return f"Label selected (card={unique})"


def assess_tabular_quality(df: pd.DataFrame, cfg: TabularAssessConfig) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    missing = df.isna().mean().sort_values(ascending=False)
    out["missingness"] = {
        "overall_missing_rate": float(df.isna().mean().mean()),
        "top_10_columns_missing_rate": missing.head(10).to_dict(),
    }
    out["duplicates"] = {"exact_duplicate_row_rate": float(df.duplicated().mean())}
    if cfg.id_cols:
        id_cols = [c for c in cfg.id_cols if c in df.columns]
        if id_cols:
            out["duplicates"]["duplicate_id_rate"] = float(df.duplicated(subset=id_cols).mean())

    num = numeric_cols(df, exclude=[c for c in [cfg.label_col, cfg.split_col, cfg.time_col] if c])
    outlier_rates: Dict[str, float] = {}
    for c in num:
        rate = approx_iqr_outlier_rate(df[c])
        if rate is not None:
            outlier_rates[c] = rate
    out["outliers_iqr"] = {
        "columns_evaluated": len(outlier_rates),
        "top_10_outlier_rate": dict(
            sorted(outlier_rates.items(), key=lambda kv: kv[1], reverse=True)[:10]
        ),
    }

    if cfg.label_col and cfg.label_col in df.columns:
        y = df[cfg.label_col]
        vc = y.value_counts(dropna=True)
        out["label_stats"] = {
            "label_missing_rate": float(y.isna().mean()),
            "label_cardinality": int(y.nunique(dropna=True)),
            "top_classes_share": (
                (vc / vc.sum()).head(10).to_dict()
                if 2 <= len(vc) <= cfg.max_categories_for_stats
                else None
            ),
        }

    if cfg.annotator_label_cols:
        annotator_cols = [c for c in cfg.annotator_label_cols if c in df.columns]
        if len(annotator_cols) >= 2:
            labels = df[annotator_cols]
            valid = ~labels.isna().all(axis=1)
            agree = (labels.nunique(axis=1, dropna=True) <= 1) & valid
            out["label_agreement"] = {
                "annotator_cols": annotator_cols,
                "rows_with_any_label": int(valid.sum()),
                "exact_agreement_rate": float(agree[valid].mean()) if valid.sum() else None,
                "disagreement_rate": float((~agree & valid).mean()) if valid.sum() else None,
            }

    if cfg.split_col and cfg.split_col in df.columns:
        split = df[cfg.split_col].astype("string")
        hashed_cols = [c for c in df.columns if c != cfg.split_col]
        row_hash = pd.util.hash_pandas_object(df[hashed_cols], index=False).astype("uint64")
        tmp = pd.DataFrame({"split": split, "row_hash": row_hash})
        distinct = tmp.groupby("row_hash")["split"].nunique()
        out["split_leakage"] = {
            "row_hash_cross_split_rate": float((distinct > 1).mean()) if len(distinct) else None,
            "num_unique_rows_hashed": int(distinct.shape[0]),
        }

    return out


def assess_tabular_reliability(df: pd.DataFrame, cfg: TabularAssessConfig) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    exclude = [c for c in [cfg.label_col, cfg.split_col, cfg.time_col] if c]
    num_all = numeric_cols(df, exclude=exclude)
    slices = None
    slice_type = None

    if cfg.time_col and cfg.time_col in df.columns:
        slice_type = "time"
        t = to_datetime_if_possible(df[cfg.time_col])
        if pd.api.types.is_datetime64_any_dtype(t.dtype):
            slices = t.dt.to_period("M").astype("string")
        else:
            v = pd.to_numeric(df[cfg.time_col], errors="coerce")
            try:
                slices = pd.qcut(v, q=4, duplicates="drop").astype("string")
            except Exception:
                slices = None
    elif cfg.split_col and cfg.split_col in df.columns:
        slice_type = "split"
        slices = df[cfg.split_col].astype("string")

    out["slice_type"] = slice_type
    if slices is None:
        out["note"] = "Select a time or split column to compute stability and drift."
        out["schema_consistency"] = {
            "num_rows": int(len(df)),
            "num_cols": int(df.shape[1]),
            "dtypes": infer_column_types(df),
            "constant_columns": [c for c in df.columns if df[c].nunique(dropna=False) <= 1],
        }
        return out

    out["missing_rate_by_slice"] = {
        str(slice_value): float(group.isna().mean().mean())
        for slice_value, group in df.groupby(slices, dropna=False)
    }

    unique_slices = pd.Series(slices).dropna().unique().tolist()
    num = num_all
    if cfg.drift_max_num_cols and len(num_all) > cfg.drift_max_num_cols:
        variances = [
            (c, float(pd.to_numeric(df[c], errors="coerce").var(skipna=True)))
            for c in num_all
        ]
        num = [c for c, _ in sorted(variances, key=lambda item: item[1], reverse=True)[: cfg.drift_max_num_cols]]

    drift: Dict[str, float] = {}
    if len(unique_slices) >= 2 and num:
        sorted_slices = sorted(map(str, unique_slices))
        first_slice, last_slice = sorted_slices[0], sorted_slices[-1]
        slice_series = pd.Series(slices).astype("string")
        first_group = df[slice_series == first_slice]
        last_group = df[slice_series == last_slice]
        for c in num:
            value = ks_statistic(
                pd.to_numeric(first_group[c], errors="coerce").to_numpy(),
                pd.to_numeric(last_group[c], errors="coerce").to_numpy(),
            )
            if value is not None:
                drift[c] = value
        out["numeric_drift_ks_first_last"] = {
            "first_slice": first_slice,
            "last_slice": last_slice,
            "first_slice_rows": int(len(first_group)),
            "last_slice_rows": int(len(last_group)),
            "num_cols_evaluated": int(len(num)),
            "top_10_ks": dict(sorted(drift.items(), key=lambda kv: kv[1], reverse=True)[:10]),
        }

    out["schema_consistency"] = {
        "num_rows": int(len(df)),
        "num_cols": int(df.shape[1]),
        "dtypes": infer_column_types(df),
        "constant_columns": [c for c in df.columns if df[c].nunique(dropna=False) <= 1],
    }
    return out


def assess_tabular_robustness(df: pd.DataFrame, cfg: TabularAssessConfig) -> Dict[str, Any]:
    out: Dict[str, Any] = {"sklearn_available": SKLEARN_OK}
    exclude = [c for c in [cfg.label_col, cfg.split_col, cfg.time_col] if c]

    if cfg.label_col and cfg.label_col in df.columns:
        y = df[cfg.label_col]
        cat_all = categorical_cols(df, exclude=exclude)
        cat = cat_all[: cfg.rare_max_cat_cols]
        suspicious = []
        for c in cat:
            vc = df[c].value_counts(dropna=True)
            rare = vc[vc <= max(5, int(0.001 * len(df)))].index.tolist()
            for value in rare[:200]:
                mask = df[c] == value
                if mask.sum() < 5:
                    continue
                distribution = y[mask].value_counts(normalize=True, dropna=True)
                if len(distribution) >= 1 and float(distribution.iloc[0]) >= 0.95:
                    suspicious.append(
                        {
                            "column": c,
                            "value": str(value),
                            "count": int(mask.sum()),
                            "top_label": str(distribution.index[0]),
                            "top_label_share": float(distribution.iloc[0]),
                        }
                    )
        out["rare_category_label_concentration"] = {
            "num_findings": len(suspicious),
            "top_findings": sorted(
                suspicious,
                key=lambda item: (-item["top_label_share"], -item["count"]),
            )[:20],
            "columns_scanned": int(len(cat)),
        }

    num = numeric_cols(df, exclude=exclude)
    if num:
        x = df[num].apply(pd.to_numeric, errors="coerce")
        med = x.median(axis=0, skipna=True)
        mad = (x - med).abs().median(axis=0, skipna=True).replace(0, np.nan)
        z = (x - med).abs().divide(mad)
        row_score = z.mean(axis=1, skipna=True)
        out["row_anomaly_score_mad"] = {
            "mean": float(row_score.mean(skipna=True)),
            "p95": float(row_score.quantile(0.95)),
            "p99": float(row_score.quantile(0.99)),
            "max": float(row_score.max(skipna=True)),
            "top_20_row_indices": row_score.sort_values(ascending=False).head(20).index.tolist(),
        }

    if SKLEARN_OK and cfg.label_col and cfg.label_col in df.columns and num:
        y = df[cfg.label_col]
        if y.dropna().nunique() == 2:
            y_enc, _ = pd.factorize(y)
            mask = y.notna()
            x = df.loc[mask, num].apply(pd.to_numeric, errors="coerce").fillna(0.0)
            y_enc = y_enc[mask.to_numpy()]
            if len(x) >= 200:
                try:
                    x_train, x_test, y_train, y_test = train_test_split(
                        x,
                        y_enc,
                        test_size=0.25,
                        random_state=cfg.random_state,
                        stratify=y_enc,
                    )
                    clf = LogisticRegression(max_iter=250, n_jobs=1)
                    clf.fit(x_train, y_train)
                    out["label_predictability_auc"] = float(
                        roc_auc_score(y_test, clf.predict_proba(x_test)[:, 1])
                    )
                except Exception as exc:
                    out["label_predictability_auc"] = {"error": str(exc)}
            else:
                out["label_predictability_auc"] = {"note": "Need >= 200 labeled rows."}
        else:
            out["label_predictability_auc"] = {"note": "Binary labels only."}
    else:
        out["label_predictability_auc"] = {
            "note": "Install scikit-learn and select label plus numeric features."
        }

    return out


def assess_tabular_fairness(df: pd.DataFrame, cfg: TabularAssessConfig) -> Dict[str, Any]:
    if not cfg.group_cols:
        return {"note": "Select group columns to compute fairness checks."}

    label_ok = bool(cfg.label_col and cfg.label_col in df.columns)
    per_group: Dict[str, Any] = {}
    for group_col in [c for c in cfg.group_cols if c in df.columns]:
        counts = df[group_col].value_counts(dropna=False)
        shares = counts / max(1, counts.sum())
        stats: Dict[str, Any] = {
            "num_groups": int(len(counts)),
            "min_group_share": float(shares.min()) if len(shares) else None,
            "max_group_share": float(shares.max()) if len(shares) else None,
            "representation_share_top10": shares.sort_values(ascending=False).head(10).to_dict(),
        }
        missingness_disparity: Dict[str, float] = {}
        for c in df.columns:
            missing_by_group = df.groupby(group_col)[c].apply(lambda series: series.isna().mean())
            if missing_by_group.shape[0] >= 2:
                missingness_disparity[c] = float(missing_by_group.max() - missing_by_group.min())
        stats["missingness_disparity_top10"] = dict(
            sorted(missingness_disparity.items(), key=lambda kv: kv[1], reverse=True)[:10]
        )
        if label_ok:
            stats["label_missingness_by_group"] = (
                df.groupby(group_col)[cfg.label_col].apply(lambda series: series.isna().mean()).to_dict()
            )
            y = df[cfg.label_col]
            if y.dropna().nunique() == 2:
                tmp = df.copy()
                tmp["_y"], _ = pd.factorize(tmp[cfg.label_col])
                positive_rate = tmp[tmp[cfg.label_col].notna()].groupby(group_col)["_y"].mean()
                if len(positive_rate) >= 2:
                    stats["positive_rate_by_group"] = positive_rate.to_dict()
                    stats["positive_rate_disparity"] = float(positive_rate.max() - positive_rate.min())
        per_group[group_col] = stats
    return {"group_checks": per_group}


def assess_tabular_security(
    df: pd.DataFrame,
    cfg: TabularAssessConfig,
    dataset_bytes: bytes,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    text_cols = categorical_cols(df, exclude=[c for c in [cfg.label_col, cfg.split_col, cfg.time_col] if c])
    text_cols = text_cols[: cfg.pii_max_text_cols]
    hits: Dict[str, Dict[str, float]] = {}
    for c in text_cols:
        series = df[c].dropna().astype("string")
        if series.empty:
            continue
        sample_size = min(len(series), cfg.pii_max_rows)
        sample = (
            series.sample(n=sample_size, random_state=cfg.random_state)
            if len(series) > sample_size
            else series
        )
        column_hits: Dict[str, float] = {}
        for name, pattern in PII_PATTERNS.items():
            hit_rate = float(sample.str.contains(pattern, regex=True).mean())
            if hit_rate >= cfg.thresholds.pii_hit_rate_threshold:
                column_hits[name] = hit_rate
        if column_hits:
            hits[c] = column_hits

    out["confidentiality_pii_heuristics"] = {
        "note": "Heuristic scan. Validate with domain and legal review.",
        "threshold_hit_rate": float(cfg.thresholds.pii_hit_rate_threshold),
        "rows_sampled_per_col_max": int(cfg.pii_max_rows),
        "text_cols_scanned_max": int(cfg.pii_max_text_cols),
        "columns_with_hits": hits,
    }
    out["integrity"] = {"sha256": sha256_bytes(dataset_bytes)}
    out["availability_asset_checks"] = {"byte_size": int(len(dataset_bytes))}
    return out


def assess_tabular_all(
    df: pd.DataFrame,
    cfg: TabularAssessConfig,
    dataset_bytes: bytes,
) -> Dict[str, Any]:
    if cfg.mode == "Full Scan":
        cfg.pii_max_rows = max(cfg.pii_max_rows, 5000)
        cfg.pii_max_text_cols = max(cfg.pii_max_text_cols, 25)
        cfg.rare_max_cat_cols = max(cfg.rare_max_cat_cols, 100)
        cfg.drift_max_num_cols = max(cfg.drift_max_num_cols, 100)

    return {
        "quality": assess_tabular_quality(df, cfg),
        "reliability": assess_tabular_reliability(df, cfg),
        "robustness": assess_tabular_robustness(df, cfg),
        "fairness": assess_tabular_fairness(df, cfg),
        "security": assess_tabular_security(df, cfg, dataset_bytes),
        "notes": {
            "sklearn_available": SKLEARN_OK,
            "mode": cfg.mode,
            "thresholds": {
                "drift_ks_threshold": cfg.thresholds.drift_ks_threshold,
                "pii_hit_rate_threshold": cfg.thresholds.pii_hit_rate_threshold,
            },
        },
    }


def tabular_verdict(report: Dict[str, Any], cfg: TabularAssessConfig) -> Tuple[str, str, List[str]]:
    reasons: List[str] = []
    missingness = report["quality"].get("missingness", {}).get("overall_missing_rate")
    duplicate_rate = report["quality"].get("duplicates", {}).get("exact_duplicate_row_rate")
    pii = report["security"]["confidentiality_pii_heuristics"]["columns_with_hits"]
    leakage = report["quality"].get("split_leakage", {}).get("row_hash_cross_split_rate")
    drift = report["reliability"].get("numeric_drift_ks_first_last", {}).get("top_10_ks", {})
    group_checks = report.get("fairness", {}).get("group_checks", {})
    disparities = [
        float(stats["positive_rate_disparity"])
        for stats in group_checks.values()
        if isinstance(stats, dict) and stats.get("positive_rate_disparity") is not None
    ]
    if missingness is not None and float(missingness) > 0.05:
        reasons.append("Missingness exceeds 5%; inspect completeness before training.")
    if duplicate_rate is not None and float(duplicate_rate) > 0.01:
        reasons.append("Duplicate rows exceed 1%; deduplicate before using this dataset.")
    if pii:
        reasons.append("PII-like patterns detected; review confidentiality and legal basis.")
    if leakage is not None and float(leakage) > 0:
        reasons.append("Potential split leakage; identical rows appear across splits.")
    if any(v is not None and float(v) > cfg.thresholds.drift_ks_threshold for v in drift.values()):
        reasons.append("Potential drift; at least one KS statistic exceeds the threshold.")
    if disparities and max(disparities) > 0.20:
        reasons.append("Group positive-rate disparity exceeds 0.20; audit fairness risk.")
    if reasons:
        return "Needs review", "warn", reasons
    return "Looks OK (evidence-based)", "ok", ["No major red flags under current checks."]


def build_tabular_recommendations(report: Dict[str, Any], cfg: TabularAssessConfig) -> List[str]:
    recs: List[str] = []
    quality = report.get("quality", {})
    missingness = quality.get("missingness", {}).get("overall_missing_rate")
    duplicate_rate = quality.get("duplicates", {}).get("exact_duplicate_row_rate")
    leakage = quality.get("split_leakage", {}).get("row_hash_cross_split_rate")
    drift = report.get("reliability", {}).get("numeric_drift_ks_first_last", {}).get("top_10_ks", {})
    pii = report.get("security", {}).get("confidentiality_pii_heuristics", {}).get("columns_with_hits", {})
    group_checks = report.get("fairness", {}).get("group_checks", {})

    if missingness is not None and float(missingness) > 0.05:
        recs.append("Missingness > 5%; inspect top missing columns and decide: drop, impute, or recollect.")
    if duplicate_rate is not None and float(duplicate_rate) > 0.01:
        recs.append("Duplicate rows > 1%; deduplicate and verify duplicates do not cross splits.")
    if leakage is not None and float(leakage) > 0:
        recs.append("Split leakage detected; re-split at entity level using ID columns.")
    if drift and any(float(v) > cfg.thresholds.drift_ks_threshold for v in drift.values() if v is not None):
        recs.append("Drift above threshold; compare distributions and consider retraining or recalibration.")
    if pii:
        recs.append("PII-like patterns found; mask or remove flagged columns and confirm legal basis.")
    disparities = [
        float(stats["positive_rate_disparity"])
        for stats in group_checks.values()
        if isinstance(stats, dict) and stats.get("positive_rate_disparity") is not None
    ]
    if disparities and max(disparities) > 0.20:
        recs.append("Fairness disparity detected; audit group coverage, labels, and collection process.")
    if not recs:
        recs.append("No major red flags. Keep the report as evidence and rerun after data updates.")
    return recs


def tabular_config_to_dict(cfg: TabularAssessConfig) -> Dict[str, Any]:
    payload = asdict(cfg)
    payload["thresholds"] = asdict(cfg.thresholds)
    return payload


def dataframe_to_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def analyze_tabular_dataframe(
    df: pd.DataFrame,
    *,
    config: Optional[TabularAssessConfig] = None,
    dataset_bytes: Optional[bytes] = None,
    dataset_name: str = "dataset.csv",
    preset: str = "Balanced (recommended)",
    mode: str = "Quick Scan",
    use_auto_columns: bool = True,
    weights: Optional[Dict[str, float]] = None,
    policy: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run a complete tabular audit without launching Streamlit."""

    working_df = df.copy()
    cfg = config or make_tabular_config(
        working_df,
        preset=preset,
        mode=mode,
        use_auto_columns=use_auto_columns,
    )
    if dataset_bytes is None:
        dataset_bytes = dataframe_to_bytes(working_df)

    report = assess_tabular_all(working_df, cfg, dataset_bytes)
    recommendations = build_tabular_recommendations(report, cfg)
    verdict, verdict_status, findings = tabular_verdict(report, cfg)
    score, grade, score_components = compute_health_score(
        report,
        cfg.thresholds.drift_ks_threshold,
        weights=weights or DEFAULT_WEIGHTS,
    )
    file_sha256 = report.get("security", {}).get("integrity", {}).get("sha256")
    audit_record = build_audit_record(
        analyzer="tabular",
        dataset_name=dataset_name,
        file_sha256=file_sha256,
        report=report,
        score=score,
        grade=grade,
        verdict=verdict,
        findings=findings,
        recommendations=recommendations,
        config=tabular_config_to_dict(cfg),
        score_components=score_components,
    )
    policy_result = evaluate_policy(audit_record, policy=policy)

    return to_json_safe(
        {
            "analyzer": "tabular",
            "dataset_name": dataset_name,
            "task_type": detect_tabular_task_type(working_df, cfg.label_col),
            "config": tabular_config_to_dict(cfg),
            "report": report,
            "score": score,
            "grade": grade,
            "score_components": score_components,
            "verdict": verdict,
            "verdict_status": verdict_status,
            "findings": findings,
            "recommendations": recommendations,
            "audit_record": audit_record,
            "policy_result": policy_result,
        }
    )


def read_tabular_file(path: Union[str, Path]) -> Tuple[pd.DataFrame, bytes]:
    path = Path(path)
    data = path.read_bytes()
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(BytesIO(data)), data
    if suffix == ".parquet":
        return pd.read_parquet(BytesIO(data)), data
    if suffix in {".xls", ".xlsx"}:
        return pd.read_excel(BytesIO(data)), data
    raise ValueError(f"Unsupported tabular file extension: {suffix}")


def analyze_tabular_file(
    path: Union[str, Path],
    *,
    config: Optional[TabularAssessConfig] = None,
    preset: str = "Balanced (recommended)",
    mode: str = "Quick Scan",
    use_auto_columns: bool = True,
    weights: Optional[Dict[str, float]] = None,
    policy: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    df, dataset_bytes = read_tabular_file(path)
    return analyze_tabular_dataframe(
        df,
        config=config,
        dataset_bytes=dataset_bytes,
        dataset_name=Path(path).name,
        preset=preset,
        mode=mode,
        use_auto_columns=use_auto_columns,
        weights=weights,
        policy=policy,
    )


__all__ = [
    "TABULAR_PRESETS",
    "TabularAssessConfig",
    "TabularThresholds",
    "analyze_tabular_dataframe",
    "analyze_tabular_file",
    "assess_tabular_all",
    "build_tabular_recommendations",
    "guess_tabular_columns",
    "make_tabular_config",
    "read_tabular_file",
]
