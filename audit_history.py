"""Local audit history, comparison, and policy-gate helpers."""
from __future__ import annotations

import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

APP_VERSION = "1.0"
HISTORY_DIR = Path(__file__).resolve().parent / "audit_runs"

DEFAULT_POLICY: Dict[str, Any] = {
    "min_health_score": 80,
    "max_missingness": 0.05,
    "max_duplicate_rate": 0.01,
    "allow_pii": False,
    "max_split_leakage": 0.0,
    "max_drift_ks": 0.30,
    "max_positive_rate_disparity": 0.20,
}


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
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


def _slug(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip())
    return value.strip("-")[:80] or "dataset"


def _get_path(payload: Dict[str, Any], path: List[str]) -> Any:
    cur: Any = payload
    for part in path:
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(value):
        return None
    return value


def _first_present(*values: Optional[float]) -> Optional[float]:
    for value in values:
        if value is not None:
            return value
    return None


def _metric_snapshot(report: Dict[str, Any]) -> Dict[str, Optional[float]]:
    drift_values = _get_path(report, ["reliability", "numeric_drift_ks_first_last", "top_10_ks"])
    if isinstance(drift_values, dict):
        max_drift = max(
            (_as_float(v) for v in drift_values.values() if _as_float(v) is not None),
            default=None,
        )
    else:
        max_drift = None

    pii_hits = _get_path(report, ["security", "confidentiality_pii_heuristics", "columns_with_hits"])
    pii_count = float(len(pii_hits)) if isinstance(pii_hits, dict) else None
    fairness_checks = _get_path(report, ["fairness", "group_checks"])
    fairness_values: List[float] = []
    if isinstance(fairness_checks, dict):
        for stats in fairness_checks.values():
            if not isinstance(stats, dict):
                continue
            for key in ("positive_rate_disparity", "max_label_parity_gap"):
                value = _as_float(stats.get(key))
                if value is not None:
                    fairness_values.append(value)
    max_positive_rate_disparity = max(fairness_values, default=None)

    return {
        "missingness": _as_float(
            _get_path(report, ["quality", "missingness", "overall_missing_rate"])
        ),
        "duplicate_rate": _first_present(
            _as_float(_get_path(report, ["quality", "duplicates", "exact_duplicate_row_rate"])),
            _as_float(_get_path(report, ["quality", "duplicates", "exact_duplicate_rate"])),
            _as_float(_get_path(report, ["quality", "exact_duplicate_row_rate"])),
        ),
        "split_leakage": _first_present(
            _as_float(_get_path(report, ["quality", "split_leakage", "row_hash_cross_split_rate"])),
            _as_float(_get_path(report, ["quality", "cross_split_leakage", "cross_split_leakage_rate"])),
        ),
        "max_drift_ks": max_drift,
        "max_positive_rate_disparity": max_positive_rate_disparity,
        "pii_flag_count": pii_count,
        "corrupt_rate": _as_float(_get_path(report, ["quality", "readability", "corrupt_rate"])),
    }


def build_audit_record(
    *,
    analyzer: str,
    dataset_name: str,
    file_sha256: Optional[str],
    report: Dict[str, Any],
    score: int,
    grade: str,
    verdict: str,
    findings: List[str],
    recommendations: List[str],
    config: Optional[Dict[str, Any]] = None,
    score_components: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    created = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    sha_part = (file_sha256 or "no-sha")[:12]
    run_id = f"{created.replace(':', '').replace('+', 'Z')}-{_slug(analyzer)}-{sha_part}"
    return _json_safe(
        {
            "schema_version": 1,
            "app_version": APP_VERSION,
            "run_id": run_id,
            "created_at_utc": created,
            "analyzer": analyzer,
            "dataset_name": dataset_name,
            "file_sha256": file_sha256,
            "score": int(score),
            "grade": str(grade),
            "verdict": str(verdict),
            "findings": findings,
            "recommendations": recommendations,
            "config": config or {},
            "score_components": score_components or {},
            "metrics": _metric_snapshot(report),
            "report": report,
        }
    )


def save_audit_record(record: Dict[str, Any], history_dir: Optional[Path] = None) -> Path:
    target_dir = Path(history_dir or HISTORY_DIR)
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / f"{_slug(str(record.get('run_id', 'audit-run')))}.json"
    path.write_text(json.dumps(_json_safe(record), indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_audit_runs(history_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
    target_dir = Path(history_dir or HISTORY_DIR)
    if not target_dir.exists():
        return []
    runs: List[Dict[str, Any]] = []
    for path in sorted(target_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["_path"] = str(path)
            runs.append(payload)
        except (OSError, json.JSONDecodeError):
            continue
    runs.sort(key=lambda r: str(r.get("created_at_utc", "")), reverse=True)
    return runs


def summarize_run(record: Dict[str, Any]) -> Dict[str, Any]:
    metrics = record.get("metrics", {}) if isinstance(record.get("metrics"), dict) else {}
    return {
        "created_at_utc": record.get("created_at_utc"),
        "analyzer": record.get("analyzer"),
        "dataset": record.get("dataset_name"),
        "score": record.get("score"),
        "grade": record.get("grade"),
        "verdict": record.get("verdict"),
        "missingness": metrics.get("missingness"),
        "duplicate_rate": metrics.get("duplicate_rate"),
        "split_leakage": metrics.get("split_leakage"),
        "max_drift_ks": metrics.get("max_drift_ks"),
        "max_positive_rate_disparity": metrics.get("max_positive_rate_disparity"),
        "pii_flag_count": metrics.get("pii_flag_count"),
        "run_id": record.get("run_id"),
    }


def evaluate_policy(
    report_or_record: Dict[str, Any],
    *,
    score: Optional[int] = None,
    policy: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    policy = {**DEFAULT_POLICY, **(policy or {})}
    report = report_or_record.get("report", report_or_record)
    metrics = (
        report_or_record.get("metrics")
        if isinstance(report_or_record.get("metrics"), dict)
        else _metric_snapshot(report)
    )
    score_value = score if score is not None else report_or_record.get("score")

    checks: List[Dict[str, Any]] = []

    def add_check(name: str, value: Any, limit: Any, passed: Optional[bool], unit: str = "") -> None:
        checks.append(
            {
                "name": name,
                "value": value,
                "limit": limit,
                "unit": unit,
                "status": "not_applicable" if passed is None else ("pass" if passed else "fail"),
            }
        )

    add_check(
        "Health score",
        score_value,
        f">= {policy['min_health_score']}",
        None if score_value is None else int(score_value) >= int(policy["min_health_score"]),
    )
    add_check(
        "Missingness",
        metrics.get("missingness"),
        f"<= {policy['max_missingness']}",
        None
        if metrics.get("missingness") is None
        else float(metrics["missingness"]) <= float(policy["max_missingness"]),
    )
    add_check(
        "Duplicate rate",
        metrics.get("duplicate_rate"),
        f"<= {policy['max_duplicate_rate']}",
        None
        if metrics.get("duplicate_rate") is None
        else float(metrics["duplicate_rate"]) <= float(policy["max_duplicate_rate"]),
    )
    add_check(
        "Split leakage",
        metrics.get("split_leakage"),
        f"<= {policy['max_split_leakage']}",
        None
        if metrics.get("split_leakage") is None
        else float(metrics["split_leakage"]) <= float(policy["max_split_leakage"]),
    )
    add_check(
        "Max drift KS",
        metrics.get("max_drift_ks"),
        f"<= {policy['max_drift_ks']}",
        None
        if metrics.get("max_drift_ks") is None
        else float(metrics["max_drift_ks"]) <= float(policy["max_drift_ks"]),
    )
    add_check(
        "Positive-rate disparity",
        metrics.get("max_positive_rate_disparity"),
        f"<= {policy['max_positive_rate_disparity']}",
        None
        if metrics.get("max_positive_rate_disparity") is None
        else float(metrics["max_positive_rate_disparity"])
        <= float(policy["max_positive_rate_disparity"]),
    )
    pii_count = metrics.get("pii_flag_count")
    add_check(
        "PII flags",
        pii_count,
        "allowed" if policy["allow_pii"] else "0",
        None if pii_count is None else (bool(policy["allow_pii"]) or int(pii_count) == 0),
    )

    violations = [c for c in checks if c["status"] == "fail"]
    return {
        "status": "PASS" if not violations else "FAIL",
        "policy": policy,
        "checks": checks,
        "violations": violations,
    }


def compare_reports(base: Dict[str, Any], candidate: Dict[str, Any]) -> Dict[str, Any]:
    base_metrics = base.get("metrics") or _metric_snapshot(base.get("report", base))
    candidate_metrics = candidate.get("metrics") or _metric_snapshot(candidate.get("report", candidate))
    metric_rows = []
    labels = {
        "missingness": "Missingness",
        "duplicate_rate": "Duplicate rate",
        "split_leakage": "Split leakage",
        "max_drift_ks": "Max drift KS",
        "max_positive_rate_disparity": "Positive-rate disparity",
        "pii_flag_count": "PII flag count",
        "corrupt_rate": "Corrupt image rate",
    }
    lower_is_better = set(labels)
    for key, label in labels.items():
        before = base_metrics.get(key)
        after = candidate_metrics.get(key)
        delta = None if before is None or after is None else after - before
        if delta is None:
            direction = "not_applicable"
        elif abs(delta) < 1e-12:
            direction = "unchanged"
        elif (delta < 0 and key in lower_is_better) or (delta > 0 and key not in lower_is_better):
            direction = "improved"
        else:
            direction = "worse"
        metric_rows.append(
            {"metric": label, "before": before, "after": after, "delta": delta, "direction": direction}
        )

    before_findings = set(base.get("findings", []))
    after_findings = set(candidate.get("findings", []))
    before_recs = set(base.get("recommendations", []))
    after_recs = set(candidate.get("recommendations", []))
    before_score = _as_float(base.get("score"))
    after_score = _as_float(candidate.get("score"))

    return {
        "score_delta": None if before_score is None or after_score is None else after_score - before_score,
        "grade_before": base.get("grade"),
        "grade_after": candidate.get("grade"),
        "metric_deltas": metric_rows,
        "new_findings": sorted(after_findings - before_findings),
        "resolved_findings": sorted(before_findings - after_findings),
        "new_recommendations": sorted(after_recs - before_recs),
        "resolved_recommendations": sorted(before_recs - after_recs),
    }
