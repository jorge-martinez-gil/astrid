"""Controlled tabular fault-injection benchmark for ASTRID.

The runner creates a synthetic industrial-style dataset, injects one known
failure mode at a time, audits the corrupted dataset with the headless ASTRID
API, and writes paper-ready summary tables.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    PLOT_OK = True
except Exception:  # pragma: no cover - optional plotting dependency
    plt = None
    PLOT_OK = False

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from astrid_core import (  # noqa: E402
    TabularAssessConfig,
    TABULAR_PRESETS,
    analyze_tabular_dataframe,
    dataframe_to_bytes,
)
from utils import to_json_safe  # noqa: E402


DEFAULT_FAULTS = ["missingness", "duplicates", "split_leakage", "drift", "pii", "fairness"]
DEFAULT_SEVERITIES = [0.0, 0.01, 0.05, 0.10, 0.20, 0.40]
NUMERIC_COLUMNS = ["sensor_temp", "sensor_pressure", "vibration_rms"]


def make_clean_dataset(n_rows: int = 4000, seed: int = 7) -> pd.DataFrame:
    """Create a deterministic industrial-style binary classification dataset."""

    rng = np.random.default_rng(seed)
    slice_ids = np.floor(np.arange(n_rows) * 4 / max(1, n_rows)).astype(int)
    block_size = max(1, int(np.ceil(n_rows / 4)))
    timestamps = [
        pd.Timestamp("2025-01-01")
        + pd.DateOffset(months=int(slice_id))
        + pd.Timedelta(hours=int(i % block_size))
        for i, slice_id in enumerate(slice_ids)
    ]
    site = rng.choice(["plant_a", "plant_b", "plant_c"], size=n_rows, p=[0.34, 0.33, 0.33])
    shift = pd.Series(site).map({"plant_a": -2.0, "plant_b": 0.0, "plant_c": 2.0}).to_numpy()

    sensor_temp = rng.normal(54.0 + shift, 7.5, size=n_rows)
    sensor_pressure = rng.normal(100.0 - shift, 12.0, size=n_rows)
    vibration_rms = rng.gamma(shape=2.0, scale=0.7, size=n_rows)
    humidity = rng.normal(45.0, 9.0, size=n_rows)
    split = rng.choice(["train", "validation", "test"], size=n_rows, p=[0.70, 0.15, 0.15])

    logit = -2.1 + 0.025 * sensor_temp + 0.008 * sensor_pressure
    probability = 1.0 / (1.0 + np.exp(-logit))
    target = rng.binomial(1, np.clip(probability, 0.02, 0.98), size=n_rows)

    return pd.DataFrame(
        {
            "sample_id": [f"sample_{i:06d}" for i in range(n_rows)],
            "timestamp": pd.Series(timestamps).astype(str),
            "split": split,
            "site": site,
            "sensor_temp": sensor_temp,
            "sensor_pressure": sensor_pressure,
            "vibration_rms": vibration_rms,
            "humidity": humidity,
            "operator_note": ["normal"] * n_rows,
            "target": target,
        }
    )


def _nonzero_count(n_rows: int, severity: float) -> int:
    if severity <= 0:
        return 0
    return max(1, int(round(n_rows * severity)))


def inject_missingness(df: pd.DataFrame, severity: float, rng: np.random.Generator) -> pd.DataFrame:
    out = df.copy()
    if severity <= 0:
        return out
    n_cells = int(round(len(out) * len(NUMERIC_COLUMNS) * severity))
    if n_cells <= 0:
        return out
    row_idx = rng.integers(0, len(out), size=n_cells)
    col_idx = rng.integers(0, len(NUMERIC_COLUMNS), size=n_cells)
    for row, col in zip(row_idx, col_idx):
        out.at[int(row), NUMERIC_COLUMNS[int(col)]] = np.nan
    return out


def inject_duplicates(df: pd.DataFrame, severity: float, rng: np.random.Generator) -> pd.DataFrame:
    out = df.copy()
    n_dup = _nonzero_count(len(out), severity)
    if n_dup == 0:
        return out
    sampled = out.sample(n=n_dup, replace=False, random_state=int(rng.integers(0, 1_000_000)))
    return pd.concat([out, sampled], ignore_index=True)


def inject_split_leakage(df: pd.DataFrame, severity: float, rng: np.random.Generator) -> pd.DataFrame:
    out = df.copy()
    n_leaked = _nonzero_count(len(out), severity)
    if n_leaked == 0:
        return out
    leaked = out.sample(n=n_leaked, replace=False, random_state=int(rng.integers(0, 1_000_000))).copy()
    leaked["split"] = np.where(leaked["split"].eq("test"), "train", "test")
    return pd.concat([out, leaked], ignore_index=True)


def inject_drift(df: pd.DataFrame, severity: float, rng: np.random.Generator) -> pd.DataFrame:
    out = df.copy()
    if severity <= 0:
        return out
    timestamps = pd.to_datetime(out["timestamp"], errors="coerce")
    cutoff = timestamps.quantile(0.75)
    mask = timestamps >= cutoff
    out.loc[mask, "sensor_temp"] += severity * 35.0
    out.loc[mask, "sensor_pressure"] -= severity * 28.0
    out.loc[mask, "vibration_rms"] += rng.normal(severity * 1.8, max(0.01, severity), size=int(mask.sum()))
    return out


def inject_pii(df: pd.DataFrame, severity: float, rng: np.random.Generator) -> pd.DataFrame:
    out = df.copy()
    n_hits = _nonzero_count(len(out), severity)
    if n_hits == 0:
        return out
    idx = rng.choice(out.index.to_numpy(), size=n_hits, replace=False)
    for i, row_idx in enumerate(idx):
        out.at[row_idx, "operator_note"] = f"contact operator_{i:05d}@example.com"
    return out


def inject_fairness(df: pd.DataFrame, severity: float, rng: np.random.Generator) -> pd.DataFrame:
    out = df.copy()
    if severity <= 0:
        return out
    target = out["target"].to_numpy(copy=True)
    advantaged = out["site"].eq("plant_c").to_numpy()
    disadvantaged = ~advantaged
    target[advantaged] = rng.binomial(1, min(0.95, 0.50 + severity), size=int(advantaged.sum()))
    target[disadvantaged] = rng.binomial(1, max(0.05, 0.50 - severity), size=int(disadvantaged.sum()))
    out["target"] = target
    return out


INJECTORS = {
    "missingness": inject_missingness,
    "duplicates": inject_duplicates,
    "split_leakage": inject_split_leakage,
    "drift": inject_drift,
    "pii": inject_pii,
    "fairness": inject_fairness,
}


def inject_fault(df: pd.DataFrame, fault_type: str, severity: float, seed: int) -> pd.DataFrame:
    if fault_type not in INJECTORS:
        raise ValueError(f"Unknown fault type: {fault_type}")
    rng = np.random.default_rng(seed)
    return INJECTORS[fault_type](df, severity, rng)


def _max_drift_ks(result: Dict[str, Any]) -> float:
    drift = (
        result.get("report", {})
        .get("reliability", {})
        .get("numeric_drift_ks_first_last", {})
        .get("top_10_ks", {})
    )
    values = [float(v) for v in drift.values() if v is not None]
    return max(values) if values else 0.0


def _max_positive_rate_disparity(result: Dict[str, Any]) -> float:
    group_checks = result.get("report", {}).get("fairness", {}).get("group_checks", {})
    values = [
        float(stats["positive_rate_disparity"])
        for stats in group_checks.values()
        if isinstance(stats, dict) and stats.get("positive_rate_disparity") is not None
    ]
    return max(values) if values else 0.0


def extract_primary_metric(result: Dict[str, Any], fault_type: str) -> Dict[str, Any]:
    report = result["report"]
    if fault_type == "missingness":
        value = report["quality"]["missingness"]["overall_missing_rate"]
        return {"metric": "overall_missing_rate", "value": value, "detected": value > 0.05}
    if fault_type == "duplicates":
        value = report["quality"]["duplicates"]["exact_duplicate_row_rate"]
        return {"metric": "exact_duplicate_row_rate", "value": value, "detected": value > 0.01}
    if fault_type == "split_leakage":
        value = report["quality"].get("split_leakage", {}).get("row_hash_cross_split_rate") or 0.0
        return {"metric": "row_hash_cross_split_rate", "value": value, "detected": value > 0.0}
    if fault_type == "drift":
        value = _max_drift_ks(result)
        threshold = result["config"]["thresholds"]["drift_ks_threshold"]
        return {"metric": "max_numeric_drift_ks", "value": value, "detected": value > threshold}
    if fault_type == "pii":
        hits = report["security"]["confidentiality_pii_heuristics"]["columns_with_hits"]
        value = len(hits)
        return {"metric": "pii_flagged_columns", "value": value, "detected": value > 0}
    if fault_type == "fairness":
        value = _max_positive_rate_disparity(result)
        return {"metric": "positive_rate_disparity", "value": value, "detected": value > 0.20}
    raise ValueError(f"Unknown fault type: {fault_type}")


def recommendation_hit(result: Dict[str, Any], fault_type: str) -> bool:
    text = " ".join(result.get("recommendations", [])).lower()
    keywords = {
        "missingness": ["missingness", "impute", "recollect"],
        "duplicates": ["duplicate", "deduplicate"],
        "split_leakage": ["leakage", "re-split"],
        "drift": ["drift", "retraining", "recalibration"],
        "pii": ["pii", "mask", "legal"],
        "fairness": ["fairness", "disparity", "group"],
    }
    return any(keyword in text for keyword in keywords[fault_type])


def run_fault_grid(
    *,
    n_rows: int,
    faults: Iterable[str],
    severities: Iterable[float],
    seed: int,
    preset: str,
    mode: str,
    save_reports: bool,
    reports_dir: Path,
) -> List[Dict[str, Any]]:
    clean = make_clean_dataset(n_rows=n_rows, seed=seed)
    rows: List[Dict[str, Any]] = []

    for fault_idx, fault_type in enumerate(faults):
        for severity_idx, severity in enumerate(severities):
            run_seed = seed + fault_idx * 10_000 + severity_idx
            corrupted = inject_fault(clean, fault_type, float(severity), run_seed)
            config = TabularAssessConfig(
                label_col="target",
                split_col="split",
                time_col="timestamp",
                group_cols=["site"],
                id_cols=["sample_id"],
                mode=mode,
                thresholds=TABULAR_PRESETS[preset],
                random_state=seed,
            )
            start = time.perf_counter()
            result = analyze_tabular_dataframe(
                corrupted,
                config=config,
                dataset_bytes=dataframe_to_bytes(corrupted),
                dataset_name=f"synthetic_seed{seed}_{fault_type}_{severity:.3f}.csv",
                preset=preset,
                mode=mode,
                use_auto_columns=False,
            )
            runtime_s = time.perf_counter() - start
            primary = extract_primary_metric(result, fault_type)
            row = {
                "seed": int(seed),
                "fault_type": fault_type,
                "severity": float(severity),
                "n_rows": int(len(corrupted)),
                "score": result["score"],
                "grade": result["grade"],
                "verdict": result["verdict"],
                "policy_status": result["policy_result"]["status"],
                "primary_metric": primary["metric"],
                "primary_metric_value": primary["value"],
                "detected": bool(primary["detected"]),
                "recommendation_hit": recommendation_hit(result, fault_type),
                "runtime_s": runtime_s,
            }
            rows.append(row)

            if save_reports:
                reports_dir.mkdir(parents=True, exist_ok=True)
                report_path = reports_dir / f"seed{seed}_{fault_type}_{severity:.3f}.json"
                report_path.write_text(
                    json.dumps(to_json_safe(result), indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )

    return rows


def run_multi_seed_fault_grid(
    *,
    n_rows: int,
    faults: Iterable[str],
    severities: Iterable[float],
    seeds: Iterable[int],
    preset: str,
    mode: str,
    save_reports: bool,
    reports_dir: Path,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for seed in seeds:
        rows.extend(
            run_fault_grid(
                n_rows=n_rows,
                faults=faults,
                severities=severities,
                seed=int(seed),
                preset=preset,
                mode=mode,
                save_reports=save_reports,
                reports_dir=reports_dir,
            )
        )
    return rows


def parse_float_list(value: str) -> List[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def parse_int_list(value: str) -> List[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def aggregate_rows(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["policy_failed"] = work["policy_status"].eq("FAIL")
    agg = (
        work.groupby(["fault_type", "severity", "primary_metric"], as_index=False)
        .agg(
            score_mean=("score", "mean"),
            score_std=("score", "std"),
            primary_metric_value_mean=("primary_metric_value", "mean"),
            primary_metric_value_std=("primary_metric_value", "std"),
            detected_rate=("detected", "mean"),
            recommendation_rate=("recommendation_hit", "mean"),
            policy_fail_rate=("policy_failed", "mean"),
            runtime_s_mean=("runtime_s", "mean"),
        )
        .sort_values(["fault_type", "severity"])
    )
    return agg.fillna(0.0)


def _ordered_faults(values: Iterable[str]) -> List[str]:
    present = set(values)
    ordered = [fault for fault in DEFAULT_FAULTS if fault in present]
    ordered.extend(sorted(present - set(ordered)))
    return ordered


def _plot_score_lines(agg: pd.DataFrame, figures_dir: Path) -> str:
    assert plt is not None
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for fault in _ordered_faults(agg["fault_type"].unique()):
        subset = agg[agg["fault_type"] == fault].sort_values("severity")
        ax.plot(subset["severity"], subset["score_mean"], marker="o", linewidth=2, label=fault)
        if "score_std" in subset:
            lo = subset["score_mean"] - subset["score_std"]
            hi = subset["score_mean"] + subset["score_std"]
            ax.fill_between(subset["severity"], lo, hi, alpha=0.12)
    ax.set_title("ASTRID Health Score Under Injected Tabular Faults")
    ax.set_xlabel("Injected fault severity")
    ax.set_ylabel("Mean health score")
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    path = figures_dir / "score_vs_severity.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return str(path)


def _plot_primary_metric_lines(agg: pd.DataFrame, figures_dir: Path) -> str:
    assert plt is not None
    faults = _ordered_faults(agg["fault_type"].unique())
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.6), sharex=True)
    axes_flat = axes.ravel()
    for ax, fault in zip(axes_flat, faults):
        subset = agg[agg["fault_type"] == fault].sort_values("severity")
        metric = subset["primary_metric"].iloc[0]
        ax.plot(
            subset["severity"],
            subset["primary_metric_value_mean"],
            marker="o",
            linewidth=2,
            color="#1f77b4",
        )
        lo = subset["primary_metric_value_mean"] - subset["primary_metric_value_std"]
        hi = subset["primary_metric_value_mean"] + subset["primary_metric_value_std"]
        ax.fill_between(subset["severity"], lo, hi, alpha=0.12, color="#1f77b4")
        ax.set_title(fault)
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.25)
    for ax in axes_flat[len(faults) :]:
        ax.axis("off")
    fig.supxlabel("Injected fault severity")
    fig.suptitle("Primary Metric Response Under Controlled Fault Injection", y=1.02)
    fig.tight_layout()
    path = figures_dir / "primary_metric_vs_severity.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def _plot_heatmap(
    agg: pd.DataFrame,
    figures_dir: Path,
    *,
    value_col: str,
    title: str,
    file_name: str,
) -> str:
    assert plt is not None
    faults = _ordered_faults(agg["fault_type"].unique())
    severities = sorted(agg["severity"].unique())
    pivot = (
        agg.pivot(index="fault_type", columns="severity", values=value_col)
        .reindex(index=faults, columns=severities)
        .fillna(0.0)
    )
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    image = ax.imshow(pivot.to_numpy(dtype=float), vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(len(severities)), [f"{v:g}" for v in severities])
    ax.set_yticks(range(len(faults)), faults)
    ax.set_xlabel("Injected fault severity")
    ax.set_title(title)
    for i in range(len(faults)):
        for j in range(len(severities)):
            value = pivot.iloc[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", color="white", fontsize=8)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path = figures_dir / file_name
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return str(path)


def generate_figures(df: pd.DataFrame, out_dir: Path) -> Dict[str, Any]:
    if not PLOT_OK:
        return {"figures_error": "matplotlib is not available"}
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    agg = aggregate_rows(df)
    paths = {
        "score_vs_severity": _plot_score_lines(agg, figures_dir),
        "primary_metric_vs_severity": _plot_primary_metric_lines(agg, figures_dir),
        "detection_heatmap": _plot_heatmap(
            agg,
            figures_dir,
            value_col="detected_rate",
            title="Detection Rate by Fault Type and Severity",
            file_name="detection_heatmap.png",
        ),
        "recommendation_heatmap": _plot_heatmap(
            agg,
            figures_dir,
            value_col="recommendation_rate",
            title="Recommendation Coverage by Fault Type and Severity",
            file_name="recommendation_heatmap.png",
        ),
        "policy_gate_heatmap": _plot_heatmap(
            agg,
            figures_dir,
            value_col="policy_fail_rate",
            title="Policy Gate Failure Rate by Fault Type and Severity",
            file_name="policy_gate_heatmap.png",
        ),
    }
    return paths


def write_outputs(rows: List[Dict[str, Any]], out_dir: Path, metadata: Dict[str, Any]) -> Dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_df = pd.DataFrame(rows)
    aggregate_df = aggregate_rows(summary_df)
    summary_csv = out_dir / "tabular_fault_injection_summary.csv"
    aggregate_csv = out_dir / "tabular_fault_injection_aggregate.csv"
    results_json = out_dir / "tabular_fault_injection_results.json"
    summary_df.to_csv(summary_csv, index=False)
    aggregate_df.to_csv(aggregate_csv, index=False)
    figures = generate_figures(summary_df, out_dir)
    results_json.write_text(
        json.dumps(
            to_json_safe(
                {
                    "metadata": metadata,
                    "results": rows,
                    "aggregate": aggregate_df.to_dict(orient="records"),
                    "figures": figures,
                }
            ),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return {
        "summary_csv": str(summary_csv),
        "aggregate_csv": str(aggregate_csv),
        "results_json": str(results_json),
        **{key: str(value) for key, value in figures.items()},
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=4000, help="Rows in the clean synthetic dataset.")
    parser.add_argument("--seed", type=int, default=7, help="Base random seed.")
    parser.add_argument(
        "--faults",
        nargs="+",
        default=DEFAULT_FAULTS,
        choices=DEFAULT_FAULTS,
        help="Fault types to inject.",
    )
    parser.add_argument(
        "--severities",
        default=",".join(str(v) for v in DEFAULT_SEVERITIES),
        help="Comma-separated severity values.",
    )
    parser.add_argument(
        "--preset",
        default="Balanced (recommended)",
        choices=sorted(TABULAR_PRESETS),
        help="ASTRID threshold preset.",
    )
    parser.add_argument(
        "--mode",
        default="Quick Scan",
        choices=["Quick Scan", "Full Scan"],
        help="ASTRID scan mode.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "experiments" / "results" / "tabular_fault_injection",
        help="Directory for experiment outputs.",
    )
    parser.add_argument(
        "--save-reports",
        action="store_true",
        help="Also write full per-run ASTRID reports under out-dir/reports.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    severities = parse_float_list(args.severities)
    reports_dir = args.out_dir / "reports"
    rows = run_fault_grid(
        n_rows=args.rows,
        faults=args.faults,
        severities=severities,
        seed=args.seed,
        preset=args.preset,
        mode=args.mode,
        save_reports=args.save_reports,
        reports_dir=reports_dir,
    )
    metadata = {
        "experiment": "tabular_fault_injection",
        "rows_requested": args.rows,
        "seed": args.seed,
        "faults": args.faults,
        "severities": severities,
        "preset": args.preset,
        "mode": args.mode,
        "save_reports": bool(args.save_reports),
    }
    paths = write_outputs(rows, args.out_dir, metadata)
    print(json.dumps(paths, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
