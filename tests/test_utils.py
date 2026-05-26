import math
import sys
import types

import numpy as np
import pandas as pd

sys.modules.setdefault("streamlit", types.ModuleType("streamlit"))

from utils import build_html_report, compute_health_score, to_json_safe


def test_compute_health_score_normalizes_custom_weights():
    report = {
        "quality": {
            "missingness": {"overall_missing_rate": 0.0},
            "duplicates": {"exact_duplicate_row_rate": 0.0},
            "split_leakage": {"row_hash_cross_split_rate": 0.0},
        },
        "security": {"confidentiality_pii_heuristics": {"columns_with_hits": {}}},
        "reliability": {"numeric_drift_ks_first_last": {"top_10_ks": {"sensor": 0.0}}},
        "robustness": {"row_anomaly_score_mad": {"p99": 0.0}},
        "fairness": {"group_checks": {"site": {"positive_rate_disparity": 0.0}}},
    }

    score, grade, components = compute_health_score(
        report,
        drift_threshold=0.30,
        weights={
            "quality": 70,
            "security": 50,
            "reliability": 40,
            "robustness": 20,
            "fairness": 20,
        },
    )

    assert score == 100
    assert grade == "A"
    assert math.isclose(sum(components.values()), 100.0)


def test_compute_health_score_penalizes_key_findings():
    report = {
        "quality": {
            "missingness": {"overall_missing_rate": 0.15},
            "duplicates": {"exact_duplicate_row_rate": 0.05},
            "split_leakage": {"row_hash_cross_split_rate": 0.01},
        },
        "security": {
            "confidentiality_pii_heuristics": {
                "columns_with_hits": {"email": {"hit_rate": 0.5}}
            }
        },
        "reliability": {"numeric_drift_ks_first_last": {"top_10_ks": {"sensor": 0.60}}},
        "robustness": {"row_anomaly_score_mad": {"p99": 30.0}},
        "fairness": {"group_checks": {"site": {"positive_rate_disparity": 0.50}}},
    }

    score, grade, _ = compute_health_score(report, drift_threshold=0.30)

    assert score < 25
    assert grade == "F"


def test_to_json_safe_converts_common_numpy_and_pandas_values():
    payload = {
        "int": np.int64(7),
        "float": np.float64(1.5),
        "bool": np.bool_(True),
        "array": np.array([1, 2]),
        "timestamp": pd.Timestamp("2026-05-06"),
        "missing": pd.NA,
    }

    converted = to_json_safe(payload)

    assert converted == {
        "int": 7,
        "float": 1.5,
        "bool": True,
        "array": [1, 2],
        "timestamp": "2026-05-06 00:00:00",
        "missing": None,
    }


def test_build_html_report_supports_image_reports():
    df = pd.DataFrame(
        {
            "path_in_zip": ["train/a.jpg", "train/b.jpg"],
            "open_ok": [True, True],
        }
    )
    report = {
        "quality": {
            "readability": {"readability_rate": 1.0, "corrupt_rate": 0.0},
            "missingness": {
                "overall_missing_rate": 0.0,
                "top_10_columns_missing_rate": {},
            },
            "duplicates": {"exact_duplicate_rate": 0.0},
            "low_resolution": {"low_res_rate": 0.0},
            "metadata_completeness": {"metadata_completeness": 1.0},
        },
        "reliability": {
            "feature_drift_ks_first_last": {"top_10_ks": {"brightness": 0.10}}
        },
        "security": {
            "confidentiality_pii_heuristics": {"columns_with_hits": {}},
        },
        "transparency": {"dataset_identity": {"total_images": 2}},
    }

    html = build_html_report(
        df=df,
        report=report,
        cfg_dict={"mode": "Quick Scan", "preset": "Balanced", "thresholds": {"drift_ks_threshold": 0.3}},
        file_name="images.zip",
        file_bytes=b"fake zip bytes",
        verdict="Looks OK",
        reasons=["No major red flags."],
        recs=["Archive the report."],
        score=95,
        grade="A",
    )

    assert "Image Quality Signals" in html
    assert "images.zip" in html
    assert "SHA-256" in html


if __name__ == "__main__":
    test_compute_health_score_normalizes_custom_weights()
    test_compute_health_score_penalizes_key_findings()
    test_to_json_safe_converts_common_numpy_and_pandas_values()
    test_build_html_report_supports_image_reports()
