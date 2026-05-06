from pathlib import Path
import shutil

from audit_history import (
    build_audit_record,
    compare_reports,
    evaluate_policy,
    load_audit_runs,
    save_audit_record,
)


def _report(missingness=0.0, duplicate_rate=0.0, leakage=0.0, drift=0.0, pii=False):
    return {
        "quality": {
            "missingness": {"overall_missing_rate": missingness},
            "duplicates": {"exact_duplicate_row_rate": duplicate_rate},
            "split_leakage": {"row_hash_cross_split_rate": leakage},
        },
        "security": {
            "confidentiality_pii_heuristics": {
                "columns_with_hits": {"email": {}} if pii else {}
            }
        },
        "reliability": {"numeric_drift_ks_first_last": {"top_10_ks": {"sensor": drift}}},
    }


def _record(report, score=90, grade="A", dataset_name="dataset.csv"):
    return build_audit_record(
        analyzer="tabular",
        dataset_name=dataset_name,
        file_sha256="abcdef1234567890",
        report=report,
        score=score,
        grade=grade,
        verdict="Ready",
        findings=[],
        recommendations=[],
        config={"preset": "Balanced"},
    )


def test_policy_gate_passes_clean_record():
    result = evaluate_policy(_record(_report()))

    assert result["status"] == "PASS"
    assert result["violations"] == []


def test_policy_gate_fails_risky_record():
    result = evaluate_policy(
        _record(
            _report(missingness=0.20, duplicate_rate=0.05, leakage=0.01, drift=0.60, pii=True),
            score=40,
            grade="F",
        )
    )

    failed_names = {check["name"] for check in result["violations"]}
    assert result["status"] == "FAIL"
    assert "Health score" in failed_names
    assert "PII flags" in failed_names


def test_save_and_load_audit_records_round_trip():
    tmp = Path.cwd() / ".audit_history_test_runs"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir()
    try:
        path = save_audit_record(_record(_report()), tmp)
        runs = load_audit_runs(tmp)
    finally:
        shutil.rmtree(tmp)

    assert path.name.endswith(".json")
    assert len(runs) == 1
    assert runs[0]["dataset_name"] == "dataset.csv"


def test_compare_reports_marks_lower_risk_metrics_as_improved():
    before = _record(_report(missingness=0.10, duplicate_rate=0.05, leakage=0.01), score=60, grade="D")
    after = _record(_report(missingness=0.02, duplicate_rate=0.00, leakage=0.00), score=90, grade="A")

    comparison = compare_reports(before, after)
    directions = {row["metric"]: row["direction"] for row in comparison["metric_deltas"]}

    assert comparison["score_delta"] == 30
    assert directions["Missingness"] == "improved"
    assert directions["Duplicate rate"] == "improved"
    assert directions["Split leakage"] == "improved"


if __name__ == "__main__":
    test_policy_gate_passes_clean_record()
    test_policy_gate_fails_risky_record()
    test_save_and_load_audit_records_round_trip()
    test_compare_reports_marks_lower_risk_metrics_as_improved()
