import pandas as pd

from astrid_core import (
    TABULAR_PRESETS,
    TabularAssessConfig,
    analyze_tabular_dataframe,
    dataframe_to_bytes,
)


def _small_dataset():
    rows = []
    for i in range(80):
        rows.append(
            {
                "sample_id": f"sample_{i:03d}",
                "timestamp": f"2025-01-{1 + (i // 24):02d}",
                "split": "train" if i < 60 else "test",
                "site": "plant_a" if i % 2 == 0 else "plant_b",
                "sensor_temp": 50 + i * 0.1,
                "sensor_pressure": 100 - i * 0.05,
                "operator_note": "normal",
                "target": i % 2,
            }
        )
    df = pd.DataFrame(rows)
    leaked = df.iloc[[0]].copy()
    leaked["split"] = "test"
    df = pd.concat([df, leaked], ignore_index=True)
    df.loc[1, "operator_note"] = "contact person@example.com"
    return df


def test_headless_tabular_analyzer_flags_pii_and_split_leakage():
    df = _small_dataset()
    cfg = TabularAssessConfig(
        label_col="target",
        split_col="split",
        time_col="timestamp",
        group_cols=["site"],
        id_cols=["sample_id"],
        thresholds=TABULAR_PRESETS["Balanced (recommended)"],
    )

    result = analyze_tabular_dataframe(
        df,
        config=cfg,
        dataset_bytes=dataframe_to_bytes(df),
        dataset_name="small.csv",
        use_auto_columns=False,
    )

    pii_hits = result["report"]["security"]["confidentiality_pii_heuristics"]["columns_with_hits"]
    leakage = result["report"]["quality"]["split_leakage"]["row_hash_cross_split_rate"]

    assert 0 <= result["score"] <= 100
    assert pii_hits
    assert leakage > 0
    assert result["policy_result"]["status"] == "FAIL"
    assert any("PII" in rec for rec in result["recommendations"])
    assert any("leakage" in rec.lower() for rec in result["recommendations"])


def test_headless_tabular_analyzer_scores_clean_dataset_at_100():
    df = pd.DataFrame(
        {
            "sample_id": [f"sample_{i:03d}" for i in range(100)],
            "sensor_a": list(range(100)),
            "sensor_b": [100 - i for i in range(100)],
            "status": ["normal"] * 100,
        }
    )

    result = analyze_tabular_dataframe(
        df,
        dataset_bytes=dataframe_to_bytes(df),
        dataset_name="clean.csv",
        use_auto_columns=False,
    )

    assert result["score"] == 100
    assert result["grade"] == "A"
    assert result["score_components"] == {
        "quality": 35.0,
        "security": 25.0,
        "reliability": 20.0,
        "robustness": 10.0,
        "fairness": 10.0,
    }


if __name__ == "__main__":
    test_headless_tabular_analyzer_flags_pii_and_split_leakage()
    test_headless_tabular_analyzer_scores_clean_dataset_at_100()
