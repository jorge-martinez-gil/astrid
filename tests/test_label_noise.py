import numpy as np
import pandas as pd

from astrid_label_noise import assess_label_noise


def test_label_noise_assessment_finds_confidently_flipped_labels():
    rng = np.random.RandomState(12)
    negative = rng.normal(loc=-3.0, scale=0.45, size=(80, 2))
    positive = rng.normal(loc=3.0, scale=0.45, size=(80, 2))
    features = pd.DataFrame(
        np.vstack([negative, positive]),
        columns=["sensor_a", "sensor_b"],
    )
    labels = pd.Series(["negative"] * 80 + ["positive"] * 80, name="target")
    flipped = [4, 11, 23, 92, 107, 135]
    labels.loc[flipped] = labels.loc[flipped].map(
        {"negative": "positive", "positive": "negative"}
    )

    result = assess_label_noise(
        features,
        labels,
        sample_ids=pd.Series([f"sample-{i}" for i in range(len(features))]),
        confidence_threshold=0.80,
    )

    candidate_rows = {item["row_index"] for item in result["top_suspected_samples"]}
    assert result["status"] == "ok"
    assert result["suspected_label_noise_rate"] >= 0.03
    assert len(candidate_rows.intersection(flipped)) >= 5
    assert result["out_of_fold_accuracy"] > 0.90


def test_label_noise_assessment_skips_unsupported_high_cardinality_labels():
    features = pd.DataFrame({"value": np.arange(60)})
    labels = pd.Series([f"label-{i}" for i in range(60)])

    result = assess_label_noise(features, labels)

    assert result["status"] == "skipped"
    assert result["suspected_label_noise_rate"] is None
    assert "at most" in result["note"]
