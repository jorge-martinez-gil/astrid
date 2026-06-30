# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jorge Martinez-Gil and the ASTRID authors. See LICENSE.
"""Reusable, model-based label-noise assessment.

The assessor is intentionally conservative: it uses out-of-fold predictions
and only flags a sample when the model confidently prefers a different class.
These are review candidates, not proven annotation errors.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

try:
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

    SKLEARN_OK = True
except Exception:  # pragma: no cover - optional dependency guard
    SKLEARN_OK = False


def _json_scalar(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return str(value)
    if pd.isna(value):
        return None
    return value


def _empty_result(note: str, *, labeled_rows: int = 0) -> dict[str, Any]:
    return {
        "status": "skipped",
        "note": note,
        "labeled_rows": int(labeled_rows),
        "evaluated_rows": 0,
        "suspected_label_noise_rate": None,
        "suspected_label_noise_count": 0,
        "top_suspected_samples": [],
    }


def assess_label_noise(
    features: pd.DataFrame,
    labels: pd.Series,
    *,
    sample_ids: pd.Series | None = None,
    random_state: int = 7,
    confidence_threshold: float = 0.80,
    warning_threshold: float = 0.05,
    min_samples: int = 40,
    max_samples: int = 5000,
    max_classes: int = 50,
    max_categorical_cardinality: int = 100,
    top_n: int = 50,
) -> dict[str, Any]:
    """Estimate classification label noise using out-of-fold probabilities.

    A row is a suspected label issue when its out-of-fold predicted class
    differs from the observed class and the suggested class probability is at
    least ``confidence_threshold``. The estimate is therefore conservative and
    depends on how informative the supplied features are.
    """

    if not SKLEARN_OK:
        return _empty_result("scikit-learn is required for label-noise assessment.")
    if not isinstance(features, pd.DataFrame):
        features = pd.DataFrame(features)
    if not isinstance(labels, pd.Series):
        labels = pd.Series(labels, index=features.index)

    common_index = features.index.intersection(labels.index)
    x_all = features.loc[common_index].copy()
    y_all = labels.loc[common_index]
    valid_label = y_all.notna()
    x_all = x_all.loc[valid_label]
    y_all = y_all.loc[valid_label]
    labeled_rows = int(len(y_all))

    if labeled_rows < min_samples:
        return _empty_result(
            f"Need at least {min_samples} labeled rows; found {labeled_rows}.",
            labeled_rows=labeled_rows,
        )

    class_counts = y_all.value_counts(dropna=True)
    num_classes = int(len(class_counts))
    if num_classes < 2:
        return _empty_result("Need at least two observed classes.", labeled_rows=labeled_rows)
    if num_classes > max_classes:
        return _empty_result(
            f"Classification labels only, with at most {max_classes} classes; found {num_classes}.",
            labeled_rows=labeled_rows,
        )
    if int(class_counts.min()) < 2:
        return _empty_result(
            "Every class needs at least two labeled rows for out-of-fold assessment.",
            labeled_rows=labeled_rows,
        )

    numeric_cols = []
    categorical_cols = []
    dropped_high_cardinality = []
    for col in x_all.columns:
        series = x_all[col]
        unique = int(series.nunique(dropna=True))
        if unique <= 1:
            continue
        if pd.api.types.is_numeric_dtype(series.dtype) or pd.api.types.is_bool_dtype(series.dtype):
            numeric_cols.append(col)
        elif unique <= max_categorical_cardinality:
            categorical_cols.append(col)
        else:
            dropped_high_cardinality.append(str(col))

    feature_cols = numeric_cols + categorical_cols
    if not feature_cols:
        return _empty_result(
            "No usable non-constant features were available for label-noise assessment.",
            labeled_rows=labeled_rows,
        )

    x_all = x_all[feature_cols]
    if sample_ids is None:
        ids_all = pd.Series(x_all.index, index=x_all.index, dtype="object")
    else:
        ids_all = sample_ids.reindex(x_all.index)
        ids_all = ids_all.where(ids_all.notna(), pd.Series(x_all.index, index=x_all.index))

    sampled = False
    if len(x_all) > max_samples:
        positions = np.arange(len(x_all))
        try:
            selected, _ = train_test_split(
                positions,
                train_size=max_samples,
                random_state=random_state,
                stratify=y_all.to_numpy(),
            )
        except ValueError:
            rng = np.random.RandomState(random_state)
            selected = rng.choice(positions, size=max_samples, replace=False)
        selected = np.sort(selected)
        x_eval = x_all.iloc[selected].copy()
        y_eval = y_all.iloc[selected].copy()
        ids_eval = ids_all.iloc[selected].copy()
        sampled = True
    else:
        x_eval = x_all.copy()
        y_eval = y_all.copy()
        ids_eval = ids_all.copy()

    for col in categorical_cols:
        x_eval[col] = x_eval[col].astype("object").where(x_eval[col].notna(), np.nan)
    if numeric_cols:
        x_eval[numeric_cols] = x_eval[numeric_cols].replace([np.inf, -np.inf], np.nan)

    eval_counts = y_eval.value_counts(dropna=True)
    folds = min(5, int(eval_counts.min()))
    if folds < 2:
        return _empty_result(
            "The sampled data do not contain enough examples per class.",
            labeled_rows=labeled_rows,
        )

    transformers = []
    if numeric_cols:
        numeric_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("numeric", numeric_pipeline, numeric_cols))
    if categorical_cols:
        categorical_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "one_hot",
                    OneHotEncoder(handle_unknown="ignore", min_frequency=2),
                ),
            ]
        )
        transformers.append(("categorical", categorical_pipeline, categorical_cols))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    pipeline = Pipeline(
        [
            ("preprocess", preprocessor),
            ("classifier", LogisticRegression(max_iter=500, random_state=random_state)),
        ]
    )

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y_eval.astype(str))
    class_lookup = {
        str(value): _json_scalar(value) for value in y_eval.drop_duplicates().tolist()
    }
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    try:
        probabilities = cross_val_predict(
            pipeline,
            x_eval,
            y_encoded,
            cv=cv,
            method="predict_proba",
            n_jobs=1,
        )
    except Exception as exc:
        result = _empty_result(
            f"Label-noise model could not be fitted: {exc}",
            labeled_rows=labeled_rows,
        )
        result["features_used"] = [str(c) for c in feature_cols]
        return result

    row_positions = np.arange(len(y_encoded))
    predicted = probabilities.argmax(axis=1)
    suggested_confidence = probabilities[row_positions, predicted]
    assigned_probability = probabilities[row_positions, y_encoded]
    suspected = (predicted != y_encoded) & (suggested_confidence >= confidence_threshold)
    noise_score = suggested_confidence - assigned_probability

    # Confident-learning estimate of the label-noise rate (Northcutt et al.).
    # Unlike the conservative confident-disagreement count above, this uses
    # per-class self-confidence thresholds and counts rows that are confident
    # examples of a class other than their observed label. It is far more
    # monotonic in the true noise rate, because it does not collapse once many
    # labels are corrupted (a single global confidence cut does).
    num_label_classes = probabilities.shape[1]
    class_thresholds = np.array(
        [
            probabilities[y_encoded == j, j].mean() if np.any(y_encoded == j) else 1.0
            for j in range(num_label_classes)
        ]
    )
    above_threshold = probabilities >= class_thresholds[None, :]
    masked_probs = np.where(above_threshold, probabilities, -np.inf)
    confident_class = masked_probs.argmax(axis=1)
    has_confident_class = np.isfinite(masked_probs.max(axis=1))
    confident_other_class = has_confident_class & (confident_class != y_encoded)
    cl_noise_rate = (
        float(confident_other_class.sum() / len(y_encoded)) if len(y_encoded) else None
    )

    candidate_positions = np.flatnonzero(suspected)
    ranked_positions = candidate_positions[np.argsort(noise_score[candidate_positions])[::-1]]
    candidates = []
    for pos in ranked_positions[:top_n]:
        candidates.append(
            {
                "row_index": _json_scalar(x_eval.index[pos]),
                "sample_id": _json_scalar(ids_eval.iloc[pos]),
                "observed_label": _json_scalar(y_eval.iloc[pos]),
                "suggested_label": class_lookup.get(
                    str(encoder.classes_[predicted[pos]]),
                    _json_scalar(encoder.classes_[predicted[pos]]),
                ),
                "suggested_confidence": float(suggested_confidence[pos]),
                "assigned_label_probability": float(assigned_probability[pos]),
                "noise_score": float(noise_score[pos]),
            }
        )

    per_class: dict[str, dict[str, Any]] = {}
    for class_pos, class_name in enumerate(encoder.classes_):
        class_mask = y_encoded == class_pos
        class_total = int(class_mask.sum())
        class_suspected = int((suspected & class_mask).sum())
        per_class[str(class_name)] = {
            "evaluated_rows": class_total,
            "suspected_count": class_suspected,
            "suspected_rate": float(class_suspected / class_total) if class_total else None,
        }

    evaluated_rows = int(len(x_eval))
    suspected_count = int(suspected.sum())
    suspected_rate = float(suspected_count / evaluated_rows) if evaluated_rows else None
    return {
        "status": "ok",
        "method": "stratified out-of-fold logistic regression",
        "interpretation": (
            "Candidates are model-label disagreements for review, not confirmed annotation errors."
        ),
        "labeled_rows": labeled_rows,
        "evaluated_rows": evaluated_rows,
        "coverage_rate": float(evaluated_rows / labeled_rows) if labeled_rows else 0.0,
        "sampled": sampled,
        "num_classes": num_classes,
        "cross_validation_folds": int(folds),
        "confidence_threshold": float(confidence_threshold),
        "warning_threshold": float(warning_threshold),
        "suspected_label_noise_count": suspected_count,
        "suspected_label_noise_rate": suspected_rate,
        "estimated_label_noise_rate": cl_noise_rate,
        "out_of_fold_accuracy": float((predicted == y_encoded).mean()),
        "mean_assigned_label_probability": float(assigned_probability.mean()),
        "features_used": [str(c) for c in feature_cols],
        "numeric_feature_count": int(len(numeric_cols)),
        "categorical_feature_count": int(len(categorical_cols)),
        "dropped_high_cardinality_features": dropped_high_cardinality[:20],
        "per_class": per_class,
        "top_suspected_samples": candidates,
    }


__all__ = ["SKLEARN_OK", "assess_label_noise"]
