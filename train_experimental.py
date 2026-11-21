from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from feature_engineering import (
    CATEGORICAL_FEATURES,
    FEATURE_NAMES,
    generate_training_features,
    load_clean_match_history,
)


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "matches.csv"
EXPERIMENTAL_MODEL_PATH = BASE_DIR / "models" / "ipl_model_experimental.joblib"
REPORT_PATH = BASE_DIR / "experimental_model_report.json"

NUMERIC_FEATURES = [
    name
    for name in FEATURE_NAMES
    if name not in CATEGORICAL_FEATURES
]
MODEL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def build_pipeline(estimator: Any) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
            ("numeric", "passthrough", NUMERIC_FEATURES),
        ],
        remainder="drop",
    )
    return Pipeline(
        [
            ("preprocessor", preprocessor),
            ("classifier", estimator),
        ]
    )


def candidates() -> dict[str, Any]:
    return {
        "Majority Baseline": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": build_pipeline(
            LogisticRegression(max_iter=1000, C=1.0, random_state=42)
        ),
        "Random Forest": build_pipeline(
            RandomForestClassifier(
                n_estimators=250,
                max_depth=8,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1,
            )
        ),
        "Gradient Boosting": build_pipeline(
            GradientBoostingClassifier(
                n_estimators=150,
                learning_rate=0.05,
                max_depth=3,
                random_state=42,
            )
        ),
        "Hist Gradient Boosting": build_pipeline(
            HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_iter=150,
                max_leaf_nodes=15,
                l2_regularization=1.0,
                random_state=42,
            )
        ),
        "Extra Trees": build_pipeline(
            ExtraTreesClassifier(
                n_estimators=250,
                max_depth=12,
                min_samples_leaf=3,
                random_state=42,
                n_jobs=-1,
            )
        ),
    }


def probability_for_positive_class(model: Any, features: pd.DataFrame) -> list[float] | None:
    if not hasattr(model, "predict_proba"):
        return None
    probabilities = model.predict_proba(features)
    classes = list(model.classes_)
    if 1 not in classes:
        return [0.0] * len(features)
    return probabilities[:, classes.index(1)].tolist()


def evaluate(
    name: str,
    model: Any,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict[str, Any]:
    X_train = train_df[MODEL_FEATURES]
    y_train = train_df["target_batting_win"]
    X_test = test_df[MODEL_FEATURES]
    y_test = test_df["target_batting_win"]

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    result: dict[str, Any] = {
        "model": name,
        "rows": int(len(test_df)),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=[0, 1]).tolist(),
    }

    probabilities = probability_for_positive_class(model, X_test)
    if probabilities is not None:
        result["roc_auc"] = float(roc_auc_score(y_test, probabilities))
        result["brier"] = float(brier_score_loss(y_test, probabilities))
        result["log_loss"] = float(log_loss(y_test, probabilities, labels=[0, 1]))

    return result


def main() -> None:
    history = load_clean_match_history(DATA_PATH)
    feature_df, _ = generate_training_features(history)

    if "match_date" not in feature_df.columns:
        raise ValueError("Generated features are missing match_date.")

    feature_df["match_date"] = pd.to_datetime(feature_df["match_date"], errors="coerce")
    feature_df = feature_df.dropna(subset=["match_date"]).copy()

    development = feature_df[feature_df["match_date"] < pd.Timestamp("2024-01-01")].copy()
    validation = feature_df[
        (feature_df["match_date"] >= pd.Timestamp("2024-01-01"))
        & (feature_df["match_date"] < pd.Timestamp("2025-01-01"))
    ].copy()
    final_test = feature_df[
        feature_df["match_date"] >= pd.Timestamp("2025-01-01")
    ].copy()

    if any(len(frame) == 0 for frame in (development, validation, final_test)):
        raise ValueError("One of the chronological periods is empty.")

    validation_results: list[dict[str, Any]] = []
    candidate_models = candidates()
    for name, model in candidate_models.items():
        validation_results.append(evaluate(name, model, development, validation))

    selectable = [
        result
        for result in validation_results
        if result["model"] != "Majority Baseline"
    ]
    selected = max(
        selectable,
        key=lambda result: (
            result["balanced_accuracy"],
            result["accuracy"],
            -result.get("log_loss", float("inf")),
        ),
    )

    selected_name = selected["model"]
    selected_model = candidates()[selected_name]
    final_train = pd.concat([development, validation], ignore_index=True)
    final_result = evaluate(
        selected_name,
        selected_model,
        final_train,
        final_test,
    )

    all_final_results = [
        evaluate(name, model, final_train, final_test)
        for name, model in candidates().items()
    ]

    selected_model.fit(
        final_train[MODEL_FEATURES],
        final_train["target_batting_win"],
    )

    artifact = {
        "model": selected_model,
        "model_name": selected_name,
        "model_version": "experimental-historical-v1",
        "feature_cols": MODEL_FEATURES,
        "categorical_feature_cols": CATEGORICAL_FEATURES,
        "numeric_feature_cols": NUMERIC_FEATURES,
        "historical_feature_names": FEATURE_NAMES,
        "dataset_source": "data/matches.csv",
        "dataset_last_updated": "2025-06-03",
        "development_period": "2008-04-18 to 2023-05-29",
        "validation_period": "2024-03-22 to 2024-05-26",
        "final_test_period": "2025-03-22 to 2025-06-03",
        "development_rows": int(len(development)),
        "validation_rows": int(len(validation)),
        "final_test_rows": int(len(final_test)),
        "selected_by": "2024 balanced accuracy, then accuracy, then log loss",
        "validation_results": validation_results,
        "final_results": all_final_results,
        "selected_final_result": final_result,
    }

    EXPERIMENTAL_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, EXPERIMENTAL_MODEL_PATH)

    report = {
        "artifact": str(EXPERIMENTAL_MODEL_PATH.relative_to(BASE_DIR)),
        "period_counts": {
            "development": int(len(development)),
            "validation": int(len(validation)),
            "final_test": int(len(final_test)),
        },
        "validation_results": validation_results,
        "selected_model": selected_name,
        "final_results": all_final_results,
        "selected_final_result": final_result,
    }
    REPORT_PATH.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("Experimental model evaluation completed.")
    print(f"Selected on 2024 validation: {selected_name}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
