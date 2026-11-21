from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "matches.csv"
MODEL_PATH = BASE_DIR / "models" / "ipl_model.joblib"

TEAM_NAME_MAP: dict[str, str] = {
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
    "Kings XI Punjab": "Punjab Kings",
    "Delhi Daredevils": "Delhi Capitals",
    "Rising Pune Supergiant": "Rising Pune Supergiants",
    "Pune Warriors": "Pune Warriors India",
}

FEATURE_COLS = [
    "batting_team",
    "bowling_team",
    "venue",
    "toss_winner",
    "toss_decision",
]


def normalize_team_name(name: object) -> str:
    if not isinstance(name, str):
        return ""
    cleaned = " ".join(name.strip().split())
    return TEAM_NAME_MAP.get(cleaned, cleaned)


def normalize_toss_decision(decision: object) -> str:
    if not isinstance(decision, str):
        return ""
    value = decision.strip().lower()
    if value in {"bat", "batting"}:
        return "bat"
    if value in {"field", "bowl", "fielding", "bowling"}:
        return "field"
    return ""


def derive_innings_teams(
    team1: str,
    team2: str,
    toss_winner: str,
    toss_decision: str,
) -> tuple[str, str]:
    team1 = normalize_team_name(team1)
    team2 = normalize_team_name(team2)
    toss_winner = normalize_team_name(toss_winner)
    toss_decision = normalize_toss_decision(toss_decision)

    if not team1 or not team2 or team1 == team2:
        raise ValueError("Invalid team pair.")
    if toss_winner not in {team1, team2}:
        raise ValueError("Invalid toss winner.")
    if toss_decision not in {"bat", "field"}:
        raise ValueError("Invalid toss decision.")

    other_team = team2 if toss_winner == team1 else team1
    if toss_decision == "bat":
        return toss_winner, other_team
    return other_team, toss_winner


def load_and_clean_dataset(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df.columns = [column.strip().lower() for column in df.columns]

    if "match_date" in df.columns and "date" not in df.columns:
        df["date"] = df["match_date"]
    if "match_winner" in df.columns and "winner" not in df.columns:
        df["winner"] = df["match_winner"]

    required = {
        "date",
        "team1",
        "team2",
        "venue",
        "toss_winner",
        "toss_decision",
        "winner",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if "result" in df.columns:
        df = df[df["result"].astype(str).str.lower() != "no result"].copy()
    if "eliminator" in df.columns:
        df = df[df["eliminator"].astype(str).str.upper() != "Y"].copy()

    df = df.dropna(
        subset=[
            "date",
            "team1",
            "team2",
            "venue",
            "toss_winner",
            "toss_decision",
            "winner",
        ]
    ).copy()

    df["parsed_date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=True)
    invalid_dates = int(df["parsed_date"].isna().sum())
    if invalid_dates:
        raise ValueError(f"Found {invalid_dates} invalid match date(s).")

    for column in ["team1", "team2", "toss_winner", "winner"]:
        df[column] = df[column].map(normalize_team_name)

    df["venue"] = df["venue"].astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
    df["toss_decision"] = df["toss_decision"].map(normalize_toss_decision)

    valid_rows: list[bool] = []
    batting_teams: list[str] = []
    bowling_teams: list[str] = []

    for _, row in df.iterrows():
        valid = (
            bool(row["team1"])
            and bool(row["team2"])
            and row["team1"] != row["team2"]
            and bool(row["venue"])
            and row["toss_winner"] in {row["team1"], row["team2"]}
            and row["toss_decision"] in {"bat", "field"}
            and row["winner"] in {row["team1"], row["team2"]}
        )
        valid_rows.append(valid)

        if valid:
            batting_team, bowling_team = derive_innings_teams(
                row["team1"],
                row["team2"],
                row["toss_winner"],
                row["toss_decision"],
            )
        else:
            batting_team, bowling_team = "", ""
        batting_teams.append(batting_team)
        bowling_teams.append(bowling_team)

    df = df.copy()
    df["batting_team"] = batting_teams
    df["bowling_team"] = bowling_teams
    df = df[pd.Series(valid_rows, index=df.index)].copy()
    df["target_batting_win"] = (df["winner"] == df["batting_team"]).astype(int)
    df = df.sort_values("parsed_date", kind="mergesort").reset_index(drop=True)
    return df


def build_pipeline(estimator: Any) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                FEATURE_COLS,
            )
        ],
        remainder="drop",
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", estimator),
        ]
    )


def build_candidates() -> dict[str, Any]:
    return {
        "Majority Baseline": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": build_pipeline(
            LogisticRegression(max_iter=1000, C=1.0, random_state=42)
        ),
        "Random Forest": build_pipeline(
            RandomForestClassifier(
                n_estimators=200,
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
    }


def positive_class_probability(model: Any, features: pd.DataFrame) -> list[float] | None:
    if not hasattr(model, "predict_proba"):
        return None

    probabilities = model.predict_proba(features)
    classes = list(model.classes_)
    if 1 not in classes:
        return [0.0] * len(features)
    positive_index = classes.index(1)
    return [float(value) for value in probabilities[:, positive_index]]


def evaluate_model(
    name: str,
    model: Any,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict[str, float | int | str]:
    X_train = train_df[FEATURE_COLS]
    y_train = train_df["target_batting_win"]
    X_test = test_df[FEATURE_COLS]
    y_test = test_df["target_batting_win"]

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    metrics: dict[str, float | int | str] = {
        "model": name,
        "rows": int(len(test_df)),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
    }

    probabilities = positive_class_probability(model, X_test)
    if probabilities is not None:
        metrics["roc_auc"] = float(roc_auc_score(y_test, probabilities))
        metrics["brier"] = float(brier_score_loss(y_test, probabilities))
        metrics["log_loss"] = float(log_loss(y_test, probabilities, labels=[0, 1]))

    return metrics


def print_period(label: str, df: pd.DataFrame) -> None:
    print(
        f"{label}: {len(df)} rows, "
        f"{df['parsed_date'].min().date()} to {df['parsed_date'].max().date()}, "
        f"class counts={df['target_batting_win'].value_counts().sort_index().to_dict()}"
    )


def print_metrics(title: str, metrics: list[dict[str, float | int | str]]) -> None:
    print(f"\n{title}")
    print(
        "model | rows | accuracy | balanced_accuracy | precision | "
        "recall | f1 | roc_auc | brier | log_loss"
    )
    for result in metrics:
        def format_metric(metric_name: str) -> str:
            value = result.get(metric_name)
            if isinstance(value, (int, float)):
                return f"{float(value):.4f}"
            return "N/A"

        print(
            f"{result['model']} | "
            f"{result['rows']} | "
            f"{result['accuracy']:.4f} | "
            f"{result['balanced_accuracy']:.4f} | "
            f"{result['precision']:.4f} | "
            f"{result['recall']:.4f} | "
            f"{result['f1']:.4f} | "
            f"{format_metric('roc_auc')} | "
            f"{format_metric('brier')} | "
            f"{format_metric('log_loss')}"
        )


def main() -> None:
    df = load_and_clean_dataset(DATA_PATH)
    if len(df) < 30:
        raise ValueError("Too few valid matches for evaluation.")

    print("IPL model evaluation - no files are written")
    print_period("Cleaned dataset", df)

    print("\nDataset limitations")
    print("- Match-level summary features only; no lineups, injuries, weather, pitch or live-state data.")
    print("- 59 venue strings exist in the source; several refer to the same stadium with naming variants.")
    print("- Team names were normalized for selected historical/current franchise names.")
    print("- Defunct teams and franchises whose squads changed substantially over time are included.")
    print("- This model uses toss information that may not be known at every pre-match prediction time.")
    print("- Venue, team and franchise effects may be sparse outside the latest seasons.")

    baseline_split = int(len(df) * 0.80)
    baseline_train = df.iloc[:baseline_split]
    baseline_test = df.iloc[baseline_split:]
    print("\nChronological baseline reproduction")
    print_period("Baseline train", baseline_train)
    print_period("Baseline test", baseline_test)

    baseline_results = [
        evaluate_model(name, candidate, baseline_train, baseline_test)
        for name, candidate in build_candidates().items()
    ]
    print_metrics("Baseline 80/20 metrics", baseline_results)

    if MODEL_PATH.exists():
        artifact = joblib.load(MODEL_PATH)
        if isinstance(artifact, dict):
            print(
                "\nExisting production artifact: "
                f"{artifact.get('model_name', 'unknown')}; "
                f"recorded metrics={artifact.get('metrics', {})}"
            )

    dev_start = pd.Timestamp("2008-01-01")
    validation_start = pd.Timestamp("2024-01-01")
    final_start = pd.Timestamp("2025-01-01")

    development_train = df[df["parsed_date"] < validation_start].copy()
    validation_df = df[
        (df["parsed_date"] >= validation_start)
        & (df["parsed_date"] < final_start)
    ].copy()
    final_test = df[df["parsed_date"] >= final_start].copy()

    print("\nReserved-period evaluation")
    print_period("Development train (< 2024)", development_train)
    print_period("Validation (2024)", validation_df)
    print_period("Final holdout (>= 2025)", final_test)

    if development_train.empty or validation_df.empty or final_test.empty:
        raise ValueError("Chronological development, validation or final period is empty.")

    validation_results = [
        evaluate_model(name, candidate, development_train, validation_df)
        for name, candidate in build_candidates().items()
    ]
    print_metrics("Validation-period metrics used for selection", validation_results)

    selectable = [result for result in validation_results if result["model"] != "Majority Baseline"]
    selected = max(
        selectable,
        key=lambda result: (
            float(result["balanced_accuracy"]),
            float(result["accuracy"]),
            -float(result["log_loss"]),
        ),
    )
    selected_name = str(selected["model"])
    print(
        f"\nSelected model on 2024 validation: {selected_name} "
        f"(balanced_accuracy={selected['balanced_accuracy']:.4f}, "
        f"accuracy={selected['accuracy']:.4f}, log_loss={selected['log_loss']:.4f})"
    )

    candidates = build_candidates()
    final_model = candidates[selected_name]
    final_result = evaluate_model(
        selected_name,
        final_model,
        pd.concat([development_train, validation_df], ignore_index=True),
        final_test,
    )
    print_metrics("Final 2025 holdout metrics", [final_result])

    majority_candidate = candidates["Majority Baseline"]
    majority_final = evaluate_model(
        "Majority Baseline",
        majority_candidate,
        pd.concat([development_train, validation_df], ignore_index=True),
        final_test,
    )
    print_metrics("Final 2025 majority baseline", [majority_final])

    all_final_train = pd.concat([development_train, validation_df], ignore_index=True)
    all_final_results = [
        evaluate_model(name, candidate, all_final_train, final_test)
        for name, candidate in candidates.items()
    ]
    print_metrics("All model metrics on the final 2025 holdout", all_final_results)


if __name__ == "__main__":
    main()
