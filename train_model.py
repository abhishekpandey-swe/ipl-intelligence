from __future__ import annotations

from pathlib import Path
from typing import Any
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
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
    val = decision.strip().lower()
    if val in {"bat", "batting"}:
        return "bat"
    if val in {"field", "bowl", "fielding", "bowling"}:
        return "field"
    return ""


def derive_innings_teams(
    team1: str,
    team2: str,
    toss_winner: str,
    toss_decision: str,
) -> tuple[str, str]:
    """Derive first-innings batting and bowling teams and enforce toss consistency."""
    t1 = normalize_team_name(team1)
    t2 = normalize_team_name(team2)
    toss_w = normalize_team_name(toss_winner)
    decision = normalize_toss_decision(toss_decision)

    if not t1 or not t2:
        raise ValueError("Both Team 1 and Team 2 must be provided.")
    if t1 == t2:
        raise ValueError("Team 1 and Team 2 must be different teams.")
    if toss_w not in {t1, t2}:
        raise ValueError("Toss winner must be either Team 1 or Team 2.")
    if decision not in {"bat", "field"}:
        raise ValueError("Toss decision must be 'bat' or 'field'.")

    other_team = t2 if toss_w == t1 else t1

    if decision == "bat":
        batting_team = toss_w
        bowling_team = other_team
    else:
        batting_team = other_team
        bowling_team = toss_w

    if decision == "bat" and batting_team != toss_w:
        raise ValueError("Invalid combination: when toss winner chooses BAT, toss winner must bat.")
    if decision == "field" and batting_team != other_team:
        raise ValueError("Invalid combination: when toss winner chooses FIELD, the other team must bat.")

    return batting_team, bowling_team


def build_feature_row(
    team1: str,
    team2: str,
    venue: str,
    toss_winner: str,
    toss_decision: str,
) -> pd.DataFrame:
    """Build a single-row feature DataFrame using the exact training schema."""
    clean_venue = " ".join((venue or "").strip().split())
    if not clean_venue:
        raise ValueError("Match venue cannot be empty.")

    batting_team, bowling_team = derive_innings_teams(
        team1=team1,
        team2=team2,
        toss_winner=toss_winner,
        toss_decision=toss_decision,
    )
    clean_toss_winner = normalize_team_name(toss_winner)
    clean_decision = normalize_toss_decision(toss_decision)

    row = {
        "batting_team": batting_team,
        "bowling_team": bowling_team,
        "venue": clean_venue,
        "toss_winner": clean_toss_winner,
        "toss_decision": clean_decision,
    }
    return pd.DataFrame([row], columns=FEATURE_COLS)


def get_selected_model_metrics(artifact: dict[str, Any]) -> dict[str, float]:
    """Return {'accuracy', 'precision', 'recall'} for the selected model in artifact."""
    model_name = artifact.get("model_name")
    all_metrics = artifact.get("all_model_metrics")

    if isinstance(all_metrics, dict) and isinstance(model_name, str) and model_name in all_metrics:
        raw_metrics = all_metrics[model_name]
    else:
        raw_metrics = artifact.get("metrics", {})

    if not isinstance(raw_metrics, dict):
        return {}

    resolved: dict[str, float] = {}
    for metric_key in ("accuracy", "precision", "recall"):
        val = raw_metrics.get(metric_key)
        if isinstance(val, (int, float)):
            resolved[metric_key] = float(val)
    return resolved


def load_and_clean_dataset(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{csv_path}'. Place a historical IPL match CSV at "
            "data/matches.csv with date/match_date, team1, team2, venue, "
            "toss_winner, toss_decision, and winner/match_winner."
        )

    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]

    if "match_date" in df.columns and "date" not in df.columns:
        df["date"] = df["match_date"]
    if "match_winner" in df.columns and "winner" not in df.columns:
        df["winner"] = df["match_winner"]

    required = {"date", "team1", "team2", "venue", "toss_winner", "toss_decision", "winner"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Dataset is missing required columns for chronological training: {sorted(missing)}"
        )

    if "result" in df.columns:
        df = df[df["result"].astype(str).str.lower() != "no result"]
    if "eliminator" in df.columns:
        df = df[df["eliminator"].astype(str).str.upper() != "Y"]

    df = df.dropna(
        subset=["date", "team1", "team2", "venue", "toss_winner", "toss_decision", "winner"]
    ).copy()

    df["parsed_date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=True)
    invalid_dates = int(df["parsed_date"].isna().sum())
    if invalid_dates > 0:
        raise ValueError(
            f"Found {invalid_dates} row(s) with invalid or unparseable match dates. "
            "Valid match dates are required for chronological train-test splitting."
        )

    for col in ["team1", "team2", "toss_winner", "winner"]:
        df[col] = df[col].map(normalize_team_name)

    df["venue"] = df["venue"].astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
    df["toss_decision"] = df["toss_decision"].map(normalize_toss_decision)

    valid_mask = [
        bool(r["team1"])
        and bool(r["team2"])
        and (r["team1"] != r["team2"])
        and bool(r["venue"])
        and (r["toss_winner"] in {r["team1"], r["team2"]})
        and (r["toss_decision"] in {"bat", "field"})
        and (r["winner"] in {r["team1"], r["team2"]})
        for _, r in df.iterrows()
    ]
    df = df[valid_mask].copy()

    df = df.sort_values(by="parsed_date", kind="mergesort").reset_index(drop=True)

    batting_teams: list[str] = []
    bowling_teams: list[str] = []
    for _, row in df.iterrows():
        bat_t, bowl_t = derive_innings_teams(
            team1=row["team1"],
            team2=row["team2"],
            toss_winner=row["toss_winner"],
            toss_decision=row["toss_decision"],
        )
        batting_teams.append(bat_t)
        bowling_teams.append(bowl_t)

    df["batting_team"] = batting_teams
    df["bowling_team"] = bowling_teams
    df["target_batting_win"] = (df["winner"] == df["batting_team"]).astype(int)

    return df


def build_pipeline(estimator) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
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


def main() -> None:
    df = load_and_clean_dataset(DATA_PATH)
    if len(df) < 30:
        raise ValueError(f"Too few valid matches after cleaning ({len(df)}). Need at least 30.")

    split_idx = int(len(df) * 0.80)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    X_train, y_train = train_df[FEATURE_COLS], train_df["target_batting_win"]
    X_test, y_test = test_df[FEATURE_COLS], test_df["target_batting_win"]

    candidates = {
        "Logistic Regression": build_pipeline(
            LogisticRegression(max_iter=1000, C=1.0, random_state=42)
        ),
        "Random Forest Classifier": build_pipeline(
            RandomForestClassifier(
                n_estimators=200,
                max_depth=8,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1,
            )
        ),
    }

    best_name = ""
    best_model = None
    best_acc = -1.0
    all_metrics: dict[str, dict[str, float]] = {}

    for name, pipe in candidates.items():
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        acc = float(accuracy_score(y_test, preds))
        prec = float(precision_score(y_test, preds, zero_division=0))
        rec = float(recall_score(y_test, preds, zero_division=0))
        all_metrics[name] = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
        }
        if acc > best_acc:
            best_acc = acc
            best_name = name
            best_model = pipe

    assert best_model is not None

    teams = sorted(set(df["team1"]).union(set(df["team2"])))
    venues = sorted(df["venue"].unique().tolist())

    artifact = {
        "model": best_model,
        "model_name": best_name,
        "feature_cols": list(FEATURE_COLS),
        "teams": teams,
        "venues": venues,
        "total_matches": int(len(df)),
        "train_matches": int(len(train_df)),
        "test_matches": int(len(test_df)),
        "metrics": all_metrics[best_name],
        "all_model_metrics": all_metrics,
        "dataset_source": str(DATA_PATH.relative_to(BASE_DIR)),
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, MODEL_PATH)

    selected_metrics = get_selected_model_metrics(artifact)
    print(f"Saved model to {MODEL_PATH}")
    print(f"Selected Model: {best_name}")
    print(f"Total Cleaned Matches: {len(df)} (Train: {len(train_df)}, Test: {len(test_df)})")
    print(f"Test Accuracy:  {selected_metrics['accuracy']:.4f}")
    print(f"Test Precision: {selected_metrics['precision']:.4f}")
    print(f"Test Recall:    {selected_metrics['recall']:.4f}")


if __name__ == "__main__":
    main()
