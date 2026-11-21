from __future__ import annotations

from pathlib import Path
from typing import Any
import joblib

from train_model import (
    FEATURE_COLS,
    build_feature_row,
    derive_innings_teams,
    normalize_team_name,
    normalize_toss_decision,
)

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = BASE_DIR / "models" / "ipl_model.joblib"


class ModelNotFoundError(FileNotFoundError):
    """Raised when the trained model artifact is missing."""


class InvalidMatchInputError(ValueError):
    """Raised when user match inputs fail validation or consistency checks."""


def load_model_artifact(model_path: Path = DEFAULT_MODEL_PATH) -> dict[str, Any]:
    if not model_path.exists():
        raise ModelNotFoundError(
            f"Trained model not found at '{model_path}'. Run `python train_model.py` first."
        )

    artifact = joblib.load(model_path)
    if not isinstance(artifact, dict) or "model" not in artifact:
        raise ValueError("Invalid model artifact format in ipl_model.joblib.")

    expected_cols = artifact.get("feature_cols")
    if expected_cols and list(expected_cols) != FEATURE_COLS:
        raise ValueError(
            f"Feature column mismatch between model artifact ({expected_cols}) "
            f"and pipeline ({FEATURE_COLS}). Re-run `python train_model.py`."
        )

    return artifact


def validate_match_inputs(
    team1: str,
    team2: str,
    toss_winner: str,
    toss_decision: str,
    venue: str,
) -> tuple[str, str, str, str, str, str, str]:
    t1 = normalize_team_name(team1)
    t2 = normalize_team_name(team2)
    toss_w = normalize_team_name(toss_winner)
    toss_d = normalize_toss_decision(toss_decision)
    ven = " ".join((venue or "").strip().split())

    if not t1 or not t2:
        raise InvalidMatchInputError("Both Team 1 and Team 2 must be selected.")
    if t1 == t2:
        raise InvalidMatchInputError("Team 1 and Team 2 must be different.")
    if not ven:
        raise InvalidMatchInputError("Match Venue cannot be empty.")
    if toss_w not in {t1, t2}:
        raise InvalidMatchInputError("Toss Winner must be either Team 1 or Team 2.")
    if toss_d not in {"bat", "field"}:
        raise InvalidMatchInputError("Toss Decision must be either 'bat' or 'field'.")

    try:
        batting_team, bowling_team = derive_innings_teams(
            team1=t1,
            team2=t2,
            toss_winner=toss_w,
            toss_decision=toss_d,
        )
    except ValueError as exc:
        raise InvalidMatchInputError(str(exc)) from exc

    return t1, t2, toss_w, toss_d, ven, batting_team, bowling_team


def predict_match_winner(
    team1: str,
    team2: str,
    toss_winner: str,
    toss_decision: str,
    venue: str,
    model_path: Path = DEFAULT_MODEL_PATH,
) -> dict[str, Any]:
    t1, t2, toss_w, toss_d, ven, batting_team, bowling_team = validate_match_inputs(
        team1=team1,
        team2=team2,
        toss_winner=toss_winner,
        toss_decision=toss_decision,
        venue=venue,
    )

    artifact = load_model_artifact(model_path)
    model = artifact["model"]

    try:
        input_df = build_feature_row(
            team1=t1,
            team2=t2,
            venue=ven,
            toss_winner=toss_w,
            toss_decision=toss_d,
        )
    except ValueError as exc:
        raise InvalidMatchInputError(str(exc)) from exc

    probabilities = model.predict_proba(input_df)[0]
    classes = list(model.classes_)

    prob_by_class = {int(cls): float(prob) for cls, prob in zip(classes, probabilities)}
    batting_win_prob = prob_by_class.get(1, 0.0)
    bowling_win_prob = prob_by_class.get(0, 1.0 - batting_win_prob)

    total = batting_win_prob + bowling_win_prob
    if total > 0:
        batting_win_prob /= total
        bowling_win_prob /= total

    predicted_winner = batting_team if batting_win_prob >= bowling_win_prob else bowling_team
    confidence = max(batting_win_prob, bowling_win_prob)

    return {
        "predicted_winner": predicted_winner,
        "confidence": confidence,
        "team1": t1,
        "team2": t2,
        "batting_team": batting_team,
        "bowling_team": bowling_team,
        "batting_win_probability": batting_win_prob,
        "bowling_win_probability": bowling_win_prob,
        "venue": ven,
        "toss_winner": toss_w,
        "toss_decision": toss_d,
        "model_name": artifact.get("model_name", "Trained Classifier"),
    }
