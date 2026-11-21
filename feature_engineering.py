from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, TypedDict

import pandas as pd


TEAM_NAME_MAP: dict[str, str] = {
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
    "Kings XI Punjab": "Punjab Kings",
    "Delhi Daredevils": "Delhi Capitals",
    "Rising Pune Supergiant": "Rising Pune Supergiants",
    "Pune Warriors": "Pune Warriors India",
}

# Only aliases with explicit source evidence are merged.
VENUE_ALIASES: dict[str, str] = {
    "Arun Jaitley Stadium": "arun_jaitley_stadium",
    "Arun Jaitley Stadium, Delhi": "arun_jaitley_stadium",
    "Feroz Shah Kotla": "arun_jaitley_stadium",
}

FEATURE_NAMES = [
    "batting_team_win_rate_5",
    "bowling_team_win_rate_5",
    "batting_team_win_rate_10",
    "bowling_team_win_rate_10",
    "batting_team_head_to_head_rate",
    "batting_team_venue_win_rate",
    "bowling_team_venue_win_rate",
    "batting_team_completed_matches_5",
    "bowling_team_completed_matches_5",
    "batting_team_completed_matches_10",
    "bowling_team_completed_matches_10",
    "head_to_head_completed_match_count",
    "batting_team_venue_completed_match_count",
    "bowling_team_venue_completed_match_count",
]

CATEGORICAL_FEATURES = [
    "batting_team",
    "bowling_team",
    "venue_key",
    "toss_winner",
    "toss_decision",
]


class ValidatedHistoryMatch(TypedDict):
    match_date: pd.Timestamp
    team1: str
    team2: str
    winner: str
    venue_key: str
    result: str


class ValidatedTrainingMatch(TypedDict):
    match_date: pd.Timestamp
    team1: str
    team2: str
    winner: str
    venue_key: str
    toss_winner: str
    toss_decision: str
    batting_team: str
    bowling_team: str
    target_batting_win: int


def clean_text(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.strip().split())


def normalize_team_name(value: object) -> str:
    cleaned = clean_text(value)
    return TEAM_NAME_MAP.get(cleaned, cleaned)


def normalize_toss_decision(value: object) -> str:
    cleaned = clean_text(value).lower()
    if cleaned in {"bat", "batting"}:
        return "bat"
    if cleaned in {"field", "bowl", "fielding", "bowling"}:
        return "field"
    return ""


def normalize_venue(value: object) -> str:
    cleaned = clean_text(value)
    if not cleaned:
        return "unknown_venue"
    return VENUE_ALIASES.get(cleaned, cleaned)


def parse_match_date(value: object) -> pd.Timestamp:
    if value is None:
        return pd.NaT
    try:
        parsed = pd.to_datetime(value, errors="coerce", dayfirst=True)
        return parsed if pd.Timestamp(parsed).tzinfo is None else pd.Timestamp(parsed).tz_localize(None)
    except (TypeError, ValueError, OverflowError):
        return pd.NaT


def is_completed_for_history(match: dict[str, Any]) -> ValidatedHistoryMatch | None:
    result = clean_text(match.get("result")).lower()
    if result not in {"win", "tie"}:
        return None

    match_date = parse_match_date(match.get("match_date"))
    if pd.isna(match_date):
        return None

    team1 = normalize_team_name(match.get("team1"))
    team2 = normalize_team_name(match.get("team2"))
    winner = normalize_team_name(match.get("winner"))

    if not team1 or not team2 or team1 == team2:
        return None
    if not winner or winner not in {team1, team2}:
        return None

    return {
        "match_date": match_date,
        "team1": team1,
        "team2": team2,
        "winner": winner,
        "venue_key": normalize_venue(match.get("venue")),
        "result": result,
    }


def validate_training_record(match: dict[str, Any]) -> ValidatedTrainingMatch | None:
    historical = is_completed_for_history(match)
    if historical is None:
        return None

    team1 = historical["team1"]
    team2 = historical["team2"]
    toss_winner = normalize_team_name(match.get("toss_winner"))
    toss_decision = normalize_toss_decision(match.get("toss_decision"))

    if toss_winner not in {team1, team2}:
        return None
    if toss_decision not in {"bat", "field"}:
        return None

    other_team = team2 if toss_winner == team1 else team1
    if toss_decision == "bat":
        batting_team = toss_winner
        bowling_team = other_team
    else:
        batting_team = other_team
        bowling_team = toss_winner

    if (
        not batting_team
        or not bowling_team
        or batting_team == bowling_team
        or batting_team not in {team1, team2}
        or bowling_team not in {team1, team2}
    ):
        return None

    return {
        "match_date": historical["match_date"],
        "team1": team1,
        "team2": team2,
        "winner": historical["winner"],
        "venue_key": historical["venue_key"],
        "toss_winner": toss_winner,
        "toss_decision": toss_decision,
        "batting_team": batting_team,
        "bowling_team": bowling_team,
        "target_batting_win": int(historical["winner"] == batting_team),
    }


def ordered_team_pair(team1: str, team2: str) -> tuple[str, str]:
    return tuple(sorted((normalize_team_name(team1), normalize_team_name(team2))))


class FeatureState:
    def __init__(self) -> None:
        self.team_history: dict[str, list[int]] = {}
        self.head_to_head_history: dict[tuple[str, str], dict[str, int]] = {}
        self.team_venue_history: dict[tuple[str, str], dict[str, int]] = {}

    def rolling_rate(self, team: str, window: int) -> tuple[float, int]:
        team = normalize_team_name(team)
        outcomes = self.team_history.get(team, [])[-window:]
        if not outcomes:
            return 0.5, 0
        return sum(outcomes) / len(outcomes), len(outcomes)

    def head_to_head_rate(
        self,
        batting_team: str,
        bowling_team: str,
    ) -> tuple[float, int]:
        batting_team = normalize_team_name(batting_team)
        bowling_team = normalize_team_name(bowling_team)
        pair = ordered_team_pair(batting_team, bowling_team)
        record = self.head_to_head_history.get(pair)
        if record is None or record["completed_matches"] == 0:
            return 0.5, 0

        total = record["completed_matches"]
        if batting_team == pair[0]:
            wins = record["first_team_wins"]
        else:
            wins = total - record["first_team_wins"]
        return wins / total, total

    def venue_rate(self, team: str, venue_key: str) -> tuple[float, int]:
        team = normalize_team_name(team)
        record = self.team_venue_history.get((team, normalize_venue(venue_key)))
        if record is None or record["completed_matches"] == 0:
            return 0.5, 0
        return record["wins"] / record["completed_matches"], record["completed_matches"]

    def generate_features(
        self,
        batting_team: str,
        bowling_team: str,
        venue: str,
    ) -> dict[str, float | int]:
        batting_team = normalize_team_name(batting_team)
        bowling_team = normalize_team_name(bowling_team)
        venue_key = normalize_venue(venue)

        if not batting_team or not bowling_team or batting_team == bowling_team:
            raise ValueError("Invalid batting/bowling team pair.")

        batting_5, batting_5_count = self.rolling_rate(batting_team, 5)
        bowling_5, bowling_5_count = self.rolling_rate(bowling_team, 5)
        batting_10, batting_10_count = self.rolling_rate(batting_team, 10)
        bowling_10, bowling_10_count = self.rolling_rate(bowling_team, 10)
        head_to_head, head_to_head_count = self.head_to_head_rate(
            batting_team,
            bowling_team,
        )
        batting_venue, batting_venue_count = self.venue_rate(
            batting_team,
            venue_key,
        )
        bowling_venue, bowling_venue_count = self.venue_rate(
            bowling_team,
            venue_key,
        )

        values: dict[str, float | int] = {
            "batting_team_win_rate_5": batting_5,
            "bowling_team_win_rate_5": bowling_5,
            "batting_team_win_rate_10": batting_10,
            "bowling_team_win_rate_10": bowling_10,
            "batting_team_head_to_head_rate": head_to_head,
            "batting_team_venue_win_rate": batting_venue,
            "bowling_team_venue_win_rate": bowling_venue,
            "batting_team_completed_matches_5": batting_5_count,
            "bowling_team_completed_matches_5": bowling_5_count,
            "batting_team_completed_matches_10": batting_10_count,
            "bowling_team_completed_matches_10": bowling_10_count,
            "head_to_head_completed_match_count": head_to_head_count,
            "batting_team_venue_completed_match_count": batting_venue_count,
            "bowling_team_venue_completed_match_count": bowling_venue_count,
        }
        return {name: values[name] for name in FEATURE_NAMES}

    def update_from_completed_match(
        self,
        match: dict[str, Any] | ValidatedHistoryMatch,
        before_date: pd.Timestamp,
    ) -> None:
        if pd.isna(before_date):
            return

        if "match_date" in match and isinstance(match["match_date"], pd.Timestamp):
            validated = match
        else:
            validated = is_completed_for_history(match)
        if validated is None:
            return

        match_date = validated["match_date"]
        if pd.isna(match_date) or match_date >= before_date:
            return

        team1 = normalize_team_name(validated["team1"])
        team2 = normalize_team_name(validated["team2"])
        winner = normalize_team_name(validated["winner"])
        venue_key = validated["venue_key"]

        if not team1 or not team2 or team1 == team2:
            return
        if winner not in {team1, team2}:
            return

        self.team_history.setdefault(team1, []).append(1 if winner == team1 else 0)
        self.team_history.setdefault(team2, []).append(1 if winner == team2 else 0)

        pair = ordered_team_pair(team1, team2)
        pair_record = self.head_to_head_history.setdefault(
            pair,
            {"completed_matches": 0, "first_team_wins": 0},
        )
        pair_record["completed_matches"] += 1
        if winner == pair[0]:
            pair_record["first_team_wins"] += 1

        for team in (team1, team2):
            venue_record = self.team_venue_history.setdefault(
                (team, venue_key),
                {"completed_matches": 0, "wins": 0},
            )
            venue_record["completed_matches"] += 1
            if winner == team:
                venue_record["wins"] += 1


def load_clean_match_history(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [column.strip().lower() for column in df.columns]

    if "match_date" in df.columns and "date" not in df.columns:
        df["date"] = df["match_date"]
    if "match_winner" in df.columns and "winner" not in df.columns:
        df["winner"] = df["match_winner"]

    if "match_date" in df.columns:
        df["match_date"] = df["match_date"]

    df["parsed_date"] = df["match_date"].map(parse_match_date)
    df["normalized_team1"] = df["team1"].map(normalize_team_name)
    df["normalized_team2"] = df["team2"].map(normalize_team_name)
    df["normalized_winner"] = df["winner"].map(normalize_team_name)
    df["venue_key"] = df["venue"].map(normalize_venue)
    df["normalized_toss_winner"] = df["toss_winner"].map(normalize_team_name)
    df["normalized_toss_decision"] = df["toss_decision"].map(normalize_toss_decision)

    df = df.sort_values(["parsed_date", "match_id"], kind="mergesort")
    df = df[df["parsed_date"].notna()].copy()
    return df.reset_index(drop=True)


def generate_training_features(
    matches: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    valid_history: list[ValidatedHistoryMatch] = []
    valid_training: list[ValidatedTrainingMatch] = []

    for _, row in matches.iterrows():
        raw = row.to_dict()
        historical = is_completed_for_history(raw)
        if historical is not None:
            valid_history.append(historical)

        training = validate_training_record(raw)
        if training is not None:
            valid_training.append(training)

    state = FeatureState()
    rows: list[dict[str, float | int]] = []

    training_dates = sorted({
        match["match_date"]
        for match in valid_training
    } | {
        match["match_date"]
        for match in valid_history
    })

    for index, date_d in enumerate(training_dates):
        training_on_d = [
            match
            for match in valid_training
            if match["match_date"] == date_d
        ]

        for match in training_on_d:
            features = state.generate_features(
                batting_team=match["batting_team"],
                bowling_team=match["bowling_team"],
                venue=match["venue_key"],
            )
            features["batting_team"] = match["batting_team"]
            features["bowling_team"] = match["bowling_team"]
            features["venue_key"] = match["venue_key"]
            features["toss_winner"] = match["toss_winner"]
            features["toss_decision"] = match["toss_decision"]
            features["match_date"] = match["match_date"]
            features["target_batting_win"] = match["target_batting_win"]
            rows.append(features)

        if index + 1 < len(training_dates):
            next_date = training_dates[index + 1]
        else:
            next_date = date_d + pd.Timedelta(days=1)

        history_on_d = [
            match
            for match in valid_history
            if match["match_date"] == date_d
        ]
        for match in history_on_d:
            state.update_from_completed_match(
                match,
                before_date=next_date,
            )

    if not rows:
        raise ValueError("No valid supervised matches could be generated.")

    result = pd.DataFrame(rows)
    categorical_history = pd.DataFrame(valid_training)
    return result, categorical_history


def generate_live_features(
    history: pd.DataFrame,
    batting_team: str,
    bowling_team: str,
    venue: str,
    prediction_date: pd.Timestamp,
) -> dict[str, float | int]:
    if pd.isna(prediction_date):
        raise ValueError("A valid prediction date is required.")

    state = FeatureState()
    for _, row in history.sort_values(["parsed_date", "match_id"], kind="mergesort").iterrows():
        validated = is_completed_for_history(row.to_dict())
        if validated is None:
            continue
        state.update_from_completed_match(
            validated,
            before_date=prediction_date,
        )

    return state.generate_features(
        batting_team=batting_team,
        bowling_team=bowling_team,
        venue=venue,
    )
