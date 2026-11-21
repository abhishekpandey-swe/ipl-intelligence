from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from feature_engineering import normalize_team_name


CURRENT_IPL_TEAMS = [
    "Chennai Super Kings",
    "Delhi Capitals",
    "Gujarat Titans",
    "Kolkata Knight Riders",
    "Lucknow Super Giants",
    "Mumbai Indians",
    "Punjab Kings",
    "Rajasthan Royals",
    "Royal Challengers Bengaluru",
    "Sunrisers Hyderabad",
]


def load_match_history(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [column.strip().lower() for column in df.columns]
    if "match_date" in df.columns and "date" not in df.columns:
        df["date"] = df["match_date"]
    if "match_winner" in df.columns and "winner" not in df.columns:
        df["winner"] = df["match_winner"]

    df["parsed_date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=True)
    df["team1"] = df["team1"].map(normalize_team_name)
    df["team2"] = df["team2"].map(normalize_team_name)
    df["winner"] = df["winner"].map(normalize_team_name)
    df["season"] = df["season"].astype(str)
    return df


def prepare_match_analytics_history(
    history: pd.DataFrame,
) -> pd.DataFrame:
    """Normalize match-team columns for team and venue analytics.

    This does not alter the source file. It returns a copy with the columns
    required by Analytics, using the project's existing team-name normalizer.
    """
    if not isinstance(history, pd.DataFrame) or history.empty:
        return pd.DataFrame()

    if "winner" not in history.columns and "match_winner" in history.columns:
        history = history.copy()
        history["winner"] = history["match_winner"]

    required = {"team1", "team2", "venue", "season", "winner"}
    missing = required - set(history.columns)
    if missing:
        raise ValueError(
            "Match history is missing required analytics columns: "
            + ", ".join(sorted(missing))
        )

    prepared = history.copy()
    prepared["normalized_team1"] = prepared["team1"].map(normalize_team_name)
    prepared["normalized_team2"] = prepared["team2"].map(normalize_team_name)
    prepared["normalized_winner"] = prepared["winner"].map(normalize_team_name)

    valid_rows = (
        prepared["normalized_team1"].fillna("").astype(str).str.strip().ne("")
        & prepared["normalized_team2"].fillna("").astype(str).str.strip().ne("")
        & prepared["normalized_team1"].ne(prepared["normalized_team2"])
    )
    prepared = prepared[valid_rows].copy()
    return prepared


def calculate_team_comparison(
    history: pd.DataFrame,
    teams: list[str],
) -> pd.DataFrame:
    """Calculate actual match counts and wins for selected teams."""
    if history.empty:
        return pd.DataFrame(columns=["Team", "Matches", "Wins", "Losses"])

    rows: list[dict[str, object]] = []
    for team in teams:
        matches = history[
            history["normalized_team1"].eq(team)
            | history["normalized_team2"].eq(team)
        ]
        wins = int(matches["normalized_winner"].eq(team).sum())
        rows.append(
            {
                "Team": team,
                "Matches": int(len(matches)),
                "Wins": wins,
                "Losses": int(len(matches) - wins),
            }
        )
    return pd.DataFrame(rows, columns=["Team", "Matches", "Wins", "Losses"])


def team_profile(history: pd.DataFrame, team: str) -> dict[str, Any]:
    team_matches = history[
        history["team1"].eq(team) | history["team2"].eq(team)
    ].copy()
    completed = team_matches[
        team_matches["winner"].notna()
        & team_matches["winner"].ne("")
    ].copy()

    wins = int(completed["winner"].eq(team).sum())
    losses = int(completed["winner"].notna().sum() - wins)

    season_record = (
        completed.assign(
            result=completed["winner"].map(
                lambda winner: "Win" if winner == team else "Loss"
            )
        )
        .groupby(["season", "result"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    if "Win" not in season_record.columns:
        season_record["Win"] = 0
    if "Loss" not in season_record.columns:
        season_record["Loss"] = 0

    venue_record = (
        completed.assign(
            result=completed["winner"].map(
                lambda winner: "Win" if winner == team else "Loss"
            )
        )
        .groupby(["venue", "result"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    if "Win" not in venue_record.columns:
        venue_record["Win"] = 0
    if "Loss" not in venue_record.columns:
        venue_record["Loss"] = 0

    return {
        "team": team,
        "matches": int(len(completed)),
        "wins": wins,
        "losses": losses,
        "win_percentage": round(wins / len(completed) * 100, 1) if len(completed) else None,
        "season_record": season_record,
        "venue_record": venue_record.sort_values("Win", ascending=False),
    }


def player_reference_directory(history: pd.DataFrame) -> pd.DataFrame:
    if "player_of_match" not in history.columns:
        return pd.DataFrame(columns=["reference", "awards", "seasons"])

    references = history["player_of_match"].dropna().astype(str).str.strip()
    references = references[references.ne("") & references.ne("nan")]
    if references.empty:
        return pd.DataFrame(columns=["reference", "awards", "seasons"])

    temp = history[["player_of_match", "season"]].copy()
    temp["reference"] = temp["player_of_match"].dropna().astype(str).str.strip()
    temp = temp[temp["reference"].ne("") & temp["reference"].ne("nan")]
    grouped = (
        temp.groupby("reference")
        .agg(
            awards=("reference", "size"),
            first_season=("season", "min"),
            last_season=("season", "max"),
        )
        .reset_index()
        .sort_values(["awards", "reference"], ascending=[False, True])
    )
    grouped["season_range"] = (
        grouped["first_season"].astype(str)
        + " to "
        + grouped["last_season"].astype(str)
    )
    return grouped[["reference", "awards", "season_range"]]


def optional_player_data_status(data_dir: Path) -> list[dict[str, str]]:
    required = [
        ("player_statistics.csv", "Season/player batting and bowling statistics"),
        ("player_registry.csv", "Verified player names and identity mapping"),
        ("team_squads.csv", "Season-specific team and player associations"),
        ("deliveries.csv", "Ball-by-ball match data"),
    ]
    return [
        {
            "file": name,
            "purpose": purpose,
            "status": "available" if (data_dir / name).exists() else "not available",
        }
        for name, purpose in required
    ]


def load_processed_player_data(processed_dir: Path) -> dict[str, pd.DataFrame]:
    filenames = {
        "registry": "player_registry.csv",
        "batting": "player_batting_stats.csv",
        "bowling": "player_bowling_stats.csv",
        "fielding": "player_fielding_stats.csv",
        "season": "player_season_stats.csv",
        "match": "player_match_stats.csv",
        "team_squads": "team_squads.csv",
        "report": "processing_report.csv",
    }
    loaded: dict[str, pd.DataFrame] = {}
    for key, filename in filenames.items():
        path = processed_dir / filename
        loaded[key] = pd.read_csv(path) if path.exists() else pd.DataFrame()
    return loaded


def get_team_squad(
    team_squads: pd.DataFrame,
    team: str,
    season: str | None = None,
) -> pd.DataFrame:
    if team_squads.empty:
        return pd.DataFrame()
    filtered = team_squads[
        team_squads["team"].astype(str).eq(team)
    ].copy()
    if season:
        filtered = filtered[
            filtered["season"].astype(str).eq(season)
        ]
    return filtered.sort_values(["season", "player_name"], ascending=[False, True])


def get_player_profile(
    data: dict[str, pd.DataFrame],
    player_id: str,
) -> dict[str, pd.DataFrame | pd.Series | None]:
    registry = data["registry"]
    batting = data["batting"]
    bowling = data["bowling"]
    fielding = data["fielding"]
    season = data["season"]

    registry_row = (
        registry[registry["player_id"].astype(str).eq(player_id)].head(1)
        if not registry.empty
        else pd.DataFrame()
    )
    batting_row = (
        batting[batting["player_id"].astype(str).eq(player_id)].head(1)
        if not batting.empty
        else pd.DataFrame()
    )
    bowling_row = (
        bowling[bowling["player_id"].astype(str).eq(player_id)].head(1)
        if not bowling.empty
        else pd.DataFrame()
    )
    fielding_row = (
        fielding[fielding["player_id"].astype(str).eq(player_id)].head(1)
        if not fielding.empty
        else pd.DataFrame()
    )
    season_rows = (
        season[season["player_id"].astype(str).eq(player_id)].copy()
        if not season.empty
        else pd.DataFrame()
    )
    if not season_rows.empty:
        season_rows = season_rows.sort_values(
            ["season", "team"],
            kind="stable",
        )

    return {
        "registry": registry_row.iloc[0] if not registry_row.empty else None,
        "batting": batting_row.iloc[0] if not batting_row.empty else None,
        "bowling": bowling_row.iloc[0] if not bowling_row.empty else None,
        "fielding": fielding_row.iloc[0] if not fielding_row.empty else None,
        "season": season_rows,
    }


def get_predictor_data_summary(
    registry: pd.DataFrame,
    season: pd.DataFrame,
) -> dict[str, object]:
    """Build read-only availability metrics for the player predictor page."""
    summary: dict[str, object] = {
        "registry_ready": not registry.empty,
        "season_ready": not season.empty,
        "player_count": 0,
        "season_count": 0,
        "season_record_count": 0,
        "earliest_season": "N/A",
        "latest_season": "N/A",
    }
    if registry.empty:
        return summary

    required_registry = {"player_id", "player_name"}
    if not required_registry.issubset(registry.columns):
        return summary

    summary["player_count"] = int(registry["player_id"].nunique())

    if season.empty:
        return summary

    required_season = {
        "player_id",
        "player_name",
        "season",
        "team",
        "batting_runs",
        "bowling_wickets",
    }
    if not required_season.issubset(season.columns):
        return summary

    valid_seasons = season["season"].dropna().astype(str).str.strip()
    valid_seasons = valid_seasons[valid_seasons.ne("")]
    unique_seasons = sorted(valid_seasons.unique(), key=season_sort_key)

    summary["season_ready"] = True
    summary["season_record_count"] = int(len(season))
    summary["season_count"] = len(unique_seasons)
    summary["earliest_season"] = unique_seasons[0] if unique_seasons else "N/A"
    summary["latest_season"] = unique_seasons[-1] if unique_seasons else "N/A"
    return summary


def season_sort_key(season: str) -> tuple[int, str]:
    """Sort calendar labels and season-range labels chronologically."""
    cleaned = str(season).strip()
    first = cleaned.split("/")[0]
    try:
        return int(first), cleaned
    except ValueError:
        return 10_000, cleaned


def get_player_predictor_history(
    season: pd.DataFrame,
    player_id: str,
) -> pd.DataFrame:
    """Aggregate actual player season records without making a projection."""
    if season.empty:
        return pd.DataFrame()

    required = {
        "player_id",
        "season",
        "team",
        "matches",
        "batting_runs",
        "balls_faced",
        "batting_dismissals",
        "bowling_wickets",
        "bowling_legal_balls",
        "bowling_runs_conceded",
    }
    if not required.issubset(season.columns):
        return pd.DataFrame()

    selected = season[
        season["player_id"].astype(str).eq(str(player_id))
    ].copy()
    if selected.empty:
        return pd.DataFrame()

    numeric_columns = [
        "matches",
        "batting_runs",
        "balls_faced",
        "batting_dismissals",
        "bowling_wickets",
        "bowling_legal_balls",
        "bowling_runs_conceded",
    ]
    for column in numeric_columns:
        selected[column] = pd.to_numeric(selected[column], errors="coerce").fillna(0)

    aggregated = (
        selected.groupby("season", as_index=False)
        .agg(
            matches=("matches", "sum"),
            batting_runs=("batting_runs", "sum"),
            balls_faced=("balls_faced", "sum"),
            batting_dismissals=("batting_dismissals", "sum"),
            bowling_wickets=("bowling_wickets", "sum"),
            bowling_legal_balls=("bowling_legal_balls", "sum"),
            bowling_runs_conceded=("bowling_runs_conceded", "sum"),
        )
    )
    aggregated["batting_average"] = aggregated.apply(
        lambda row: (
            row["batting_runs"] / row["batting_dismissals"]
            if row["batting_dismissals"] > 0
            else None
        ),
        axis=1,
    )
    aggregated["strike_rate"] = aggregated.apply(
        lambda row: (
            row["batting_runs"] / row["balls_faced"] * 100
            if row["balls_faced"] > 0
            else None
        ),
        axis=1,
    )
    aggregated["bowling_economy"] = aggregated.apply(
        lambda row: (
            row["bowling_runs_conceded"]
            / (row["bowling_legal_balls"] / 6)
            if row["bowling_legal_balls"] > 0
            else None
        ),
        axis=1,
    )
    aggregated["_sort_key"] = aggregated["season"].map(season_sort_key)
    aggregated = aggregated.sort_values("_sort_key", kind="stable").drop(
        columns="_sort_key"
    )
    return aggregated


def filter_match_results(
    history: pd.DataFrame,
    *,
    search: str = "",
    team: str = "All teams",
    winner: str = "All winners",
    venue: str = "All venues",
    sort_order: str = "Most recent first",
) -> pd.DataFrame:
    """Filter and sort historical match rows for the match-results center."""
    if history.empty:
        return pd.DataFrame()

    required = {
        "normalized_team1",
        "normalized_team2",
        "normalized_winner",
        "venue",
        "parsed_date",
    }
    if not required.issubset(history.columns):
        return pd.DataFrame()

    filtered = history.copy()
    search_text = " ".join(str(search).strip().lower().split())
    if search_text:
        searchable = (
            filtered["normalized_team1"].fillna("").astype(str).str.lower()
            + " "
            + filtered["normalized_team2"].fillna("").astype(str).str.lower()
            + " "
            + filtered["normalized_winner"].fillna("").astype(str).str.lower()
            + " "
            + filtered["venue"].fillna("").astype(str).str.lower()
        )
        filtered = filtered[searchable.str.contains(search_text, regex=False)].copy()

    if team != "All teams":
        filtered = filtered[
            filtered["normalized_team1"].eq(team)
            | filtered["normalized_team2"].eq(team)
        ].copy()
    if winner != "All winners":
        filtered = filtered[
            filtered["normalized_winner"].eq(winner)
        ].copy()
    if venue != "All venues":
        filtered = filtered[filtered["venue"].eq(venue)].copy()

    if "parsed_date" in filtered.columns:
        descending = sort_order != "Oldest first"
        filtered = filtered.sort_values(
            ["parsed_date", "match_id"],
            ascending=[not descending, False],
            kind="mergesort",
            na_position="last",
        )

    return filtered.reset_index(drop=True)


def paginate_match_results(
    filtered: pd.DataFrame,
    page: int,
    page_size: int,
) -> tuple[pd.DataFrame, int, int]:
    """Return a bounded page and total page/count metadata."""
    if filtered.empty:
        return pd.DataFrame(), 0, 0

    total_pages = max(1, (len(filtered) + page_size - 1) // page_size)
    safe_page = max(1, min(int(page), total_pages))
    start = (safe_page - 1) * page_size
    end = start + page_size
    return filtered.iloc[start:end].copy(), safe_page, total_pages


def summarize_match_results(filtered: pd.DataFrame) -> dict[str, object]:
    """Calculate summary values from the currently filtered match rows."""
    if filtered.empty:
        return {
            "matches": 0,
            "teams": 0,
            "venues": 0,
            "most_frequent_winner": "N/A",
        }

    teams = set(
        filtered["normalized_team1"].dropna().astype(str).unique()
    ).union(
        set(filtered["normalized_team2"].dropna().astype(str).unique())
    )
    venues = set(
        filtered["venue"].dropna().astype(str)
        .str.strip()
        .loc[lambda values: values.ne("")]
        .unique()
    )
    winners = filtered["normalized_winner"].dropna().astype(str)
    winners = winners[winners.str.strip().ne("")]
    most_frequent = "N/A"
    if not winners.empty:
        most_frequent = str(winners.value_counts().idxmax())

    return {
        "matches": int(len(filtered)),
        "teams": len(teams),
        "venues": len(venues),
        "most_frequent_winner": most_frequent,
    }
