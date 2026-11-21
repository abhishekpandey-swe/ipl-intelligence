from __future__ import annotations

from pathlib import Path

import pandas as pd
import csv

from data_ingestion import inspect_optional_datasets
from player_intelligence import (
    CURRENT_IPL_TEAMS,
    calculate_team_comparison,
    filter_match_results,
    get_player_predictor_history,
    get_predictor_data_summary,
    optional_player_data_status,
    paginate_match_results,
    player_reference_directory,
    prepare_match_analytics_history,
    summarize_match_results,
    team_profile,
)


def test_team_profile_uses_actual_match_results() -> None:
    df = pd.DataFrame(
        [
            {
                "team1": "Team A",
                "team2": "Team B",
                "winner": "Team A",
                "season": "2024",
                "venue": "Venue X",
            },
            {
                "team1": "Team B",
                "team2": "Team A",
                "winner": "Team B",
                "season": "2024",
                "venue": "Venue X",
            },
            {
                "team1": "Team A",
                "team2": "Team C",
                "winner": None,
                "season": "2025",
                "venue": "Venue Y",
            },
        ]
    )
    profile = team_profile(df, "Team A")
    assert profile["matches"] == 2
    assert profile["wins"] == 1
    assert profile["losses"] == 1
    assert profile["win_percentage"] == 50.0


def test_player_reference_directory_is_honest() -> None:
    df = pd.DataFrame(
        [
            {"player_of_match": "245.0", "season": "2024"},
            {"player_of_match": "245.0", "season": "2025"},
            {"player_of_match": None, "season": "2025"},
        ]
    )
    result = player_reference_directory(df)
    assert len(result) == 1
    assert result.iloc[0]["reference"] == "245.0"
    assert result.iloc[0]["awards"] == 2


def test_optional_data_status_reports_missing_files() -> None:
    status = optional_player_data_status(Path("missing-data-directory"))
    assert len(status) == 4
    assert all(row["status"] == "not available" for row in status)
    report = inspect_optional_datasets(Path("missing-data-directory"))
    assert len(report) == 4


def test_current_team_count_is_ten() -> None:
    assert len(CURRENT_IPL_TEAMS) == 10


def test_player_image_mapping_has_exactly_50_unique_rows() -> None:
    csv_path = Path("data/player_images.csv")
    assert csv_path.exists()
    frame = pd.read_csv(csv_path, dtype=str).fillna("")
    assert len(frame) == 50
    assert frame["player_name"].nunique() == 50
    assert set(frame.columns) == {
        "player_name",
        "image_filename",
        "source_url",
        "verification_status",
        "image_license_or_usage_notes",
    }


def test_downloaded_player_images_exist_and_are_jpegs() -> None:
    frame = pd.read_csv("data/player_images.csv", dtype=str).fillna("")
    downloaded = frame[frame["verification_status"] == "verified_download"]
    for filename in downloaded["image_filename"]:
        path = Path("assets/players") / filename
        assert path.exists(), filename
        data = path.read_bytes()
        assert data.startswith(b"\xff\xd8") and data.endswith(b"\xff\xd9"), filename


def test_fallback_rows_are_transparent() -> None:
    frame = pd.read_csv("data/player_images.csv", dtype=str).fillna("")
    fallbacks = frame[frame["verification_status"] == "fallback_initials"]
    assert len(fallbacks) > 0
    assert (fallbacks["image_filename"] == "").all()
    assert (fallbacks["source_url"] == "").all()


def test_prepare_match_analytics_history_creates_expected_columns() -> None:
    source = pd.DataFrame(
        [
            {
                "team1": "Royal Challengers Bangalore",
                "team2": "Mumbai Indians",
                "winner": "Royal Challengers Bangalore",
                "season": "2024",
                "venue": "M Chinnaswamy Stadium",
            }
        ]
    )
    prepared = prepare_match_analytics_history(source)
    assert not prepared.empty
    assert set(
        ["normalized_team1", "normalized_team2", "normalized_winner"]
    ).issubset(prepared.columns)
    assert prepared.iloc[0]["normalized_team1"] == "Royal Challengers Bengaluru"
    assert prepared.iloc[0]["normalized_team2"] == "Mumbai Indians"
    assert prepared.iloc[0]["normalized_winner"] == "Royal Challengers Bengaluru"


def test_team_comparison_calculates_matches_wins_losses() -> None:
    history = prepare_match_analytics_history(
        pd.DataFrame(
            [
                {
                    "team1": "Team A",
                    "team2": "Team B",
                    "winner": "Team A",
                    "season": "2024",
                    "venue": "Venue X",
                },
                {
                    "team1": "Team B",
                    "team2": "Team A",
                    "winner": "Team B",
                    "season": "2024",
                    "venue": "Venue X",
                },
            ]
        )
    )
    comparison = calculate_team_comparison(history, ["Team A", "Team B"])
    team_a = comparison[comparison["Team"].eq("Team A")].iloc[0]
    team_b = comparison[comparison["Team"].eq("Team B")].iloc[0]
    assert team_a["Matches"] == 2
    assert team_a["Wins"] == 1
    assert team_a["Losses"] == 1
    assert team_b["Matches"] == 2
    assert team_b["Wins"] == 1
    assert team_b["Losses"] == 1


def test_empty_match_history_is_handled() -> None:
    prepared = prepare_match_analytics_history(pd.DataFrame())
    assert prepared.empty
    comparison = calculate_team_comparison(prepared, ["Team A"])
    assert comparison.empty


def test_missing_required_analytics_columns_raise_clear_error() -> None:
    try:
        prepare_match_analytics_history(
            pd.DataFrame({"team1": ["Team A"], "team2": ["Team B"]})
        )
        raise AssertionError("Expected missing analytics column validation.")
    except ValueError as exc:
        assert "required analytics columns" in str(exc)


def test_predictor_summary_uses_real_processed_schema() -> None:
    registry = pd.read_csv("data/processed/player_registry.csv")
    season = pd.read_csv("data/processed/player_season_stats.csv")
    summary = get_predictor_data_summary(registry, season)
    assert summary["registry_ready"] is True
    assert summary["season_ready"] is True
    assert summary["player_count"] == 815
    assert summary["season_record_count"] == 3387
    assert summary["season_count"] > 0
    assert summary["earliest_season"] != "N/A"
    assert summary["latest_season"] == "2026"


def test_predictor_history_missing_data_is_empty() -> None:
    summary = get_predictor_data_summary(
        pd.DataFrame(),
        pd.DataFrame(),
    )
    assert summary["registry_ready"] is False
    assert summary["season_ready"] is False
    assert get_player_predictor_history(
        pd.DataFrame(),
        "unknown-player",
    ).empty


def test_predictor_history_aggregates_actual_seasons() -> None:
    season = pd.DataFrame(
        [
            {
                "player_id": "P1",
                "season": "2024",
                "team": "Team A",
                "matches": 10,
                "batting_runs": 300,
                "balls_faced": 200,
                "batting_dismissals": 8,
                "bowling_wickets": 5,
                "bowling_legal_balls": 120,
                "bowling_runs_conceded": 180,
            },
            {
                "player_id": "P1",
                "season": "2024",
                "team": "Team B",
                "matches": 2,
                "batting_runs": 100,
                "balls_faced": 80,
                "batting_dismissals": 2,
                "bowling_wickets": 1,
                "bowling_legal_balls": 24,
                "bowling_runs_conceded": 40,
            },
            {
                "player_id": "P2",
                "season": "2024",
                "team": "Team C",
                "matches": 5,
                "batting_runs": 50,
                "balls_faced": 40,
                "batting_dismissals": 2,
                "bowling_wickets": 0,
                "bowling_legal_balls": 0,
                "bowling_runs_conceded": 0,
            },
        ]
    )
    history = get_player_predictor_history(season, "P1")
    assert len(history) == 1
    row = history.iloc[0]
    assert row["matches"] == 12
    assert row["batting_runs"] == 400
    assert row["bowling_wickets"] == 6
    assert row["batting_average"] == 40
    assert abs(row["strike_rate"] - (400 / 280 * 100)) < 1e-9


def test_match_result_search_and_filters() -> None:
    source = pd.DataFrame(
        [
            {
                "normalized_team1": "Mumbai Indians",
                "normalized_team2": "Chennai Super Kings",
                "normalized_winner": "Mumbai Indians",
                "venue": "Wankhede Stadium",
                "match_id": 101,
                "parsed_date": pd.Timestamp("2024-04-01"),
            },
            {
                "normalized_team1": "Chennai Super Kings",
                "normalized_team2": "Royal Challengers Bengaluru",
                "normalized_winner": "Royal Challengers Bengaluru",
                "venue": "M Chinnaswamy Stadium",
                "match_id": 102,
                "parsed_date": pd.Timestamp("2024-04-02"),
            },
        ]
    )
    filtered = filter_match_results(
        source,
        search="wankhede",
    )
    assert len(filtered) == 1
    assert filtered.iloc[0]["match_id"] == 101

    filtered = filter_match_results(
        source,
        team="Chennai Super Kings",
        winner="Royal Challengers Bengaluru",
        venue="M Chinnaswamy Stadium",
    )
    assert len(filtered) == 1
    assert filtered.iloc[0]["match_id"] == 102


def test_match_result_sorting_and_pagination() -> None:
    source = pd.DataFrame(
        [
            {
                "normalized_team1": "Team A",
                "normalized_team2": "Team B",
                "normalized_winner": "Team A",
                "venue": "Venue X",
                "match_id": 1,
                "parsed_date": pd.Timestamp("2024-01-01"),
            },
            {
                "normalized_team1": "Team C",
                "normalized_team2": "Team D",
                "normalized_winner": "Team D",
                "venue": "Venue Y",
                "match_id": 2,
                "parsed_date": pd.Timestamp("2024-02-01"),
            },
            {
                "normalized_team1": "Team E",
                "normalized_team2": "Team F",
                "normalized_winner": "Team E",
                "venue": "Venue Z",
                "match_id": 3,
                "parsed_date": pd.Timestamp("2024-03-01"),
            },
        ]
    )
    recent = filter_match_results(source, sort_order="Most recent first")
    oldest = filter_match_results(source, sort_order="Oldest first")
    assert recent.iloc[0]["match_id"] == 3
    assert oldest.iloc[0]["match_id"] == 1

    page, safe_page, total_pages = paginate_match_results(recent, 2, 1)
    assert safe_page == 2
    assert total_pages == 3
    assert page.iloc[0]["match_id"] == 2


def test_match_result_summary() -> None:
    source = pd.DataFrame(
        [
            {
                "normalized_team1": "Team A",
                "normalized_team2": "Team B",
                "normalized_winner": "Team A",
                "venue": "Venue X",
                "match_id": 1,
                "parsed_date": pd.Timestamp("2024-01-01"),
            },
            {
                "normalized_team1": "Team A",
                "normalized_team2": "Team C",
                "normalized_winner": "Team A",
                "venue": "Venue Y",
                "match_id": 2,
                "parsed_date": pd.Timestamp("2024-01-02"),
            },
        ]
    )
    summary = summarize_match_results(source)
    assert summary["matches"] == 2
    assert summary["teams"] == 3
    assert summary["venues"] == 2
    assert summary["most_frequent_winner"] == "Team A"


if __name__ == "__main__":
    test_team_profile_uses_actual_match_results()
    test_player_reference_directory_is_honest()
    test_optional_data_status_reports_missing_files()
    test_current_team_count_is_ten()
    test_player_image_mapping_has_exactly_50_unique_rows()
    test_downloaded_player_images_exist_and_are_jpegs()
    test_fallback_rows_are_transparent()
    test_prepare_match_analytics_history_creates_expected_columns()
    test_team_comparison_calculates_matches_wins_losses()
    test_empty_match_history_is_handled()
    test_missing_required_analytics_columns_raise_clear_error()
    test_predictor_summary_uses_real_processed_schema()
    test_predictor_history_missing_data_is_empty()
    test_predictor_history_aggregates_actual_seasons()
    test_match_result_search_and_filters()
    test_match_result_sorting_and_pagination()
    test_match_result_summary()
    print("PASS: Platform data tests succeeded.")
