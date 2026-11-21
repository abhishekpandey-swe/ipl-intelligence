from __future__ import annotations

import copy

import pandas as pd

from feature_engineering import (
    FEATURE_NAMES,
    FeatureState,
    is_completed_for_history,
    normalize_team_name,
    validate_training_record,
)


def test_same_date_state_is_updated_only_after_generation() -> None:
    state = FeatureState()
    before = copy.deepcopy(state.team_history)

    match1 = {
        "match_date": "01-01-2024",
        "result": "win",
        "team1": "Team A",
        "team2": "Team B",
        "winner": "Team A",
        "venue": "Test Stadium",
        "toss_winner": "Team A",
        "toss_decision": "bat",
    }

    features = state.generate_features("Team A", "Team B", "Test Stadium")
    assert state.team_history == before
    assert features["batting_team_win_rate_5"] == 0.5

    next_date = pd.Timestamp("2024-01-02")
    state.update_from_completed_match(match1, before_date=next_date)
    assert state.team_history["Team A"][-1] == 1
    assert state.team_history["Team B"][-1] == 0


def test_strict_cutoff_rejects_equal_and_later_matches() -> None:
    state = FeatureState()
    cutoff = pd.Timestamp("2024-01-10")

    equal = {
        "match_date": "10-01-2024",
        "result": "win",
        "team1": "Team A",
        "team2": "Team B",
        "winner": "Team A",
        "venue": "Test Stadium",
    }
    later = {
        "match_date": "11-01-2024",
        "result": "win",
        "team1": "Team A",
        "team2": "Team B",
        "winner": "Team A",
        "venue": "Test Stadium",
    }

    state.update_from_completed_match(equal, cutoff)
    state.update_from_completed_match(later, cutoff)
    assert state.team_history == {}
    assert state.head_to_head_history == {}
    assert state.team_venue_history == {}


def test_three_date_buckets() -> None:
    state = FeatureState()

    date1 = pd.Timestamp("2024-01-01")
    date2 = pd.Timestamp("2024-01-02")
    date3 = pd.Timestamp("2024-01-03")

    match1 = {
        "match_date": "01-01-2024",
        "result": "win",
        "team1": "Team A",
        "team2": "Team B",
        "winner": "Team A",
        "venue": "Test Stadium",
    }
    match2 = {
        "match_date": "02-01-2024",
        "result": "win",
        "team1": "Team A",
        "team2": "Team C",
        "winner": "Team A",
        "venue": "Test Stadium",
    }
    match3 = {
        "match_date": "03-01-2024",
        "result": "win",
        "team1": "Team A",
        "team2": "Team B",
        "winner": "Team B",
        "venue": "Test Stadium",
    }

    assert state.generate_features("Team A", "Team B", "Test Stadium")["batting_team_win_rate_5"] == 0.5
    state.update_from_completed_match(match1, before_date=date2)

    features2 = state.generate_features("Team A", "Team C", "Test Stadium")
    assert features2["batting_team_win_rate_5"] == 1.0
    assert features2["batting_team_completed_matches_5"] == 1
    state.update_from_completed_match(match2, before_date=date3)

    features3 = state.generate_features("Team A", "Team B", "Test Stadium")
    assert features3["batting_team_win_rate_5"] == 1.0
    assert features3["batting_team_completed_matches_5"] == 2
    state.update_from_completed_match(match3, before_date=date3 + pd.Timedelta(days=1))


def test_head_to_head_perspective_reversal() -> None:
    state = FeatureState()

    for winner in ("Team A", "Team A", "Team B"):
        state.update_from_completed_match(
            {
                "match_date": "01-01-2024",
                "result": "win",
                "team1": "Team A",
                "team2": "Team B",
                "winner": winner,
                "venue": "Test Stadium",
            },
            before_date=pd.Timestamp("2024-01-02"),
        )

    ab = state.generate_features("Team A", "Team B", "Test Stadium")
    ba = state.generate_features("Team B", "Team A", "Test Stadium")
    assert ab["batting_team_head_to_head_rate"] == 2 / 3
    assert ba["batting_team_head_to_head_rate"] == 1 / 3
    assert ab["head_to_head_completed_match_count"] == 3
    assert ba["head_to_head_completed_match_count"] == 3


def test_history_valid_but_missing_toss_is_not_training_record() -> None:
    match = {
        "match_date": "01-01-2024",
        "result": "win",
        "team1": "Team A",
        "team2": "Team B",
        "winner": "Team A",
        "venue": "Test Stadium",
        "toss_winner": None,
        "toss_decision": None,
    }

    assert is_completed_for_history(match) is not None
    assert validate_training_record(match) is None


def test_invalid_history_records_do_not_update_state() -> None:
    state = FeatureState()
    cutoff = pd.Timestamp("2024-01-10")
    invalid_records = [
        {"match_date": "01-01-2024", "result": "no result", "team1": "A", "team2": "B", "winner": "A", "venue": "X"},
        {"match_date": "01-01-2024", "result": "abandoned", "team1": "A", "team2": "B", "winner": "A", "venue": "X"},
        {"match_date": "bad-date", "result": "win", "team1": "A", "team2": "B", "winner": "A", "venue": "X"},
        {"match_date": "01-01-2024", "result": "win", "team1": "A", "team2": "A", "winner": "A", "venue": "X"},
        {"match_date": "01-01-2024", "result": "win", "team1": "A", "team2": "B", "winner": "C", "venue": "X"},
        {"match_date": "01-01-2024", "result": "unknown", "team1": "A", "team2": "B", "winner": "A", "venue": "X"},
    ]

    for record in invalid_records:
        state.update_from_completed_match(record, cutoff)

    assert state.team_history == {}
    assert state.head_to_head_history == {}
    assert state.team_venue_history == {}


def test_generate_features_is_state_immutable() -> None:
    state = FeatureState()
    before = copy.deepcopy(
        (
            state.team_history,
            state.head_to_head_history,
            state.team_venue_history,
        )
    )
    state.generate_features("Team A", "Team B", "Unknown Stadium")
    after = (
        state.team_history,
        state.head_to_head_history,
        state.team_venue_history,
    )
    assert before == after


def test_team_normalization() -> None:
    assert normalize_team_name("Kings XI Punjab") == "Punjab Kings"
    assert normalize_team_name("Delhi Daredevils") == "Delhi Capitals"


if __name__ == "__main__":
    test_same_date_state_is_updated_only_after_generation()
    test_strict_cutoff_rejects_equal_and_later_matches()
    test_three_date_buckets()
    test_head_to_head_perspective_reversal()
    test_history_valid_but_missing_toss_is_not_training_record()
    test_invalid_history_records_do_not_update_state()
    test_generate_features_is_state_immutable()
    test_team_normalization()
    print("PASS: All feature engineering tests succeeded.")
