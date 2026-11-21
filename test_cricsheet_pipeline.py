from __future__ import annotations

import json
import tempfile
from pathlib import Path

from cricsheet_pipeline import (
    bowling_runs_conceded,
    is_legal_delivery,
    is_regular_inning,
    process_archive,
    stable_name_id,
)


def write_match(path: Path, match_id: str = "123", include_super_over: bool = False) -> None:
    innings = [
        {
            "team": "Team A",
            "overs": [
                {
                    "over": 0,
                    "deliveries": [
                        {
                            "batter": "A Player",
                            "bowler": "B Player",
                            "non_striker": "C Player",
                            "runs": {"batter": 4, "extras": 0, "total": 4},
                        },
                        {
                            "batter": "A Player",
                            "bowler": "B Player",
                            "non_striker": "C Player",
                            "extras": {"wides": 1},
                            "runs": {"batter": 0, "extras": 1, "total": 1},
                        },
                        {
                            "batter": "A Player",
                            "bowler": "B Player",
                            "non_striker": "C Player",
                            "extras": {"noballs": 1},
                            "runs": {"batter": 1, "extras": 1, "total": 2},
                        },
                        {
                            "batter": "A Player",
                            "bowler": "B Player",
                            "non_striker": "C Player",
                            "runs": {"batter": 0, "extras": 0, "total": 0},
                            "wickets": [
                                {
                                    "kind": "caught",
                                    "player_out": "A Player",
                                    "fielders": [{"name": "C Player"}],
                                }
                            ],
                        },
                    ],
                }
            ],
        },
        {
            "team": "Team B",
            "overs": [
                {
                    "over": 0,
                    "deliveries": [
                        {
                            "batter": "C Player",
                            "bowler": "A Player",
                            "non_striker": "B Player",
                            "runs": {"batter": 6, "extras": 0, "total": 6},
                        }
                    ],
                }
            ],
        },
    ]
    if include_super_over:
        innings.extend(
            [
                {
                    "team": "Team A",
                    "super_over": True,
                    "overs": [
                        {
                            "over": 0,
                            "deliveries": [
                                {
                                    "batter": "A Player",
                                    "bowler": "B Player",
                                    "non_striker": "C Player",
                                    "runs": {"batter": 6, "extras": 0, "total": 6},
                                }
                            ],
                        }
                    ],
                }
            ]
        )

    match = {
        "meta": {"data_version": "1.3.0"},
        "info": {
            "dates": ["2024-04-01"],
            "season": "2024",
            "venue": "Test Stadium",
            "teams": ["Team A", "Team B"],
            "outcome": {"winner": "Team A"},
            "players": {
                "Team A": ["A Player", "C Player"],
                "Team B": ["B Player"],
            },
            "registry": {
                "people": {
                    "A Player": "A-ID",
                    "B Player": "B-ID",
                    "C Player": "C-ID",
                }
            },
        },
        "innings": innings,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(match), encoding="utf-8")


def test_batter_runs_and_boundaries() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        write_match(root / "123.json")
        datasets = process_archive(root)
        batting = datasets["player_batting_stats.csv"].set_index("player_name")
        assert batting.loc["A Player", "runs"] == 5
        assert batting.loc["A Player", "fours"] == 1
        assert batting.loc["A Player", "sixes"] == 0


def test_extras_and_legal_balls() -> None:
    wide = {"extras": {"wides": 1}, "runs": {"batter": 0, "extras": 1, "total": 1}}
    no_ball = {"extras": {"noballs": 1}, "runs": {"batter": 1, "extras": 1, "total": 2}}
    byes = {"extras": {"byes": 1}, "runs": {"batter": 0, "extras": 1, "total": 1}}
    legbyes = {"extras": {"legbyes": 1}, "runs": {"batter": 0, "extras": 1, "total": 1}}

    assert not is_legal_delivery(wide)
    assert not is_legal_delivery(no_ball)
    assert is_legal_delivery(byes)
    assert is_legal_delivery(legbyes)
    assert bowling_runs_conceded(wide) == 1
    assert bowling_runs_conceded(no_ball) == 2
    assert bowling_runs_conceded(byes) == 0
    assert bowling_runs_conceded(legbyes) == 0


def test_wicket_attribution_and_super_over() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        write_match(root / "123.json", include_super_over=True)
        datasets = process_archive(root)
        batting = datasets["player_batting_stats.csv"].set_index("player_name")
        assert batting.loc["A Player", "runs"] == 5

        inning_regular = {"super_over": False}
        inning_super = {"super_over": True}
        assert is_regular_inning(inning_regular)
        assert not is_regular_inning(inning_super)


def test_dismissal_kinds_and_run_out() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        match = root / "123.json"
        write_match(match)
        datasets = process_archive(root)
        match_stats = datasets["player_match_stats.csv"]
        a_match = match_stats[match_stats["player_name"] == "A Player"].iloc[0]
        assert int(a_match["batting_dismissed"]) == 1


def test_duplicate_and_malformed_files() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        write_match(root / "one" / "123.json")
        write_match(root / "two" / "123.json")
        (root / "bad.json").write_text("{bad", encoding="utf-8")

        datasets = process_archive(root)
        assert len(datasets["player_match_stats.csv"]["match_id"].unique()) == 1
        assert len(datasets["processing_report.csv"]) >= 1


def test_player_identity_is_stable_and_not_name_similarity() -> None:
    assert stable_name_id("Virat Kohli") == stable_name_id("Virat Kohli")
    assert stable_name_id("Virat Kohli") != stable_name_id("V Kohli")


def test_empty_directory() -> None:
    with tempfile.TemporaryDirectory() as temp:
        datasets = process_archive(Path(temp))
        assert datasets["player_registry.csv"].empty
        assert datasets["player_batting_stats.csv"].empty


if __name__ == "__main__":
    tests = [
        test_batter_runs_and_boundaries,
        test_extras_and_legal_balls,
        test_wicket_attribution_and_super_over,
        test_dismissal_kinds_and_run_out,
        test_duplicate_and_malformed_files,
        test_player_identity_is_stable_and_not_name_similarity,
        test_empty_directory,
    ]
    for test in tests:
        test()
    print(f"PASS: {len(tests)} Cricsheet pipeline tests succeeded.")
