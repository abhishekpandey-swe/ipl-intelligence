from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


TEAM_NAME_MAP = {
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
    "Kings XI Punjab": "Punjab Kings",
    "Delhi Daredevils": "Delhi Capitals",
    "Rising Pune Supergiant": "Rising Pune Supergiants",
    "Pune Warriors": "Pune Warriors India",
}

BOWLER_CREDITED_WICKET_KINDS = {
    "bowled",
    "caught",
    "caught and bowled",
    "lbw",
    "stumped",
    "hit wicket",
}

BATTING_DISMISSAL_KINDS = {
    "bowled",
    "caught",
    "caught and bowled",
    "lbw",
    "stumped",
    "hit wicket",
    "run out",
    "obstructing the field",
    "timed out",
    "retired out",
}


def clean_text(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.strip().split())


def normalize_team_name(value: object) -> str:
    cleaned = clean_text(value)
    return TEAM_NAME_MAP.get(cleaned, cleaned)


def normalize_person_name(value: object) -> str:
    return clean_text(value)


def stable_name_id(name: str) -> str:
    """Create an identity only when Cricsheet registry data does not provide an ID."""
    digest = hashlib.sha1(name.casefold().encode("utf-8")).hexdigest()[:12]
    return f"name-{digest}"


def parse_match_date(info: dict[str, Any]) -> pd.Timestamp:
    dates = info.get("dates")
    if not isinstance(dates, list) or not dates:
        return pd.NaT
    return pd.to_datetime(dates[0], errors="coerce")


def bowling_runs_conceded(delivery: dict[str, Any]) -> int:
    """Runs charged to the bowler, excluding byes, leg-byes and penalty extras."""
    runs = delivery.get("runs") or {}
    extras = delivery.get("extras") or {}
    batter_runs = int(runs.get("batter", 0) or 0)
    wides = int(extras.get("wides", 0) or 0)
    no_balls = int(extras.get("noballs", 0) or 0)
    return batter_runs + wides + no_balls


def is_legal_delivery(delivery: dict[str, Any]) -> bool:
    extras = delivery.get("extras") or {}
    return not (
        int(extras.get("wides", 0) or 0) > 0
        or int(extras.get("noballs", 0) or 0) > 0
    )


def is_regular_inning(inning: dict[str, Any]) -> bool:
    return inning.get("super_over") is not True


def get_registry_id(
    person_name: str,
    people_registry: dict[str, str],
) -> str:
    if person_name in people_registry:
        return str(people_registry[person_name])
    return stable_name_id(person_name)


def _match_is_valid(match: dict[str, Any], match_id: str) -> bool:
    info = match.get("info")
    if not isinstance(info, dict):
        return False
    if not isinstance(match.get("innings"), list):
        return False
    if pd.isna(parse_match_date(info)):
        return False
    teams = info.get("teams")
    if not isinstance(teams, list) or len(teams) != 2:
        return False
    if not info.get("season"):
        return False
    if not info.get("venue"):
        return False
    outcome = info.get("outcome") or {}
    if outcome.get("result") == "no result":
        return False
    if outcome.get("result") not in {"tie"} and not outcome.get("winner") and not outcome.get("eliminator"):
        return False
    if not match_id:
        return False
    return True


def process_archive(
    raw_dir: Path,
) -> dict[str, pd.DataFrame]:
    raw_files = sorted(raw_dir.rglob("*.json"))
    selected_matches: dict[str, tuple[dict[str, Any], Path, int]] = {}
    malformed: list[dict[str, str]] = []

    for path in raw_files:
        match_id = path.stem
        try:
            with path.open("r", encoding="utf-8") as handle:
                match = json.load(handle)
        except (json.JSONDecodeError, UnicodeDecodeError, OSError) as exc:
            malformed.append(
                {
                    "match_id": match_id,
                    "filename": path.name,
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )
            continue

        if not isinstance(match, dict):
            malformed.append(
                {
                    "match_id": match_id,
                    "filename": path.name,
                    "reason": "top-level JSON value is not an object",
                }
            )
            continue

        if not _match_is_valid(match, match_id):
            malformed.append(
                {
                    "match_id": match_id,
                    "filename": path.name,
                    "reason": "invalid or incomplete match metadata/result",
                }
            )
            continue

        innings = match.get("innings") or []
        regular_deliveries = 0
        for inning in innings:
            if not is_regular_inning(inning):
                continue
            regular_deliveries += sum(
                len(over.get("deliveries", []) or [])
                for over in inning.get("overs", []) or []
            )

        current = selected_matches.get(match_id)
        if current is None or regular_deliveries > current[2]:
            selected_matches[match_id] = (match, path, regular_deliveries)

    registry_names: dict[str, set[str]] = defaultdict(set)
    registry_primary_names: dict[str, str] = {}
    match_rows: list[dict[str, Any]] = []
    batting_rows: list[dict[str, Any]] = []
    bowling_rows: list[dict[str, Any]] = []
    fielding_rows: list[dict[str, Any]] = []
    squad_rows: list[dict[str, Any]] = []

    batting_innings: dict[str, dict[str, Any]] = {}
    bowling_innings: dict[str, dict[str, Any]] = {}
    fielding_totals: dict[str, dict[str, Any]] = {}
    match_registry: dict[str, dict[str, Any]] = {}
    team_seasons: dict[tuple[str, str, str], dict[str, Any]] = {}

    processed_match_ids: set[str] = set()

    for match_id, (match, path, regular_deliveries) in sorted(
        selected_matches.items(),
        key=lambda item: item[0],
    ):
        info = match["info"]
        match_date = parse_match_date(info)
        season = clean_text(str(info.get("season")))
        venue = clean_text(info.get("venue"))
        teams = [normalize_team_name(team) for team in info.get("teams", [])]
        people_registry = (
            info.get("registry", {}).get("people", {})
            if isinstance(info.get("registry"), dict)
            else {}
        )

        player_team_by_match: dict[str, set[str]] = {}
        players = info.get("players") or {}
        if isinstance(players, dict):
            for team_name, names in players.items():
                normalized_team = normalize_team_name(team_name)
                if not isinstance(names, list):
                    continue
                for name in names:
                    clean_name = normalize_person_name(name)
                    if not clean_name:
                        continue
                    player_id = get_registry_id(clean_name, people_registry)
                    player_team_by_match.setdefault(player_id, set()).add(normalized_team)

                    registry_names[player_id].add(clean_name)
                    registry_primary_names.setdefault(player_id, clean_name)

                    key = (player_id, season, normalized_team)
                    team_record = team_seasons.setdefault(
                        key,
                        {
                            "player_id": player_id,
                            "player_name": clean_name,
                            "season": season,
                            "team": normalized_team,
                            "matches": 0,
                        },
                    )
                    if clean_name and team_record["player_name"] == "":
                        team_record["player_name"] = clean_name

        for match_team in player_team_by_match:
            key_team_set = player_team_by_match[match_team]
            for team_name in key_team_set:
                key = (match_team, season, team_name)
                if key in team_seasons:
                    team_seasons[key]["matches"] += 1

        outcome = info.get("outcome") or {}
        eliminator = clean_text(outcome.get("eliminator"))
        winner = clean_text(outcome.get("winner"))
        player_of_match_names = [
            clean_text(name)
            for name in info.get("player_of_match", []) or []
        ]

        match_per_player: dict[str, dict[str, Any]] = {}

        for player_id in player_team_by_match:
            match_record = {
                "match_id": match_id,
                "match_date": match_date,
                "season": season,
                "team": next(iter(player_team_by_match[player_id]), ""),
                "player_id": player_id,
                "player_name": registry_primary_names.get(player_id, ""),
                "batting_runs": 0,
                "balls_faced": 0,
                "batting_dismissed": 0,
                "batting_not_out": 0,
                "fours": 0,
                "sixes": 0,
                "highest_score": 0,
                "bowling_balls": 0,
                "bowling_runs_conceded": 0,
                "bowling_wickets": 0,
                "catches": 0,
                "stumpings": 0,
                "run_outs": 0,
                "player_of_match": int(
                    any(
                        get_registry_id(name, people_registry) == player_id
                        for name in player_of_match_names
                    )
                ),
            }
            match_per_player[player_id] = match_record

        innings_index = 0
        match_maiden_bowlers: set[str] = set()
        over_bowler_stats: dict[tuple[str, int, int, str], dict[str, int]] = {}
        for inning in match.get("innings", []) or []:
            if not is_regular_inning(inning):
                continue
            innings_index += 1
            batting_team = normalize_team_name(inning.get("team"))
            batting_inning_players: set[str] = set()
            dismissal_by_batter: dict[str, str] = {}
            innings_runs: dict[str, int] = defaultdict(int)
            innings_balls: dict[str, int] = defaultdict(int)

            for over in inning.get("overs", []) or []:
                over_number = int(over.get("over", 0) or 0)
                for delivery in over.get("deliveries", []) or []:
                    batter = normalize_person_name(delivery.get("batter"))
                    bowler = normalize_person_name(delivery.get("bowler"))
                    non_striker = normalize_person_name(delivery.get("non_striker"))
                    if batter:
                        batter_id = get_registry_id(batter, people_registry)
                        registry_names[batter_id].add(batter)
                        registry_primary_names.setdefault(batter_id, batter)
                        batting_inning_players.add(batter_id)
                        batter_runs = int(
                            (delivery.get("runs") or {}).get("batter", 0) or 0
                        )
                        innings_runs[batter_id] += batter_runs
                        if not int((delivery.get("extras") or {}).get("wides", 0) or 0):
                            innings_balls[batter_id] += 1

                        record = match_per_player.setdefault(
                            batter_id,
                            {
                                "match_id": match_id,
                                "match_date": match_date,
                                "season": season,
                                "team": batting_team,
                                "player_id": batter_id,
                                "player_name": registry_primary_names.get(batter_id, batter),
                                "batting_runs": 0,
                                "balls_faced": 0,
                                "batting_dismissed": 0,
                                "batting_not_out": 0,
                                "fours": 0,
                                "sixes": 0,
                                "highest_score": 0,
                                "bowling_balls": 0,
                                "bowling_runs_conceded": 0,
                                "bowling_wickets": 0,
                                "catches": 0,
                                "stumpings": 0,
                                "run_outs": 0,
                                "player_of_match": int(
                                    any(
                                        get_registry_id(name, people_registry) == batter_id
                                        for name in player_of_match_names
                                    )
                                ),
                            },
                        )
                        record["team"] = record["team"] or batting_team
                        record["batting_runs"] += batter_runs
                        record["balls_faced"] += (
                            0
                            if int((delivery.get("extras") or {}).get("wides", 0) or 0)
                            else 1
                        )
                        if batter_runs == 4:
                            record["fours"] += 1
                        if batter_runs == 6:
                            record["sixes"] += 1

                    if bowler:
                        bowler_id = get_registry_id(bowler, people_registry)
                        registry_names[bowler_id].add(bowler)
                        registry_primary_names.setdefault(bowler_id, bowler)

                        record = match_per_player.setdefault(
                            bowler_id,
                            {
                                "match_id": match_id,
                                "match_date": match_date,
                                "season": season,
                                "team": next(
                                    (
                                        team
                                        for team in teams
                                        if team != batting_team
                                    ),
                                    "",
                                ),
                                "player_id": bowler_id,
                                "player_name": registry_primary_names.get(bowler_id, bowler),
                                "batting_runs": 0,
                                "balls_faced": 0,
                                "batting_dismissed": 0,
                                "batting_not_out": 0,
                                "fours": 0,
                                "sixes": 0,
                                "highest_score": 0,
                                "bowling_balls": 0,
                                "bowling_runs_conceded": 0,
                                "bowling_wickets": 0,
                                "catches": 0,
                                "stumpings": 0,
                                "run_outs": 0,
                                "player_of_match": int(
                                    any(
                                        get_registry_id(name, people_registry) == bowler_id
                                        for name in player_of_match_names
                                    )
                                ),
                            },
                        )
                        if is_legal_delivery(delivery):
                            record["bowling_balls"] += 1
                        record["bowling_runs_conceded"] += bowling_runs_conceded(delivery)

                        key = (match_id, innings_index, over_number, bowler_id)
                        over_record = over_bowler_stats.setdefault(
                            key,
                            {"legal_balls": 0, "runs_conceded": 0},
                        )
                        if is_legal_delivery(delivery):
                            over_record["legal_balls"] += 1
                        over_record["runs_conceded"] += bowling_runs_conceded(delivery)

                    for wicket in delivery.get("wickets", []) or []:
                        kind = clean_text(wicket.get("kind")).lower()
                        player_out = normalize_person_name(wicket.get("player_out"))
                        if player_out:
                            player_out_id = get_registry_id(player_out, people_registry)
                            registry_names[player_out_id].add(player_out)
                            registry_primary_names.setdefault(player_out_id, player_out)

                            if kind in BATTING_DISMISSAL_KINDS:
                                dismissal_by_batter[player_out_id] = kind

                            record = match_per_player.setdefault(
                                player_out_id,
                                {
                                    "match_id": match_id,
                                    "match_date": match_date,
                                    "season": season,
                                    "team": batting_team,
                                    "player_id": player_out_id,
                                    "player_name": registry_primary_names.get(player_out_id, player_out),
                                    "batting_runs": 0,
                                    "balls_faced": 0,
                                    "batting_dismissed": 0,
                                    "batting_not_out": 0,
                                    "fours": 0,
                                    "sixes": 0,
                                    "highest_score": 0,
                                    "bowling_balls": 0,
                                    "bowling_runs_conceded": 0,
                                    "bowling_wickets": 0,
                                    "catches": 0,
                                    "stumpings": 0,
                                    "run_outs": 0,
                                    "player_of_match": 0,
                                },
                            )
                            record["team"] = record["team"] or batting_team

                        if bowler and kind in BOWLER_CREDITED_WICKET_KINDS:
                            bowler_id = get_registry_id(bowler, people_registry)
                            record = match_per_player.setdefault(
                                bowler_id,
                                {
                                    "match_id": match_id,
                                    "match_date": match_date,
                                    "season": season,
                                    "team": next(
                                        (
                                            team
                                            for team in teams
                                            if team != batting_team
                                        ),
                                        "",
                                    ),
                                    "player_id": bowler_id,
                                    "player_name": registry_primary_names.get(bowler_id, bowler),
                                    "batting_runs": 0,
                                    "balls_faced": 0,
                                    "batting_dismissed": 0,
                                    "batting_not_out": 0,
                                    "fours": 0,
                                    "sixes": 0,
                                    "highest_score": 0,
                                    "bowling_balls": 0,
                                    "bowling_runs_conceded": 0,
                                    "bowling_wickets": 0,
                                    "catches": 0,
                                    "stumpings": 0,
                                    "run_outs": 0,
                                    "player_of_match": 0,
                                },
                            )
                            record["bowling_wickets"] += 1

                        kind = clean_text(wicket.get("kind")).lower()
                        fielders = wicket.get("fielders") or []
                        for fielder in fielders:
                            fielder_name = normalize_person_name(
                                fielder.get("name") if isinstance(fielder, dict) else fielder
                            )
                            if not fielder_name:
                                continue
                            fielder_id = get_registry_id(fielder_name, people_registry)
                            registry_names[fielder_id].add(fielder_name)
                            registry_primary_names.setdefault(fielder_id, fielder_name)
                            record = match_per_player.setdefault(
                                fielder_id,
                                {
                                    "match_id": match_id,
                                    "match_date": match_date,
                                    "season": season,
                                    "team": next(
                                        (
                                            team
                                            for team in teams
                                            if team != batting_team
                                        ),
                                        "",
                                    ),
                                    "player_id": fielder_id,
                                    "player_name": registry_primary_names.get(fielder_id, fielder_name),
                                    "batting_runs": 0,
                                    "balls_faced": 0,
                                    "batting_dismissed": 0,
                                    "batting_not_out": 0,
                                    "fours": 0,
                                    "sixes": 0,
                                    "highest_score": 0,
                                    "bowling_balls": 0,
                                    "bowling_runs_conceded": 0,
                                    "bowling_wickets": 0,
                                    "catches": 0,
                                    "stumpings": 0,
                                    "run_outs": 0,
                                    "player_of_match": 0,
                                },
                            )
                            if kind == "caught":
                                record["catches"] += 1
                            elif kind == "stumped":
                                record["stumpings"] += 1
                            elif kind == "run out":
                                record["run_outs"] += 1

                        if kind == "caught and bowled" and bowler:
                            bowler_id = get_registry_id(bowler, people_registry)
                            record = match_per_player.setdefault(
                                bowler_id,
                                {
                                    "match_id": match_id,
                                    "match_date": match_date,
                                    "season": season,
                                    "team": next(
                                        (
                                            team
                                            for team in teams
                                            if team != batting_team
                                        ),
                                        "",
                                    ),
                                    "player_id": bowler_id,
                                    "player_name": registry_primary_names.get(bowler_id, bowler),
                                    "batting_runs": 0,
                                    "balls_faced": 0,
                                    "batting_dismissed": 0,
                                    "batting_not_out": 0,
                                    "fours": 0,
                                    "sixes": 0,
                                    "highest_score": 0,
                                    "bowling_balls": 0,
                                    "bowling_runs_conceded": 0,
                                    "bowling_wickets": 0,
                                    "catches": 0,
                                    "stumpings": 0,
                                    "run_outs": 0,
                                    "player_of_match": 0,
                                },
                            )
                            record["catches"] += 1

            for player_id in batting_inning_players:
                record = match_per_player.get(player_id)
                if record is None:
                    continue
                record["highest_score"] = innings_runs[player_id]
                kind = dismissal_by_batter.get(player_id)
                if kind is None or kind == "retired hurt":
                    record["batting_not_out"] = 1
                elif kind in BATTING_DISMISSAL_KINDS:
                    record["batting_dismissed"] = 1

            for (match_key, inning_key, over_key, bowler_id), over_record in over_bowler_stats.items():
                if match_key != match_id or inning_key != innings_index:
                    continue
                if (
                    over_record["legal_balls"] == 6
                    and over_record["runs_conceded"] == 0
                ):
                    match_maiden_bowlers.add(bowler_id)

        for bowler_id in match_maiden_bowlers:
            if bowler_id in match_per_player:
                match_per_player[bowler_id]["bowling_maidens"] = 1

        for player_id, record in match_per_player.items():
            record.setdefault("bowling_maidens", 0)
            match_rows.append(record)
            match_registry.setdefault(
                player_id,
                {
                    "player_id": player_id,
                    "player_name": registry_primary_names.get(player_id, ""),
                    "teams": set(),
                    "seasons": set(),
                },
            )
            if record["team"]:
                match_registry[player_id]["teams"].add(record["team"])
            match_registry[player_id]["seasons"].add(season)

        processed_match_ids.add(match_id)

    if not match_rows:
        empty_columns = {
            "player_registry.csv": [
                "player_id",
                "player_name",
                "name_variants",
                "seasons",
                "teams",
                "stable_source",
            ],
            "player_batting_stats.csv": [
                "player_id",
                "player_name",
                "matches",
                "innings",
                "runs",
                "balls_faced",
                "dismissals",
                "not_outs",
                "batting_average",
                "strike_rate",
                "highest_score",
                "fours",
                "sixes",
                "fifties",
                "hundreds",
            ],
            "player_bowling_stats.csv": [
                "player_id",
                "player_name",
                "matches",
                "innings",
                "overs",
                "overs_decimal",
                "legal_balls",
                "runs_conceded",
                "wickets",
                "bowling_average",
                "economy",
                "bowling_strike_rate",
                "best_wickets",
                "best_runs",
                "best_bowling",
                "maidens",
            ],
            "player_fielding_stats.csv": [
                "player_id",
                "player_name",
                "matches",
                "catches",
                "stumpings",
                "run_outs",
            ],
            "player_season_stats.csv": [
                "player_id",
                "player_name",
                "season",
                "team",
                "matches",
                "batting_innings",
                "batting_runs",
                "balls_faced",
                "batting_dismissals",
                "batting_average",
                "strike_rate",
                "highest_score",
                "fours",
                "sixes",
                "not_outs",
                "fifties",
                "hundreds",
                "bowling_innings",
                "bowling_legal_balls",
                "bowling_runs_conceded",
                "bowling_wickets",
                "bowling_maidens",
                "bowling_average",
                "bowling_economy",
                "bowling_strike_rate",
                "catches",
                "stumpings",
                "run_outs",
            ],
            "player_match_stats.csv": [
                "match_id",
                "match_date",
                "season",
                "team",
                "player_id",
                "player_name",
            ],
            "team_squads.csv": [
                "player_id",
                "player_name",
                "season",
                "team",
                "matches",
            ],
            "processing_report.csv": [
                "match_id",
                "filename",
                "reason",
            ],
        }
        empty_datasets = {
            filename: pd.DataFrame(columns=columns)
            for filename, columns in empty_columns.items()
        }
        empty_datasets["processing_report.csv"] = pd.DataFrame(
            malformed,
            columns=["match_id", "filename", "reason"],
        )
        return empty_datasets

    player_match_df = pd.DataFrame(match_rows)
    player_match_df = player_match_df.drop_duplicates(
        subset=["match_id", "player_id"],
        keep="last",
    )

    for row in player_match_df.to_dict("records"):
        player_id = row["player_id"]
        player_name = registry_primary_names.get(player_id, row["player_name"])

        batting = batting_innings.setdefault(
            player_id,
            {
                "player_id": player_id,
                "player_name": player_name,
                "matches": set(),
                "innings": 0,
                "runs": 0,
                "balls": 0,
                "dismissals": 0,
                "not_outs": 0,
                "fours": 0,
                "sixes": 0,
                "highest": 0,
                "fifties": 0,
                "hundreds": 0,
            },
        )
        if row["balls_faced"] > 0 or row["batting_runs"] > 0:
            batting["matches"].add(row["match_id"])
            batting["innings"] += 1
            batting["runs"] += int(row["batting_runs"])
            batting["balls"] += int(row["balls_faced"])
            batting["dismissals"] += int(row["batting_dismissed"])
            batting["not_outs"] += int(row["batting_not_out"])
            batting["fours"] += int(row["fours"])
            batting["sixes"] += int(row["sixes"])
            batting["highest"] = max(
                batting["highest"],
                int(row["highest_score"]),
            )
            score = int(row["highest_score"])
            if score >= 100:
                batting["hundreds"] += 1
            elif score >= 50:
                batting["fifties"] += 1

        bowling = bowling_innings.setdefault(
            player_id,
            {
                "player_id": player_id,
                "player_name": player_name,
                "matches": set(),
                "innings": 0,
                "balls": 0,
                "runs": 0,
                "wickets": 0,
                "maidens": 0,
                "best_wickets": 0,
                "best_runs": 0,
            },
        )
        if row["bowling_balls"] > 0:
            bowling["matches"].add(row["match_id"])
            bowling["innings"] += 1
            bowling["balls"] += int(row["bowling_balls"])
            bowling["runs"] += int(row["bowling_runs_conceded"])
            bowling["wickets"] += int(row["bowling_wickets"])
            bowling["maidens"] += int(row.get("bowling_maidens", 0) or 0)

            match_wickets = int(row["bowling_wickets"])
            match_runs = int(row["bowling_runs_conceded"])
            if (
                match_wickets > bowling["best_wickets"]
                or (
                    match_wickets == bowling["best_wickets"]
                    and match_runs < bowling["best_runs"]
                )
            ):
                bowling["best_wickets"] = match_wickets
                bowling["best_runs"] = match_runs

        fielding = fielding_totals.setdefault(
            player_id,
            {
                "player_id": player_id,
                "player_name": player_name,
                "matches": set(),
                "catches": 0,
                "stumpings": 0,
                "run_outs": 0,
            },
        )
        fielding["matches"].add(row["match_id"])
        fielding["catches"] += int(row["catches"])
        fielding["stumpings"] += int(row["stumpings"])
        fielding["run_outs"] += int(row["run_outs"])

    batting_records: list[dict[str, Any]] = []
    for record in batting_innings.values():
        if record["innings"] == 0:
            continue
        dismissals = record["dismissals"]
        balls = record["balls"]
        runs = record["runs"]
        batting_records.append(
            {
                "player_id": record["player_id"],
                "player_name": record["player_name"],
                "matches": len(record["matches"]),
                "innings": record["innings"],
                "runs": runs,
                "balls_faced": balls,
                "dismissals": dismissals,
                "not_outs": record["not_outs"],
                "batting_average": (
                    runs / dismissals if dismissals > 0 else None
                ),
                "strike_rate": (
                    runs / balls * 100 if balls > 0 else None
                ),
                "highest_score": record["highest"],
                "fours": record["fours"],
                "sixes": record["sixes"],
                "fifties": record["fifties"],
                "hundreds": record["hundreds"],
            }
        )

    bowling_records: list[dict[str, Any]] = []
    for record in bowling_innings.values():
        if record["innings"] == 0:
            continue
        balls = record["balls"]
        overs_decimal = balls / 6
        overs_complete = balls // 6
        overs_remainder = balls % 6
        wickets = record["wickets"]
        bowling_records.append(
            {
                "player_id": record["player_id"],
                "player_name": record["player_name"],
                "matches": len(record["matches"]),
                "innings": record["innings"],
                "overs": f"{overs_complete}.{overs_remainder}",
                "overs_decimal": overs_decimal,
                "legal_balls": balls,
                "runs_conceded": record["runs"],
                "wickets": wickets,
                "bowling_average": (
                    record["runs"] / wickets if wickets > 0 else None
                ),
                "economy": (
                    record["runs"] / overs_decimal if overs_decimal > 0 else None
                ),
                "bowling_strike_rate": (
                    balls / wickets if wickets > 0 else None
                ),
                "best_wickets": record["best_wickets"],
                "best_runs": record["best_runs"],
                "best_bowling": (
                    f"{record['best_wickets']}/{record['best_runs']}"
                    if record["best_wickets"] or record["best_runs"]
                    else ""
                ),
                "maidens": record["maidens"],
            }
        )

    fielding_records: list[dict[str, Any]] = []
    for record in fielding_totals.values():
        fielding_records.append(
            {
                "player_id": record["player_id"],
                "player_name": record["player_name"],
                "matches": len(record["matches"]),
                "catches": record["catches"],
                "stumpings": record["stumpings"],
                "run_outs": record["run_outs"],
            }
        )

    registry_records: list[dict[str, Any]] = []
    for player_id, names in registry_names.items():
        match_registry_record = match_registry.get(
            player_id,
            {"teams": set(), "seasons": set()},
        )
        registry_records.append(
            {
                "player_id": player_id,
                "player_name": registry_primary_names.get(player_id, sorted(names)[0]),
                "name_variants": "; ".join(sorted(names)),
                "seasons": "; ".join(sorted(match_registry_record["seasons"])),
                "teams": "; ".join(sorted(match_registry_record["teams"])),
                "stable_source": (
                    "cricsheet_registry"
                    if not player_id.startswith("name-")
                    else "normalized_name_fallback"
                ),
            }
        )

    squad_records = []
    for (player_id, season, team), record in sorted(team_seasons.items()):
        squad_records.append(record)

    player_season_rows: list[dict[str, Any]] = []
    season_team_groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in player_match_df.to_dict("records"):
        team = row["team"]
        if team:
            season_team_groups[
                (row["player_id"], row["season"], team)
            ].append(row)

    for (player_id, season, team), rows in sorted(season_team_groups.items()):
        player_name = registry_primary_names.get(
            player_id,
            rows[0]["player_name"],
        )
        match_ids = {row["match_id"] for row in rows}
        batting_rows_for_player = [
            row for row in rows
            if row["balls_faced"] > 0 or row["batting_runs"] > 0
        ]
        runs = sum(int(row["batting_runs"]) for row in batting_rows_for_player)
        balls = sum(int(row["balls_faced"]) for row in batting_rows_for_player)
        dismissals = sum(int(row["batting_dismissed"]) for row in batting_rows_for_player)
        not_outs = sum(int(row["batting_not_out"]) for row in batting_rows_for_player)
        highest = max(
            (int(row["highest_score"]) for row in batting_rows_for_player),
            default=0,
        )
        fours = sum(int(row["fours"]) for row in batting_rows_for_player)
        sixes = sum(int(row["sixes"]) for row in batting_rows_for_player)
        bowling_rows_for_player = [
            row for row in rows
            if row["bowling_balls"] > 0
        ]
        bowling_balls = sum(int(row["bowling_balls"]) for row in bowling_rows_for_player)
        bowling_runs = sum(
            int(row["bowling_runs_conceded"])
            for row in bowling_rows_for_player
        )
        bowling_wickets = sum(
            int(row["bowling_wickets"])
            for row in bowling_rows_for_player
        )
        bowling_maidens = sum(
            int(row.get("bowling_maidens", 0) or 0)
            for row in bowling_rows_for_player
        )
        overs_decimal = bowling_balls / 6

        catches = sum(int(row["catches"]) for row in rows)
        stumpings = sum(int(row["stumpings"]) for row in rows)
        run_outs = sum(int(row["run_outs"]) for row in rows)

        player_season_rows.append(
            {
                "player_id": player_id,
                "player_name": player_name,
                "season": season,
                "team": team,
                "matches": len(match_ids),
                "batting_innings": len(batting_rows_for_player),
                "batting_runs": runs,
                "balls_faced": balls,
                "batting_dismissals": dismissals,
                "batting_average": (
                    runs / dismissals if dismissals > 0 else None
                ),
                "strike_rate": (
                    runs / balls * 100 if balls > 0 else None
                ),
                "highest_score": highest,
                "fours": fours,
                "sixes": sixes,
                "not_outs": not_outs,
                "fifties": sum(
                    1
                    for row in batting_rows_for_player
                    if 50 <= int(row["highest_score"]) < 100
                ),
                "hundreds": sum(
                    1
                    for row in batting_rows_for_player
                    if int(row["highest_score"]) >= 100
                ),
                "bowling_innings": len(bowling_rows_for_player),
                "bowling_legal_balls": bowling_balls,
                "bowling_runs_conceded": bowling_runs,
                "bowling_wickets": bowling_wickets,
                "bowling_maidens": bowling_maidens,
                "bowling_average": (
                    bowling_runs / bowling_wickets
                    if bowling_wickets > 0
                    else None
                ),
                "bowling_economy": (
                    bowling_runs / overs_decimal
                    if overs_decimal > 0
                    else None
                ),
                "bowling_strike_rate": (
                    bowling_balls / bowling_wickets
                    if bowling_wickets > 0
                    else None
                ),
                "catches": catches,
                "stumpings": stumpings,
                "run_outs": run_outs,
            }
        )

    return {
        "player_registry.csv": pd.DataFrame(registry_records).sort_values(
            ["player_name", "player_id"]
        ),
        "player_batting_stats.csv": pd.DataFrame(batting_records).sort_values(
            ["runs", "player_name"],
            ascending=[False, True],
        ),
        "player_bowling_stats.csv": pd.DataFrame(bowling_records).sort_values(
            ["wickets", "player_name"],
            ascending=[False, True],
        ),
        "player_fielding_stats.csv": pd.DataFrame(fielding_records).sort_values(
            ["catches", "player_name"],
            ascending=[False, True],
        ),
        "player_season_stats.csv": pd.DataFrame(player_season_rows).sort_values(
            ["season", "player_name", "team"]
        ),
        "player_match_stats.csv": player_match_df.sort_values(
            ["match_date", "match_id", "player_name"]
        ),
        "team_squads.csv": pd.DataFrame(squad_records).sort_values(
            ["season", "team", "player_name"],
            ascending=[False, True, True],
        ),
        "processing_report.csv": pd.DataFrame(
            [
                {
                    "match_id": record["match_id"],
                    "filename": record["filename"],
                    "reason": record["reason"],
                }
                for record in malformed
            ]
        ),
    }
