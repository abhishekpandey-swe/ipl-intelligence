from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


OPTIONAL_SCHEMAS: dict[str, set[str]] = {
    "deliveries.csv": {
        "match_id",
        "inning",
        "over",
        "ball",
        "batting_team",
        "bowling_team",
        "batter",
        "bowler",
        "batsman_runs",
        "total_runs",
    },
    "player_statistics.csv": {
        "player_name",
        "season",
        "team",
        "role",
        "matches",
    },
    "player_registry.csv": {
        "player_id",
        "player_name",
    },
    "team_squads.csv": {
        "season",
        "team",
        "player_name",
        "role",
    },
}


def inspect_optional_datasets(data_dir: Path) -> list[dict[str, object]]:
    report: list[dict[str, object]] = []
    for filename, required_columns in OPTIONAL_SCHEMAS.items():
        path = data_dir / filename
        if not path.exists():
            report.append(
                {
                    "file": filename,
                    "status": "not_available",
                    "coverage": "unknown",
                    "source": "not supplied",
                    "missing_columns": sorted(required_columns),
                }
            )
            continue

        df = pd.read_csv(path)
        available = set(df.columns.str.lower().str.strip())
        missing = required_columns - available
        report.append(
            {
                "file": filename,
                "status": "available" if not missing else "invalid_schema",
                "coverage": f"{len(df):,} rows",
                "source": "verified external source metadata required",
                "missing_columns": sorted(missing),
            }
        )
    return report


def print_report(data_dir: Path) -> None:
    for item in inspect_optional_datasets(data_dir):
        print(
            f"{item['file']} | {item['status']} | "
            f"coverage={item['coverage']} | "
            f"source={item['source']} | "
            f"missing_columns={item['missing_columns']}"
        )
