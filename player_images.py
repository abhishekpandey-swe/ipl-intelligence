from __future__ import annotations

from pathlib import Path

import pandas as pd


TARGET_PLAYER_NAMES = [
    "Virat Kohli",
    "Rohit Sharma",
    "MS Dhoni",
    "Suryakumar Yadav",
    "Shubman Gill",
    "KL Rahul",
    "Rishabh Pant",
    "Hardik Pandya",
    "Sanju Samson",
    "Shreyas Iyer",
    "Yashasvi Jaiswal",
    "Ruturaj Gaikwad",
    "Tilak Varma",
    "Rinku Singh",
    "Robin Uthappa",
    "Jasprit Bumrah",
    "Mohammed Shami",
    "Mohammed Siraj",
    "Yuzvendra Chahal",
    "Kuldeep Yadav",
    "Ravindra Jadeja",
    "Axar Patel",
    "Bhuvneshwar Kumar",
    "Varun Chakravarthy",
    "Arshdeep Singh",
    "AB de Villiers",
    "Chris Gayle",
    "David Warner",
    "Jos Buttler",
    "David Miller",
    "Kane Williamson",
    "Faf du Plessis",
    "Glenn Maxwell",
    "Andre Russell",
    "Kieron Pollard",
    "Rashid Khan",
    "Sunil Narine",
    "Pat Cummins",
    "Mitchell Starc",
    "Trent Boult",
    "Kagiso Rabada",
    "Nicholas Pooran",
    "Quinton de Kock",
    "Shane Watson",
    "Lasith Malinga",
    "Dwayne Bravo",
    "Shane Warne",
    "Dale Steyn",
    "Brett Lee",
    "Suresh Raina",
]

PLAYER_IMAGE_ALIASES = {
    "Virat Kohli": ["V Kohli", "Virat Kohli"],
    "Rohit Sharma": ["RG Sharma", "Rohit Sharma"],
    "MS Dhoni": ["MS Dhoni"],
    "Suryakumar Yadav": ["SA Yadav", "Suryakumar Yadav"],
    "Shubman Gill": ["Shubman Gill"],
    "KL Rahul": ["KL Rahul"],
    "Rishabh Pant": ["RR Pant", "Rishabh Pant"],
    "Hardik Pandya": ["HH Pandya", "Hardik Pandya"],
    "Sanju Samson": ["SV Samson", "Sanju Samson"],
    "Shreyas Iyer": ["SS Iyer", "Shreyas Iyer"],
    "Yashasvi Jaiswal": ["YBK Jaiswal", "Yashasvi Jaiswal"],
    "Ruturaj Gaikwad": ["RD Gaikwad", "Ruturaj Gaikwad"],
    "Tilak Varma": ["Tilak Varma"],
    "Rinku Singh": ["RK Singh", "Rinku Singh"],
    "Robin Uthappa": ["RV Uthappa", "Robin Uthappa"],
    "Jasprit Bumrah": ["JJ Bumrah", "Jasprit Bumrah"],
    "Mohammed Shami": ["Mohammed Shami"],
    "Mohammed Siraj": ["Mohammed Siraj"],
    "Yuzvendra Chahal": ["YS Chahal", "Yuzvendra Chahal"],
    "Kuldeep Yadav": ["Kuldeep Yadav"],
    "Ravindra Jadeja": ["RA Jadeja", "Ravindra Jadeja"],
    "Axar Patel": ["AR Patel", "Axar Patel"],
    "Bhuvneshwar Kumar": ["B Kumar", "Bhuvneshwar Kumar"],
    "Varun Chakravarthy": ["CV Varun", "Varun Chakravarthy"],
    "Arshdeep Singh": ["Arshdeep Singh"],
    "AB de Villiers": ["AB de Villiers"],
    "Chris Gayle": ["CH Gayle", "Chris Gayle"],
    "David Warner": ["DA Warner", "David Warner"],
    "Jos Buttler": ["JC Buttler", "Jos Buttler"],
    "David Miller": ["DA Miller", "David Miller"],
    "Kane Williamson": ["KS Williamson", "Kane Williamson"],
    "Faf du Plessis": ["F du Plessis", "Faf du Plessis"],
    "Glenn Maxwell": ["GJ Maxwell", "Glenn Maxwell"],
    "Andre Russell": ["AD Russell", "Andre Russell"],
    "Kieron Pollard": ["KA Pollard", "Kieron Pollard"],
    "Rashid Khan": ["Rashid Khan"],
    "Sunil Narine": ["SP Narine", "Sunil Narine"],
    "Pat Cummins": ["PJ Cummins", "Pat Cummins"],
    "Mitchell Starc": ["MA Starc", "Mitchell Starc"],
    "Trent Boult": ["TA Boult", "Trent Boult"],
    "Kagiso Rabada": ["K Rabada", "Kagiso Rabada"],
    "Nicholas Pooran": ["N Pooran", "Nicholas Pooran"],
    "Quinton de Kock": ["Q de Kock", "Quinton de Kock"],
    "Shane Watson": ["SR Watson", "Shane Watson"],
    "Lasith Malinga": ["SL Malinga", "Lasith Malinga"],
    "Dwayne Bravo": ["DJ Bravo", "Dwayne Bravo"],
    "Shane Warne": ["SK Warne", "Shane Warne"],
    "Dale Steyn": ["DW Steyn", "Dale Steyn"],
    "Brett Lee": ["B Lee", "Brett Lee"],
    "Suresh Raina": ["SK Raina", "Suresh Raina"],
}


def player_image_filename(player_name: str) -> str:
    return (
        "".join(
            character.lower()
            if character.isalnum()
            else "_"
            for character in player_name
        )
        .strip("_")
        .replace("__", "_")
        + ".jpg"
    )


def load_player_images(data_dir: Path) -> dict[str, dict[str, str]]:
    csv_path = data_dir / "player_images.csv"
    if not csv_path.exists():
        return {}

    frame = pd.read_csv(csv_path).fillna("")
    return {
        str(row["player_name"]): {
            "player_name": str(row["player_name"]),
            "image_filename": str(row["image_filename"]),
            "source_url": str(row["source_url"]),
            "verification_status": str(row["verification_status"]),
            "image_license_or_usage_notes": str(
                row["image_license_or_usage_notes"]
            ),
        }
        for row in frame.to_dict("records")
    }


def resolve_player_image(
    data_dir: Path,
    player_name: str,
) -> tuple[Path | None, dict[str, str] | None]:
    mapping = load_player_images(data_dir)
    record = mapping.get(player_name)
    if record is None:
        return None, None

    filename = record["image_filename"].strip()
    if not filename:
        return None, record

    image_path = data_dir.parent / "assets" / "players" / filename
    if not image_path.exists():
        return None, record
    return image_path, record


def player_initials(player_name: str) -> str:
    parts = [part for part in player_name.split() if part]
    if not parts:
        return "IPL"
    if len(parts) == 1:
        return parts[0][:2].upper()
    return (parts[0][0] + parts[-1][0]).upper()


def resolve_registry_player_image(
    data_dir: Path,
    registry_name: str,
) -> tuple[Path | None, dict[str, str] | None]:
    if registry_name in TARGET_PLAYER_NAMES:
        return resolve_player_image(data_dir, registry_name)

    for target_name, aliases in PLAYER_IMAGE_ALIASES.items():
        if registry_name in aliases:
            return resolve_player_image(data_dir, target_name)
    return None, None
