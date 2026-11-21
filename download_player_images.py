from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

from player_images import TARGET_PLAYER_NAMES, player_image_filename


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "assets" / "players"
DATA_CSV = BASE_DIR / "data" / "player_images.csv"

ALLOWED_LICENSE_MARKERS = (
    "cc by",
    "cc0",
    "public domain",
    "godl-india",
)


def commons_api_search(name: str) -> list[dict]:
    query = urllib.parse.quote(f'"{name}" cricket')
    url = (
        "https://commons.wikimedia.org/w/api.php"
        "?action=query"
        "&generator=search"
        f"&gsrsearch={query}"
        "&gsrnamespace=6"
        "&gsrlimit=12"
        "&prop=imageinfo"
        "&iiprop=url|extmetadata"
        "&iiurlwidth=600"
        "&format=json"
    )
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "IPL-Intelligence-Educational-Project/1.0"},
    )

    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=25) as response:
                payload = json.load(response)
            pages = payload.get("query", {}).get("pages", {})
            return list(pages.values())
        except Exception:
            if attempt == 2:
                return []
            time.sleep(2 * (attempt + 1))
    return []


def license_is_reusable(license_name: str) -> bool:
    normalized = license_name.casefold()
    if "fair use" in normalized or "rights reserved" in normalized:
        return False
    return any(marker in normalized for marker in ALLOWED_LICENSE_MARKERS)


def choose_candidate(
    name: str,
    candidates: list[dict],
) -> dict | None:
    first, *remaining = name.casefold().split()
    for page in candidates:
        title = page.get("title", "")
        image_info = page.get("imageinfo", [{}])[0]
        metadata = image_info.get("extmetadata", {})
        license_name = metadata.get("LicenseShortName", {}).get("value", "")
        description_url = image_info.get("descriptionurl", "")
        thumb_url = image_info.get("thumburl", "")
        original_url = image_info.get("url", "")

        if not license_is_reusable(license_name):
            continue
        title_lower = title.casefold()
        if first not in title_lower:
            continue
        if remaining and remaining[0] not in title_lower:
            continue

        return {
            "title": title,
            "license": license_name,
            "source_url": description_url or original_url,
            "download_url": thumb_url or original_url,
        }
    return None


def download_image(url: str, destination: Path) -> bool:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "IPL-Intelligence-Educational-Project/1.0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            content_type = response.headers.get("Content-Type", "")
            if not content_type.startswith("image/"):
                return False
            data = response.read()
        if len(data) < 5_000:
            return False
        destination.write_bytes(data)
        return True
    except Exception:
        return False


def build_initial_csv() -> None:
    DATA_CSV.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "player_name": name,
            "image_filename": "",
            "source_url": "",
            "verification_status": "fallback_initials",
            "image_license_or_usage_notes": "No verified reusable portrait; use initials-based avatar.",
        }
        for name in TARGET_PLAYER_NAMES
    ]
    pd.DataFrame(rows).to_csv(DATA_CSV, index=False)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    build_initial_csv()
    frame = pd.read_csv(DATA_CSV, dtype=str).fillna("")

    for index, row in frame.iterrows():
        name = str(row["player_name"])
        candidates = commons_api_search(name)
        candidate = choose_candidate(name, candidates)

        if candidate is None:
            time.sleep(0.25)
            continue

        filename = player_image_filename(name)
        destination = OUTPUT_DIR / filename
        if destination.exists() or download_image(
            candidate["download_url"],
            destination,
        ):
            frame.at[index, "image_filename"] = filename
            frame.at[index, "source_url"] = candidate["source_url"]
            frame.at[index, "verification_status"] = "verified_download"
            frame.at[index, "image_license_or_usage_notes"] = (
                f"{candidate['license']}; source: Commons; "
                "verify attribution requirements before public use."
            )

        time.sleep(0.25)

    frame.to_csv(DATA_CSV, index=False)
    print(
        f"Wrote {len(frame)} player image mappings; "
        f"{int((frame['verification_status'] == 'verified_download').sum())} local portraits."
    )


if __name__ == "__main__":
    main()
