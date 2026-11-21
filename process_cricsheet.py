from __future__ import annotations

import argparse
from pathlib import Path

from cricsheet_pipeline import process_archive


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_RAW_DIR = BASE_DIR / "data" / "raw" / "cricsheet_ipl"
DEFAULT_OUTPUT_DIR = BASE_DIR / "data" / "processed"


def write_processed_datasets(
    raw_dir: Path,
    output_dir: Path,
) -> dict[str, int]:
    datasets = process_archive(raw_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    counts: dict[str, int] = {}
    for filename, dataframe in datasets.items():
        if dataframe is not None:
            dataframe.to_csv(output_dir / filename, index=False)
            counts[filename] = len(dataframe)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Process the Cricsheet IPL JSON archive into verified player CSV datasets."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    args = parser.parse_args()

    if not args.raw_dir.exists():
        raise FileNotFoundError(
            f"Cricsheet raw archive not found: {args.raw_dir}"
        )

    counts = write_processed_datasets(args.raw_dir, args.output_dir)
    print("Cricsheet IPL processing complete")
    for filename, count in sorted(counts.items()):
        print(f"{filename}: {count:,} rows")


if __name__ == "__main__":
    main()
