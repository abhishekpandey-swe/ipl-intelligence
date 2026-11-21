# Data Sources and Limitations

## Existing match dataset

`data/matches.csv` is the only verified match dataset currently used by the application.

- Coverage: 2008 through 3 June 2025
- Grain: one row per IPL match
- Fields used: match date, teams, venue, toss, winner and season
- Limitations: no ball-by-ball records, playing XI, player batting statistics, player bowling statistics or player identity registry

The player-of-match field in the current CSV contains numeric source references in the inspected data. It does not provide a player name or verified player profile.

## Optional data ingestion

The following files are not currently present:

- `data/deliveries.csv`
- `data/player_statistics.csv`
- `data/player_registry.csv`
- `data/team_squads.csv`

`data_ingestion.py` validates optional file schemas but does not invent or merge records.

## Cricsheet IPL JSON archive

`data/raw/cricsheet_ipl/` contains 1,243 JSON match files with a data range of 18 April 2008 through 31 May 2026.

The archive has a registry of stable people IDs and 295,732 deliveries. The processing pipeline produces real career and season-level batting, bowling, fielding and team-participation datasets.

The pipeline:

- Excludes super-over innings from regular player statistics.
- Excludes no-result matches.
- Counts wides and no-balls separately from legal bowling balls.
- Excludes byes and leg-byes from bowler-conceded runs.
- Credits bowler wickets only for bowled, caught, caught-and-bowled, LBW, stumped and hit-wicket dismissals.
- Uses Cricsheet registry IDs where available and a deterministic name fallback otherwise.

## Player images

Player portraits are sourced only from reusable-license candidates found in Wikimedia Commons. The source page URL and license/usage note are retained in `data/player_images.csv`.

GODL-India and Creative Commons candidates remain subject to their original attribution and share-alike conditions. Public-use attribution should be verified against the source metadata.

The following are excluded from local downloads:

- Group photographs
- Jerseys or museum objects
- Event/collage images
- Candidates with unknown or unverified usage terms

Excluded players display a local initials-based avatar.

## Approved external-source policy

Suitable sources include Cricsheet and established public IPL ball-by-ball repositories. Any imported dataset must include:

- Source URL or repository
- Download date
- Coverage period
- Field definitions
- License or redistribution terms
- Name and identity normalization method
- Verification that every player was not merged by approximate name matching

No external record should be included until source and schema verification is complete.
