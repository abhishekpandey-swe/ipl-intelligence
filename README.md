# IPL / INTELLIGENCE

A premium, responsive IPL analytics platform built with Python and Streamlit. It includes match prediction, historical team and venue analytics, model comparison, and transparent data-limited player intelligence.

## Project Structure

```text
ipl-prediction/
├── app.py
├── train_model.py
├── train_experimental.py
├── evaluate_models.py
├── feature_engineering.py
├── player_intelligence.py
├── data_ingestion.py
├── DATA_SOURCES.md
├── predict.py
├── test_predict.py
├── test_feature_engineering.py
├── requirements.txt
├── README.md
├── data/
│   └── matches.csv
├── models/
│   ├── ipl_model.joblib
│   └── ipl_model_experimental.joblib
└── assets/
    └── style.css
```

## Production and Experimental Models

- `models/ipl_model.joblib` is the existing production Logistic Regression model and remains unchanged.
- `models/ipl_model_experimental.joblib` is created only by `train_experimental.py`.
- The Streamlit interface lets the user explicitly choose between the production and experimental models.
- The experimental model uses leakage-free historical team, head-to-head and venue features.

## Application Navigation

- Overview
- Match Prediction
- IPL Teams
- Players
- Player Performance Predictor
- Analytics
- Model Intelligence

The player module is intentionally data-limited. `matches.csv` does not contain verified player names, career statistics, season batting and bowling records, playing XI or squad membership. The application does not fabricate these values.

No player prediction is generated until a verified player-level dataset is added and validated.

## Player portraits

The 50-player image registry is stored in `data/player_images.csv`. Reusable local portraits are under `assets/players/`.

```powershell
python download_player_images.py
```

This searches Wikimedia Commons for reusable-license candidates. A missing or excluded portrait is shown as a transparent initials-based fallback. The registry always records 50 unique player names, source URLs and usage notes.

## Cricsheet player processing

The raw IPL JSON archive is read once with:

```powershell
python process_cricsheet.py
```

Generated files are written to `data/processed/`:

- `player_registry.csv`
- `player_batting_stats.csv`
- `player_bowling_stats.csv`
- `player_fielding_stats.csv`
- `player_season_stats.csv`
- `player_match_stats.csv`
- `team_squads.csv`
- `processing_report.csv`

No-result matches are excluded from generated match/player aggregates. Super Overs are excluded from regular career and season batting/bowling statistics.

## Run

```powershell
python test_predict.py
python test_feature_engineering.py
streamlit run app.py
```

Optional experimental evaluation:

```powershell
python train_experimental.py
```

## Setup & Run (Windows)

1. Place your historical match CSV at `data/matches.csv` (must include valid match dates `date` or `match_date`, plus `team1`, `team2`, `venue`, `toss_winner`, `toss_decision`, and `winner` or `match_winner`).
2. Run from the `ipl-prediction` directory:

```powershell
pip install -r requirements.txt
python train_model.py
python test_predict.py
streamlit run app.py
```

## Streamlit Community Cloud Deployment

1. Push this repository to GitHub. Do not commit `.venv/`, cache folders, local environment files or `.streamlit/secrets.toml`.
2. In Streamlit Community Cloud, select the repository and set the main file to `app.py`.
3. Use `requirements.txt` as the dependency file.
4. The runtime uses `runtime.txt` to request Python 3.11.
5. Keep these deployment-required files in the repository:
   - `data/matches.csv`
   - `data/player_images.csv`
   - `data/processed/*.csv`
   - `models/ipl_model.joblib`
   - `models/ipl_model_experimental.joblib`
   - `assets/style.css`
   - `assets/players/*.jpg`
6. The raw `data/raw/cricsheet_ipl/` archive is excluded because the running app uses processed CSV files. It is not required for prediction or player UI loading.

No API keys or secrets are required by the current application.

## Notes

- Predictions use **pre-match features only** (`batting_team`, `bowling_team`, `venue`, `toss_winner`, `toss_decision`) derived consistently from Team 1, Team 2, Toss Winner, and Toss Decision.
- Probabilities are statistical estimates and do not guarantee match outcomes.
