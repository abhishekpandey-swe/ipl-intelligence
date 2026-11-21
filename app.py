from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any
import base64
import html

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from feature_engineering import (
    generate_live_features,
    load_clean_match_history,
    normalize_venue,
)
from player_intelligence import (
    CURRENT_IPL_TEAMS,
    get_player_profile,
    get_team_squad,
    load_processed_player_data,
    load_match_history,
    team_profile,
    calculate_team_comparison,
    prepare_match_analytics_history,
    get_predictor_data_summary,
    get_player_predictor_history,
    filter_match_results,
    paginate_match_results,
    summarize_match_results,
)
from player_images import (
    player_initials,
    resolve_registry_player_image,
)
from predict import (
    DEFAULT_MODEL_PATH,
    InvalidMatchInputError,
    ModelNotFoundError,
    load_model_artifact,
    predict_match_winner,
)
from train_model import derive_innings_teams, get_selected_model_metrics


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_PATH = DATA_DIR / "matches.csv"
CSS_PATH = BASE_DIR / "assets" / "style.css"
EXPERIMENTAL_MODEL_PATH = BASE_DIR / "models" / "ipl_model_experimental.joblib"
DATASET_LAST_UPDATED = pd.Timestamp("2025-06-03")

st.set_page_config(
    page_title="IPL / Intelligence",
    page_icon="I",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(show_spinner=False)
def load_history() -> pd.DataFrame:
    return load_clean_match_history(DATA_PATH)


@st.cache_data(show_spinner=False)
def load_player_history() -> pd.DataFrame:
    return load_match_history(DATA_PATH)


@st.cache_data(show_spinner=False)
def load_processed_players() -> dict[str, pd.DataFrame]:
    return load_processed_player_data(DATA_DIR / "processed")


@st.cache_data(show_spinner=False)
def player_image_as_data_uri(player_name: str) -> str | None:
    image_path, _ = resolve_registry_player_image(DATA_DIR, player_name)
    if image_path is None:
        return None
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


@st.cache_resource(show_spinner=False)
def load_experimental_artifact() -> dict[str, Any] | None:
    if not EXPERIMENTAL_MODEL_PATH.exists():
        return None
    artifact = joblib.load(EXPERIMENTAL_MODEL_PATH)
    return artifact if isinstance(artifact, dict) else None


def inject_css() -> None:
    if CSS_PATH.exists():
        st.markdown(
            f"<style>{CSS_PATH.read_text(encoding='utf-8')}</style>",
            unsafe_allow_html=True,
        )


def render_header(title: str, eyebrow: str, status: str = "SYSTEM READY") -> None:
    st.markdown(
        f"""
        <header class="ipl-header">
            <div>
                <div class="ipl-eyebrow">{eyebrow}</div>
                <div class="ipl-title">{title}</div>
            </div>
            <div class="ipl-status"><span></span>{status}</div>
        </header>
        """,
        unsafe_allow_html=True,
    )


def metric(label: str, value: str, accent: str = "") -> None:
    st.markdown(
        f"""
        <div class="ipl-metric {accent}">
            <div class="ipl-metric-label">{label}</div>
            <div class="ipl-metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def empty_state(title: str, message: str) -> None:
    st.markdown(
        f"""
        <div class="ipl-empty">
            <div class="ipl-empty-glyph">.</div>
            <div>
                <strong>{title}</strong>
                <p>{message}</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def plot_theme(fig: go.Figure, height: int = 360) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=20, r=20, t=55, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0b0b0b",
        font=dict(
            family="Inter, Arial, sans-serif",
            color="#e7e7e7",
        ),
        title_font=dict(
            family="Inter, Arial, sans-serif",
            color="#ffffff",
            size=16,
        ),
    )
    fig.update_xaxes(gridcolor="#232323", zeroline=False)
    fig.update_yaxes(gridcolor="#232323", zeroline=False)
    return fig


def render_overview(history: pd.DataFrame, production: dict[str, Any]) -> None:
    render_header("OVERVIEW", "IPL / MATCH INTELLIGENCE")

    teams = sorted(set(history["normalized_team1"]).union(set(history["normalized_team2"])))
    venues = sorted(history["venue"].dropna().astype(str).unique())

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric("COMPLETED MATCHES", f"{len(history):,}", "red")
    with c2:
        metric("FRANCHISES", str(len(teams)))
    with c3:
        metric("VENUES", str(len(venues)))
    with c4:
        metric("DATA THROUGH", "03 JUN 2025")

    wins: dict[str, int] = {}
    for team in teams:
        wins[team] = int(history["normalized_winner"].eq(team).sum())
    win_df = pd.DataFrame(
        sorted(wins.items(), key=lambda item: item[1], reverse=True),
        columns=["Team", "Wins"],
    )
    team_height = max(420, len(win_df) * 38)
    fig = px.bar(
        win_df,
        x="Wins",
        y="Team",
        orientation="h",
        text="Wins",
        hover_data={"Team": True, "Wins": True},
    )
    fig.update_traces(
        marker_color="#ff3b30",
        texttemplate="%{text}",
        textposition="outside",
        cliponaxis=False,
        hovertemplate="<b>%{y}</b><br>Wins: %{x}<extra></extra>",
    )
    fig.update_layout(
        title={
            "text": "IPL Teams — Total Wins<br><sup>Historical wins across IPL franchises</sup>",
            "x": 0,
            "xanchor": "left",
        },
        showlegend=False,
        height=team_height,
        margin=dict(l=170, r=55, t=95, b=45),
        xaxis_title="Total wins",
        yaxis_title=None,
        yaxis=dict(
            categoryorder="array",
            categoryarray=win_df["Team"].tolist(),
            autorange="reversed",
            automargin=True,
        ),
        xaxis=dict(
            rangemode="tozero",
            gridcolor="#232323",
            zeroline=False,
        ),
    )
    st.plotly_chart(
        plot_theme(fig, team_height),
        use_container_width=True,
    )

    venue_counts = (
        history["venue"]
        .fillna("Unknown")
        .value_counts()
        .head(10)
        .rename_axis("Venue")
        .reset_index(name="Matches")
    )
    venue_counts = venue_counts.sort_values(
        "Matches",
        ascending=False,
        kind="stable",
    )
    venue_height = max(420, len(venue_counts) * 48)
    fig = px.bar(
        venue_counts,
        x="Matches",
        y="Venue",
        orientation="h",
        text="Matches",
        hover_data={"Venue": True, "Matches": True},
    )
    fig.update_traces(
        marker_color="#ff3b30",
        texttemplate="%{text}",
        textposition="outside",
        cliponaxis=False,
        hovertemplate="<b>%{y}</b><br>Matches hosted: %{x}<extra></extra>",
    )
    fig.update_layout(
        title={
            "text": "IPL Venues — Matches Hosted<br><sup>Distribution of historical matches across the most frequently used venues</sup>",
            "x": 0,
            "xanchor": "left",
        },
        showlegend=False,
        height=venue_height,
        margin=dict(l=210, r=55, t=95, b=45),
        xaxis_title="Matches hosted",
        yaxis_title=None,
        yaxis=dict(
            categoryorder="array",
            categoryarray=venue_counts["Venue"].tolist(),
            autorange="reversed",
            automargin=True,
        ),
        xaxis=dict(
            rangemode="tozero",
            gridcolor="#232323",
            zeroline=False,
        ),
    )
    st.plotly_chart(
        plot_theme(fig, venue_height),
        use_container_width=True,
    )

    st.markdown("### IPL MATCH RESULTS CENTER")

    team_options = ["All teams"] + sorted(
        set(history["normalized_team1"].dropna().astype(str))
        .union(set(history["normalized_team2"].dropna().astype(str)))
    )
    winner_options = ["All winners"] + sorted(
        history["normalized_winner"].dropna().astype(str).unique().tolist()
    )
    venue_options = ["All venues"] + sorted(
        history["venue"].dropna().astype(str).str.strip().unique().tolist()
    )

    search = st.text_input(
        "SEARCH MATCHES",
        placeholder="Team, winner or venue",
        key="match_results_search",
    )
    filter_col1, filter_col2, filter_col3, filter_col4 = st.columns([1, 1, 1.35, 0.8])
    with filter_col1:
        team_filter = st.selectbox(
            "TEAM",
            team_options,
            key="match_results_team",
        )
    with filter_col2:
        winner_filter = st.selectbox(
            "WINNING TEAM",
            winner_options,
            key="match_results_winner",
        )
    with filter_col3:
        venue_filter = st.selectbox(
            "VENUE",
            venue_options,
            key="match_results_venue",
        )
    with filter_col4:
        sort_order = st.selectbox(
            "SORT",
            ["Most recent first", "Oldest first"],
            key="match_results_sort",
        )

    signature = (
        search,
        team_filter,
        winner_filter,
        venue_filter,
        sort_order,
    )
    if st.session_state.get("_match_results_filter_signature") != signature:
        st.session_state["_match_results_filter_signature"] = signature
        st.session_state["match_results_page"] = 1
    if "match_results_page" not in st.session_state:
        st.session_state["match_results_page"] = 1

    if st.button("RESET FILTERS", key="match_results_reset"):
        st.session_state["match_results_search"] = ""
        st.session_state["match_results_team"] = "All teams"
        st.session_state["match_results_winner"] = "All winners"
        st.session_state["match_results_venue"] = "All venues"
        st.session_state["match_results_sort"] = "Most recent first"
        st.session_state["match_results_page"] = 1
        st.session_state["_match_results_filter_signature"] = (
            "",
            "All teams",
            "All winners",
            "All venues",
            "Most recent first",
        )
        st.rerun()

    filtered = filter_match_results(
        history,
        search=search,
        team=team_filter,
        winner=winner_filter,
        venue=venue_filter,
        sort_order=sort_order,
    )
    summary = summarize_match_results(filtered)

    s1, s2, s3, s4 = st.columns(4)
    with s1:
        metric("MATCHES IN FILTER", f"{summary['matches']:,}", "red")
    with s2:
        metric("UNIQUE TEAMS", str(summary["teams"]))
    with s3:
        metric("UNIQUE VENUES", str(summary["venues"]))
    with s4:
        metric("MOST FREQUENT WINNER", str(summary["most_frequent_winner"]))

    if filtered.empty:
        empty_state(
            "NO MATCHES FOUND",
            "No historical matches match the current search and filter selection.",
        )
    else:
        page_size = 10
        page_data, safe_page, total_pages = paginate_match_results(
            filtered,
            st.session_state["match_results_page"],
            page_size,
        )
        st.session_state["match_results_page"] = safe_page

        card_columns = st.columns(2)
        for index, (_, match) in enumerate(page_data.iterrows()):
            team1 = str(match.get("normalized_team1") or match.get("team1") or "Team unavailable")
            team2 = str(match.get("normalized_team2") or match.get("team2") or "Team unavailable")
            winner = str(match.get("normalized_winner") or match.get("winner") or "Winner unavailable")
            venue = str(match.get("venue") or "Venue unavailable")
            match_date = match.get("parsed_date")
            if isinstance(match_date, pd.Timestamp) and not pd.isna(match_date):
                date_text = match_date.strftime("%d %b %Y")
            else:
                date_text = "Date unavailable"

            match_id = match.get("match_id")
            id_text = ""
            if pd.notna(match_id):
                id_text = f'<span class="match-card-id">MATCH {int(match_id)}</span>'

            team1_safe = html.escape(team1)
            team2_safe = html.escape(team2)
            winner_safe = html.escape(winner)
            venue_safe = html.escape(venue)
            date_safe = html.escape(date_text)
            team1_class = (
                "match-card-team match-card-winner"
                if team1 == winner
                else "match-card-team"
            )
            team2_class = (
                "match-card-team match-card-winner"
                if team2 == winner
                else "match-card-team"
            )

            card = card_columns[index % 2]
            card.markdown(
                f"""
                <article class="match-result-card">
                    <div class="match-card-top">
                        <span class="match-card-type">HISTORICAL MATCH</span>
                        <span class="match-card-date">{date_safe}</span>
                    </div>
                    <div class="match-card-teams">
                        <div class="{team1_class}">
                            <strong>{team1_safe}</strong>
                        </div>
                        <span class="match-card-vs">VS</span>
                        <div class="{team2_class}">
                            <strong>{team2_safe}</strong>
                        </div>
                    </div>
                    <div class="match-card-result">
                        <span>WINNER</span>
                        <strong>◆ {winner_safe}</strong>
                        {id_text}
                    </div>
                    <div class="match-card-venue">
                        <span>VENUE</span>
                        <strong>{venue_safe}</strong>
                    </div>
                </article>
                """,
                unsafe_allow_html=True,
            )

            detail_fields: list[str] = []
            for key, label in (
                ("season", "Season"),
                ("city", "City"),
                ("match_type", "Match type"),
                ("toss_winner", "Toss winner"),
                ("toss_decision", "Toss decision"),
                ("result", "Result"),
                ("stage", "Stage"),
            ):
                value = match.get(key)
                if pd.notna(value) and str(value).strip():
                    detail_fields.append(
                        f"{label}: {html.escape(str(value).strip())}"
                    )
            if detail_fields:
                with card.expander(
                    "MATCH DETAILS",
                    expanded=False,
                ):
                    st.markdown(
                        "<br>".join(detail_fields),
                        unsafe_allow_html=True,
                    )

        nav_left, nav_center, nav_right = st.columns([1, 2, 1])
        with nav_left:
            if st.button(
                "← PREVIOUS",
                key="match_results_previous",
                disabled=safe_page <= 1,
            ):
                st.session_state["match_results_page"] = safe_page - 1
                st.rerun()
        with nav_center:
            st.markdown(
                f'<div class="match-results-pagination">PAGE {safe_page} OF {total_pages}</div>',
                unsafe_allow_html=True,
            )
        with nav_right:
            if st.button(
                "NEXT →",
                key="match_results_next",
                disabled=safe_page >= total_pages,
            ):
                st.session_state["match_results_page"] = safe_page + 1
                st.rerun()

    st.markdown("### DATA STATUS")
    c1, c2, c3 = st.columns(3)
    with c1:
        metric("DATA SOURCE", "MATCH METADATA")
    with c2:
        metric("PLAYER STATS", "NOT AVAILABLE")
    with c3:
        metrics = get_selected_model_metrics(production)
        metric(
            "PRODUCTION ACCURACY",
            f"{metrics.get('accuracy', 0) * 100:.2f}%",
            "red",
        )


def probability_chart(
    batting_team: str,
    bowling_team: str,
    batting_probability: float,
    bowling_probability: float,
) -> go.Figure:
    fig = go.Figure(
        go.Bar(
            x=[batting_team, bowling_team],
            y=[batting_probability * 100, bowling_probability * 100],
            text=[
                f"{batting_probability * 100:.1f}%",
                f"{bowling_probability * 100:.1f}%",
            ],
            textposition="outside",
            marker_color=["#ff3b30", "#e6e6e6"],
        )
    )
    fig.update_layout(
        yaxis=dict(range=[0, 100], ticksuffix="%"),
        showlegend=False,
    )
    return plot_theme(fig, 320)


def predict_experimental(
    artifact: dict[str, Any],
    history: pd.DataFrame,
    batting_team: str,
    bowling_team: str,
    toss_winner: str,
    toss_decision: str,
    venue: str,
    prediction_date: pd.Timestamp,
) -> dict[str, Any]:
    features = generate_live_features(
        history,
        batting_team,
        bowling_team,
        venue,
        prediction_date,
    )
    features.update(
        {
            "batting_team": batting_team,
            "bowling_team": bowling_team,
            "venue_key": normalize_venue(venue),
            "toss_winner": toss_winner,
            "toss_decision": toss_decision,
        }
    )
    row = pd.DataFrame([features], columns=artifact["feature_cols"])
    model = artifact["model"]
    probabilities = model.predict_proba(row)[0]
    class_probs = {
        int(cls): float(prob)
        for cls, prob in zip(model.classes_, probabilities)
    }
    batting_probability = class_probs.get(1, 0.0)
    bowling_probability = class_probs.get(0, 1.0 - batting_probability)
    total = batting_probability + bowling_probability
    if total:
        batting_probability /= total
        bowling_probability /= total
    winner = batting_team if batting_probability >= bowling_probability else bowling_team

    return {
        "predicted_winner": winner,
        "confidence": max(batting_probability, bowling_probability),
        "batting_team": batting_team,
        "bowling_team": bowling_team,
        "batting_win_probability": batting_probability,
        "bowling_win_probability": bowling_probability,
        "venue": venue,
        "model_name": artifact["model_name"],
        "model_version": artifact["model_version"],
        "historical_features": features,
    }


def render_prediction(
    production: dict[str, Any],
    experimental: dict[str, Any] | None,
    history: pd.DataFrame,
) -> None:
    render_header("MATCH PREDICTION", "IPL / PRE-MATCH ANALYSIS")

    teams = sorted(set(history["normalized_team1"]).union(set(history["normalized_team2"])))
    teams = [team for team in teams if team in CURRENT_IPL_TEAMS]
    if not teams:
        teams = sorted(set(history["normalized_team1"]).union(set(history["normalized_team2"])))
    venues = sorted(history["venue"].dropna().astype(str).unique())

    model_options = ["Production Model"]
    if experimental:
        model_options.append("Experimental Historical Model")
    model_choice = st.sidebar.selectbox("Prediction model", model_options)

    left, right = st.columns([1.25, 0.75])
    with left:
        team1 = st.selectbox("TEAM 1", teams)
        team2_options = [team for team in teams if team != team1]
        team2 = st.selectbox("TEAM 2", team2_options)
        venue = st.selectbox("VENUE", venues)
        toss_winner = st.selectbox("TOSS WINNER", [team1, team2])
        toss_decision = st.selectbox(
            "TOSS DECISION",
            ["bat", "field"],
            format_func=lambda value: "BAT FIRST" if value == "bat" else "FIELD FIRST",
        )
        prediction_date = st.date_input(
            "PREDICTION DATE",
            value=date.today(),
            min_value=date(2008, 4, 18),
        )
        predict_clicked = st.button(
            "GENERATE PREDICTION",
            type="primary",
            use_container_width=True,
        )

    with right:
        try:
            batting_team, bowling_team = derive_innings_teams(
                team1,
                team2,
                toss_winner,
                toss_decision,
            )
            st.markdown(
                f"""
                <div class="innings-map">
                    <div>1ST INNINGS · BAT</div>
                    <strong>{batting_team}</strong>
                    <div>1ST INNINGS · BOWL</div>
                    <strong>{bowling_team}</strong>
                </div>
                """,
                unsafe_allow_html=True,
            )
        except ValueError as exc:
            st.error(str(exc))

        st.markdown(
            """
            <div class="data-boundary">
                <div>DATA BOUNDARY</div>
                <strong>03 JUN 2025</strong>
                <p>Matches on or after the selected prediction date are excluded.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if prediction_date > DATASET_LAST_UPDATED.date():
            st.warning("Future match results are unavailable and are not fabricated.")

    if not predict_clicked:
        return

    try:
        if model_choice == "Production Model":
            result = predict_match_winner(
                team1=team1,
                team2=team2,
                toss_winner=toss_winner,
                toss_decision=toss_decision,
                venue=venue,
                model_path=DEFAULT_MODEL_PATH,
            )
            result["model_version"] = "production-ipl-v1"
        else:
            result = predict_experimental(
                experimental,
                history,
                batting_team,
                bowling_team,
                toss_winner,
                toss_decision,
                venue,
                pd.Timestamp(prediction_date),
            )

        st.markdown(
            f"""
            <section class="prediction-result">
                <div>
                    <div class="ipl-eyebrow">MODEL ESTIMATE</div>
                    <h2>{result['predicted_winner']}</h2>
                    <p>{max(result['batting_win_probability'], result['bowling_win_probability']) * 100:.1f}% estimated probability</p>
                </div>
                <div class="prediction-version">
                    <span>{result['model_name']}</span>
                    <small>{result['model_version']}</small>
                </div>
            </section>
            """,
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            probability_chart(
                result["batting_team"],
                result["bowling_team"],
                result["batting_win_probability"],
                result["bowling_win_probability"],
            ),
            use_container_width=True,
        )

        c1, c2 = st.columns(2)
        with c1:
            metric(
                f"{result['batting_team']} · BAT",
                f"{result['batting_win_probability'] * 100:.1f}%",
                "red",
            )
        with c2:
            metric(
                f"{result['bowling_team']} · BOWL",
                f"{result['bowling_win_probability'] * 100:.1f}%",
            )

        st.caption(
            "This is a statistical estimate, not a guaranteed outcome."
        )
    except (InvalidMatchInputError, ModelNotFoundError, ValueError) as exc:
        st.error(str(exc))
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")


def render_team_directory(history: pd.DataFrame) -> None:
    render_header("IPL TEAMS", "FRANCHISE / HISTORICAL RECORD")
    team = st.selectbox("SELECT FRANCHISE", CURRENT_IPL_TEAMS)
    profile = team_profile(history, team)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric("MATCHES", str(profile["matches"]))
    with c2:
        metric("WINS", str(profile["wins"]), "red")
    with c3:
        metric("LOSSES", str(profile["losses"]))
    with c4:
        win_pct = (
            f"{profile['win_percentage']}%"
            if profile["win_percentage"] is not None
            else "N/A"
        )
        metric("WIN RATE", win_pct)

    left, right = st.columns([1.15, 0.85])
    with left:
        season_df = profile["season_record"]
        if not season_df.empty:
            fig = px.bar(
                season_df,
                x="season",
                y=["Win", "Loss"],
                barmode="stack",
                color_discrete_map={"Win": "#ff3b30", "Loss": "#777777"},
            )
            st.plotly_chart(plot_theme(fig), use_container_width=True)
    with right:
        venue_df = profile["venue_record"].head(12)
        if not venue_df.empty:
            fig = px.bar(
                venue_df,
                x="Win",
                y="venue",
                orientation="h",
                color_discrete_sequence=["#ff3b30"],
            )
            st.plotly_chart(plot_theme(fig), use_container_width=True)

    data = load_processed_players()
    season_options = (
        sorted(data["team_squads"]["season"].dropna().astype(str).unique(), reverse=True)
        if not data["team_squads"].empty
        else []
    )
    season = st.selectbox(
        "SQUAD SEASON",
        season_options,
        index=0 if season_options else None,
    ) if season_options else None

    squad = get_team_squad(data["team_squads"], team, season)
    st.markdown("### VERIFIED SEASON PARTICIPATION")
    if squad.empty:
        empty_state(
            "NO VERIFIED SQUAD ROWS",
            "No participation rows are available for this team and season.",
        )
    else:
        st.dataframe(
            squad[["player_name", "season", "team", "matches"]],
            use_container_width=True,
            hide_index=True,
        )


def render_players(history: pd.DataFrame) -> None:
    render_header("PLAYERS", "CRICSHEET / VERIFIED REGISTRY")
    data = load_processed_players()
    registry = data["registry"]
    if registry.empty:
        empty_state(
            "NO VERIFIED PLAYER DIRECTORY",
            "Run the Cricsheet processor to generate player_registry.csv.",
        )
        return

    tab_profile, tab_compare = st.tabs(["Player profile", "Player comparison"])
    with tab_profile:
        search = st.text_input("SEARCH PLAYER")
        directory = registry
        if search:
            directory = directory[
                directory["player_name"].str.contains(search, case=False, na=False)
            ]
        if directory.empty:
            empty_state("NO MATCHING PLAYER", "Try another player name.")
            return

        team_filter = st.multiselect(
            "TEAM",
            sorted(set(
                team
                for value in directory["teams"].dropna()
                for team in str(value).split("; ")
                if team
            )),
        )
        if team_filter:
            directory = directory[
                directory["teams"].apply(
                    lambda value: any(team in str(value).split("; ") for team in team_filter)
                )
            ]
        if directory.empty:
            empty_state("NO MATCHING PLAYER", "Try another player name or team.")
            return

        st.markdown("### PLAYER CARDS")
        card_limit = min(len(directory), 12)
        card_columns = st.columns(3)
        for index, (_, player_row) in enumerate(directory.head(card_limit).iterrows()):
            player_name = str(player_row["player_name"])
            player_id = str(player_row["player_id"])
            player_profile = get_player_profile(data, player_id)
            batting_row = player_profile["batting"]
            bowling_row = player_profile["bowling"]
            runs = int(batting_row["runs"]) if batting_row is not None else 0
            wickets = int(bowling_row["wickets"]) if bowling_row is not None else 0
            image_uri = player_image_as_data_uri(player_name)
            if image_uri:
                portrait = f'<img class="player-card-image" src="{image_uri}" alt="{player_name}">'
            else:
                portrait = (
                    '<div class="player-card-fallback">'
                    f"{player_initials(player_name)}"
                    "</div>"
                )
            card_columns[index % 3].markdown(
                f"""
                <article class="player-card">
                    {portrait}
                    <div class="player-card-name">{player_name}</div>
                    <div class="player-card-team">{player_row['teams'] or 'Team unavailable'}</div>
                    <div class="player-card-stats">
                        <span><b>{runs:,}</b> RUNS</span>
                        <span><b>{wickets:,}</b> WKTS</span>
                    </div>
                </article>
                """,
                unsafe_allow_html=True,
            )

        selected_id = st.selectbox(
            "SELECT PLAYER",
            directory["player_id"].astype(str).tolist(),
            format_func=lambda pid: str(
                directory[directory["player_id"].astype(str).eq(pid)]
                ["player_name"]
                .iloc[0]
            ),
        )
        profile = get_player_profile(data, selected_id)
        registry_row = profile["registry"]
        batting_row = profile["batting"]
        bowling_row = profile["bowling"]
        fielding_row = profile["fielding"]
        season_rows = profile["season"]

        if registry_row is not None:
            profile_image = player_image_as_data_uri(str(registry_row["player_name"]))
            portrait_column, identity_column = st.columns([0.25, 1])
            with portrait_column:
                if profile_image:
                    st.markdown(
                        f'<img class="player-profile-image" src="{profile_image}" '
                        f'alt="{registry_row["player_name"]}">',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        '<div class="player-profile-fallback">'
                        f'{player_initials(str(registry_row["player_name"]))}'
                        "</div>",
                        unsafe_allow_html=True,
                    )
            with identity_column:
                st.markdown(
                    f"""
                    <section class="prediction-result player-profile-header">
                        <div>
                            <div class="ipl-eyebrow">VERIFIED PLAYER</div>
                            <h2>{registry_row['player_name']}</h2>
                            <p>{registry_row['stable_source']}</p>
                        </div>
                        <div class="prediction-version">
                            <span>{registry_row['teams'] or 'Team unavailable'}</span>
                            <small>{registry_row['seasons'] or 'Season unavailable'}</small>
                        </div>
                    </section>
                    """,
                    unsafe_allow_html=True,
                )

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            metric(
                "CAREER RUNS",
                str(int(batting_row["runs"])) if batting_row is not None else "N/A",
                "red",
            )
        with c2:
            metric(
                "CAREER WICKETS",
                str(int(bowling_row["wickets"])) if bowling_row is not None else "N/A",
            )
        with c3:
            metric(
                "CATCHES",
                str(int(fielding_row["catches"])) if fielding_row is not None else "N/A",
            )
        with c4:
            metric(
                "MATCHES",
                str(
                    int(
                        max(
                            [
                                int(batting_row["matches"]) if batting_row is not None else 0,
                                int(bowling_row["matches"]) if bowling_row is not None else 0,
                            ]
                        )
                    )
                ),
            )

        left, right = st.columns(2)
        with left:
            st.markdown("### BATTING")
            if batting_row is None:
                st.info("No verified batting record.")
            else:
                batting_metrics = pd.DataFrame(
                    [
                        ("Innings", batting_row["innings"]),
                        ("Runs", batting_row["runs"]),
                        ("Balls faced", batting_row["balls_faced"]),
                        ("Average", batting_row["batting_average"]),
                        ("Strike rate", batting_row["strike_rate"]),
                        ("Highest", batting_row["highest_score"]),
                        ("Fours", batting_row["fours"]),
                        ("Sixes", batting_row["sixes"]),
                        ("50s", batting_row["fifties"]),
                        ("100s", batting_row["hundreds"]),
                    ],
                    columns=["Metric", "Value"],
                )
                batting_metrics["Value"] = batting_metrics["Value"].map(
                    lambda value: "" if pd.isna(value) else str(value)
                )
                st.dataframe(batting_metrics, use_container_width=True, hide_index=True)

        with right:
            st.markdown("### BOWLING")
            if bowling_row is None:
                st.info("No verified bowling record.")
            else:
                bowling_metrics = pd.DataFrame(
                    [
                        ("Innings", bowling_row["innings"]),
                        ("Overs", bowling_row["overs"]),
                        ("Legal balls", bowling_row["legal_balls"]),
                        ("Runs conceded", bowling_row["runs_conceded"]),
                        ("Wickets", bowling_row["wickets"]),
                        ("Average", bowling_row["bowling_average"]),
                        ("Economy", bowling_row["economy"]),
                        ("Strike rate", bowling_row["bowling_strike_rate"]),
                        ("Best", bowling_row["best_bowling"]),
                        ("Maidens", bowling_row["maidens"]),
                    ],
                    columns=["Metric", "Value"],
                )
                bowling_metrics["Value"] = bowling_metrics["Value"].map(
                    lambda value: "" if pd.isna(value) else str(value)
                )
                st.dataframe(bowling_metrics, use_container_width=True, hide_index=True)

        if not season_rows.empty:
            st.markdown("### SEASON PERFORMANCE")
            fig = go.Figure()
            if "batting_runs" in season_rows:
                fig.add_trace(
                    go.Bar(
                        x=season_rows["season"].astype(str),
                        y=season_rows["batting_runs"],
                        name="Runs",
                        marker_color="#ff3333",
                    )
                )
            if "bowling_wickets" in season_rows:
                fig.add_trace(
                    go.Scatter(
                        x=season_rows["season"].astype(str),
                        y=season_rows["bowling_wickets"],
                        name="Wickets",
                        mode="lines+markers",
                        yaxis="y2",
                        line=dict(color="#e8e8e8", width=2),
                    )
                )
            fig.update_layout(
                yaxis=dict(title="Runs"),
                yaxis2=dict(title="Wickets", overlaying="y", side="right"),
                height=390,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="#0b0b0b",
                font=dict(color="#e7e7e7"),
            )
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(
                season_rows[
                    [
                        "season",
                        "team",
                        "matches",
                        "batting_runs",
                        "batting_average",
                        "strike_rate",
                        "bowling_wickets",
                        "bowling_average",
                        "bowling_economy",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )

    with tab_compare:
        ids = st.multiselect(
            "SELECT PLAYERS",
            registry["player_id"].astype(str).tolist(),
            format_func=lambda pid: str(
                registry[registry["player_id"].astype(str).eq(pid)]["player_name"].iloc[0]
            ),
            max_selections=5,
        )
        if len(ids) >= 2:
            selected_batting = data["batting"][
                data["batting"]["player_id"].astype(str).isin(ids)
            ]
            fig = px.bar(
                selected_batting,
                x="player_name",
                y="runs",
                color="player_name",
                title="Career batting runs",
            )
            st.plotly_chart(plot_theme(fig), use_container_width=True)

            selected_bowling = data["bowling"][
                data["bowling"]["player_id"].astype(str).isin(ids)
            ]
            if not selected_bowling.empty:
                fig = px.bar(
                    selected_bowling,
                    x="player_name",
                    y="wickets",
                    color="player_name",
                    title="Career bowling wickets",
                )
                st.plotly_chart(plot_theme(fig), use_container_width=True)
        else:
            st.caption("Select at least two players.")


def render_player_prediction() -> None:
    render_header("PLAYER PERFORMANCE", "HISTORICAL DATA / PROJECTION STATUS")

    data = load_processed_players()
    registry = data["registry"]
    season = data["season"]
    summary = get_predictor_data_summary(registry, season)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric(
            "VERIFIED PLAYERS",
            str(summary["player_count"]),
            "red",
        )
    with c2:
        metric(
            "SEASON / TEAM RECORDS",
            str(summary["season_record_count"]),
        )
    with c3:
        metric(
            "SEASONS COVERED",
            str(summary["season_count"]),
        )
    with c4:
        metric(
            "DATA RANGE",
            f"{summary['earliest_season']} / {summary['latest_season']}",
        )

    st.markdown(
        """
        <section class="ipl-info">
            <strong>Projection status: unavailable</strong>
            <span>Historical statistics available · Future estimates not generated</span>
        </section>
        """,
        unsafe_allow_html=True,
    )

    if not summary["registry_ready"]:
        st.info(
            "Player registry data is missing or empty. "
            "Run the Cricsheet processor first to generate "
            "`data/processed/player_registry.csv`."
        )
        return
    if not summary["season_ready"]:
        st.info(
            "Season-level player data is missing or empty. "
            "Run the Cricsheet processor first to generate "
            "`data/processed/player_season_stats.csv`."
        )
        return

    registry = registry[
        registry["player_id"].notna()
        & registry["player_name"].notna()
    ].copy()
    registry = registry.sort_values(
        ["player_name", "player_id"],
        kind="stable",
    )

    left, right = st.columns([1.25, 0.75])
    with left:
        player_id = st.selectbox(
            "SELECT PLAYER",
            registry["player_id"].astype(str).tolist(),
            format_func=lambda pid: str(
                registry[
                    registry["player_id"].astype(str).eq(pid)
                ]["player_name"].iloc[0]
            ),
        )
    with right:
        player_history = get_player_predictor_history(season, player_id)
        available_seasons = (
            player_history["season"].dropna().astype(str).tolist()
            if not player_history.empty
            else []
        )
        season_filter = st.selectbox(
            "HISTORICAL SEASON",
            ["All available seasons"] + available_seasons,
        )

    if player_history.empty:
        st.info(
            "No season-level batting or bowling record is available for "
            "the selected player."
        )
        return

    if season_filter != "All available seasons":
        player_history = player_history[
            player_history["season"].astype(str).eq(season_filter)
        ].copy()

    player_name = str(
        registry[
            registry["player_id"].astype(str).eq(str(player_id))
        ]["player_name"].iloc[0]
    )

    st.markdown(
        f"""
        <section class="prediction-result">
            <div>
                <div class="ipl-eyebrow">HISTORICAL PLAYER RECORD</div>
                <h2>{player_name}</h2>
                <p>Verified season-level performance from Cricsheet</p>
            </div>
            <div class="prediction-version">
                <span>{len(player_history)} season row(s)</span>
                <small>No future projection generated</small>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric(
            "BATTING RUNS",
            f"{int(player_history['batting_runs'].sum()):,}",
            "red",
        )
    with c2:
        metric(
            "MATCHES",
            f"{int(player_history['matches'].sum()):,}",
        )
    with c3:
        metric(
            "BOWLING WICKETS",
            f"{int(player_history['bowling_wickets'].sum()):,}",
        )
    with c4:
        metric(
            "SEASONS",
            str(player_history["season"].nunique()),
        )

    chart_left, chart_right = st.columns(2)
    with chart_left:
        if player_history["batting_runs"].sum() > 0:
            fig = px.bar(
                player_history,
                x="season",
                y="batting_runs",
                color_discrete_sequence=["#ff3333"],
                title="Season batting runs",
            )
            st.plotly_chart(plot_theme(fig, 360), use_container_width=True)
        else:
            st.info("No batting runs are recorded in this selection.")

    with chart_right:
        if player_history["bowling_wickets"].sum() > 0:
            fig = px.bar(
                player_history,
                x="season",
                y="bowling_wickets",
                color_discrete_sequence=["#e7e7e7"],
                title="Season bowling wickets",
            )
            st.plotly_chart(plot_theme(fig, 360), use_container_width=True)
        else:
            st.info("No bowling wickets are recorded in this selection.")

    st.markdown("### HISTORICAL SEASON TABLE")
    display_history = player_history[
        [
            "season",
            "matches",
            "batting_runs",
            "balls_faced",
            "batting_average",
            "strike_rate",
            "bowling_wickets",
            "bowling_legal_balls",
            "bowling_runs_conceded",
            "bowling_economy",
        ]
    ].rename(
        columns={
            "season": "Season",
            "matches": "Matches",
            "batting_runs": "Runs",
            "balls_faced": "Balls faced",
            "batting_average": "Batting average",
            "strike_rate": "Strike rate",
            "bowling_wickets": "Wickets",
            "bowling_legal_balls": "Legal balls",
            "bowling_runs_conceded": "Runs conceded",
            "bowling_economy": "Economy",
        }
    )
    st.dataframe(
        display_history.round(2),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### PROJECTION REQUIREMENTS")
    st.markdown(
        """
        <div class="ipl-empty">
            <div class="ipl-empty-glyph">.</div>
            <div>
                <strong>Why projections are not shown</strong>
                <p>
                    A reliable future-season estimate requires chronological player
                    validation, enough historical seasons, explicit target metrics,
                    and documented baseline and regression models. This page currently
                    exposes verified historical totals only; it does not fabricate runs,
                    wickets, averages or fantasy points.
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_analytics(history: pd.DataFrame) -> None:
    render_header("ANALYTICS", "MATCH / TEAM / VENUE")

    try:
        history = prepare_match_analytics_history(history)
    except ValueError as exc:
        st.warning(str(exc))
        return

    if history.empty:
        empty_state(
            "NO MATCH DATA AVAILABLE",
            "Team and venue analytics will appear when valid match-history rows are available.",
        )
        return

    tab1, tab2, tab3 = st.tabs(
        ["Team comparison", "Season performance", "Venue distribution"]
    )
    with tab1:
        teams = sorted(set(history["normalized_team1"]).union(set(history["normalized_team2"])))
        if not teams:
            st.info("No valid teams are available for comparison.")
        else:
            selected = st.multiselect(
                "TEAMS",
                teams,
                default=teams[:4],
                max_selections=6,
            )
            comparison = calculate_team_comparison(history, selected)
            if comparison.empty:
                st.info("Select at least one team to view match comparisons.")
            else:
                fig = px.bar(
                    comparison,
                    x="Team",
                    y=["Wins", "Losses"],
                    barmode="stack",
                    color_discrete_map={"Wins": "#ff3b30", "Losses": "#777777"},
                )
                st.plotly_chart(plot_theme(fig), use_container_width=True)

    with tab2:
        season_df = (
            history.dropna(subset=["normalized_winner"])
            .groupby(["season", "normalized_winner"])
            .size()
            .reset_index(name="Wins")
        )
        if season_df.empty:
            st.info("No season-winning records are available.")
        else:
            fig = px.line(
                season_df,
                x="season",
                y="Wins",
                color="normalized_winner",
                markers=True,
            )
            st.plotly_chart(plot_theme(fig, 420), use_container_width=True)

    with tab3:
        venue_df = history["venue"].dropna().astype(str).str.strip()
        venue_df = venue_df[venue_df.ne("")].value_counts().rename_axis("Venue").reset_index(name="Matches")
        if venue_df.empty:
            st.info("No venue records are available.")
        else:
            fig = px.bar(
                venue_df.head(20),
                x="Matches",
                y="Venue",
                orientation="h",
                color_discrete_sequence=["#ff3b30"],
            )
            st.plotly_chart(plot_theme(fig, 520), use_container_width=True)


def render_model_intelligence(
    production: dict[str, Any],
    experimental: dict[str, Any] | None,
) -> None:
    st.markdown(
        """
        <header class="model-intelligence-header">
            <div>
                <div class="ipl-eyebrow">ML MODEL EVALUATION</div>
                <h1>MODEL INTELLIGENCE</h1>
                <p>IPL Match Prediction — Model Performance &amp; Evaluation</p>
            </div>
            <div class="model-status-group">
                <div class="model-status active"><span></span><b>Production Model</b><small>Active</small></div>
                <div class="model-status evaluation"><span></span><b>Experimental Models</b><small>Evaluation Only</small></div>
            </div>
        </header>
        """,
        unsafe_allow_html=True,
    )

    production_metrics = get_selected_model_metrics(production)
    accuracy = production_metrics.get("accuracy")
    precision = production_metrics.get("precision")
    recall = production_metrics.get("recall")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f"""
            <article class="model-kpi model-kpi-primary">
                <div class="model-kpi-label">ACCURACY</div>
                <div class="model-kpi-value">{accuracy * 100:.2f}%</div>
                <div class="model-kpi-caption">Share of all predictions correct</div>
            </article>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <article class="model-kpi">
                <div class="model-kpi-label">PRECISION</div>
                <div class="model-kpi-value">{precision * 100:.2f}%</div>
                <div class="model-kpi-caption">Batting-win predictions confirmed</div>
            </article>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""
            <article class="model-kpi">
                <div class="model-kpi-label">RECALL</div>
                <div class="model-kpi-value">{recall * 100:.2f}%</div>
                <div class="model-kpi-caption">Actual batting wins identified</div>
            </article>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### PRODUCTION MODEL")
    st.markdown(
        f"""
        <section class="model-production-panel">
            <div class="model-production-main">
                <div class="ipl-eyebrow">ACTIVE MODEL</div>
                <h2>{production.get('model_name', 'Unknown')}</h2>
                <p>The production match-prediction pipeline currently used by the application.</p>
            </div>
            <div class="model-production-meta">
                <div><span>Status</span><strong>Production · Active</strong></div>
                <div><span>Input features</span><strong>5 categorical match features</strong></div>
                <div><span>Evaluation</span><strong>Existing chronological 80/20 test</strong></div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    if not experimental:
        st.info("Experimental evaluation data is unavailable.")
        st.markdown("### MODEL LIMITATIONS")
        empty_state(
            "DATASET LIMITATION",
            "Match-level data ends on 3 June 2025. No live 2026 match, player, pitch or lineup "
            "data is included.",
        )
        return

    validation_rows = experimental.get("validation_results", [])
    final_rows = experimental.get("final_results", [])
    if not validation_rows and not final_rows:
        st.info("No experimental model metrics are available.")
        return

    st.markdown("### EXPERIMENTAL MODEL COMPARISON")

    comparison_source = pd.DataFrame(final_rows or validation_rows)
    if comparison_source.empty:
        st.info("Experimental comparison data is empty.")
        return

    available_metrics: list[tuple[str, str, bool]] = [
        ("accuracy", "Accuracy", True),
        ("balanced_accuracy", "Balanced Accuracy", True),
        ("precision", "Precision", True),
        ("recall", "Recall", True),
        ("f1", "F1 Score", True),
        ("roc_auc", "ROC-AUC", True),
        ("brier", "Brier Score", False),
    ]
    selectable_metrics = [
        (key, label, higher_is_better)
        for key, label, higher_is_better in available_metrics
        if key in comparison_source.columns
        and comparison_source[key].notna().any()
    ]
    if not selectable_metrics:
        st.info("No selectable experimental metrics are available.")
        return

    metric_labels = [label for _, label, _ in selectable_metrics]
    label_to_definition = {
        label: (key, higher_is_better)
        for key, label, higher_is_better in selectable_metrics
    }
    selected_label = st.selectbox(
        "METRIC",
        metric_labels,
        index=0,
    )
    selected_key, higher_is_better = label_to_definition[selected_label]

    ranked = comparison_source[
        ["model", selected_key]
    ].dropna().sort_values(
        selected_key,
        ascending=not higher_is_better,
    )

    rank_left, rank_right = st.columns([1.55, 0.45])
    with rank_left:
        fig = px.bar(
            ranked,
            x=selected_key,
            y="model",
            orientation="h",
            text=selected_key,
            color="model",
            color_discrete_sequence=[
                "#ff3333",
                "#d8d8d8",
                "#a5a5a5",
                "#777777",
                "#555555",
                "#333333",
            ],
        )
        fig.update_traces(
            texttemplate=(
                "%{text:.3f}"
                if selected_key in {"brier", "roc_auc"}
                else "%{text:.2%}"
            ),
            textposition="outside",
            cliponaxis=False,
            hovertemplate="<b>%{y}</b><br>"
                          f"{selected_label}: %{{x:.4f}}"
                          "<extra></extra>",
        )
        fig.update_layout(
            title=f"{selected_label} — 2025 Final Holdout",
            height=430,
            margin=dict(l=180, r=45, t=65, b=45),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#0b0b0b",
            font=dict(family="Inter, Arial, sans-serif", color="#e7e7e7"),
            showlegend=False,
            yaxis=dict(
                categoryorder="array",
                categoryarray=ranked["model"].tolist(),
                autorange="reversed",
                gridcolor="#1f1f1f",
            ),
            xaxis=dict(
                gridcolor="#232323",
                zeroline=False,
                rangemode="tozero",
            ),
        )
        st.plotly_chart(fig, use_container_width=True)

    with rank_right:
        st.markdown('<div class="model-ranking">', unsafe_allow_html=True)
        st.markdown(
            f"#### {selected_label} ranking"
        )
        for position, (_, row) in enumerate(ranked.iterrows(), start=1):
            value = float(row[selected_key])
            display_value = (
                f"{value:.4f}"
                if selected_key in {"brier", "roc_auc"}
                else f"{value * 100:.2f}%"
            )
            st.markdown(
                f"""
                <div class="model-rank-row">
                    <span>{position:02d}</span>
                    <div>
                        <strong>{row['model']}</strong>
                        <small>{display_value}</small>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

    validation_frame = pd.DataFrame(validation_rows)
    final_frame = pd.DataFrame(final_rows)
    if not validation_frame.empty and not final_frame.empty:
        st.markdown("### VALIDATION VS HOLDOUT")
        period_frame = pd.merge(
            validation_frame,
            final_frame,
            on="model",
            suffixes=("_2024_validation", "_2025_holdout"),
        )
        chart_frame = period_frame[
            ["model", "accuracy_2024_validation", "accuracy_2025_holdout"]
        ].dropna()
        chart = px.bar(
            chart_frame,
            x="model",
            y=["accuracy_2024_validation", "accuracy_2025_holdout"],
            barmode="group",
            labels={
                "value": "Accuracy",
                "variable": "Evaluation period",
                "model": "Model",
            },
            color_discrete_map={
                "accuracy_2024_validation": "#d8d8d8",
                "accuracy_2025_holdout": "#ff3333",
            },
        )
        chart.update_layout(
            title="Accuracy — 2024 Validation vs 2025 Holdout",
            height=390,
            margin=dict(l=55, r=25, t=65, b=90),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#0b0b0b",
            font=dict(family="Inter, Arial, sans-serif", color="#e7e7e7"),
            legend_title_text="Evaluation period",
            xaxis=dict(tickangle=-25),
            yaxis=dict(
                tickformat=".0%",
                gridcolor="#232323",
                rangemode="tozero",
            ),
        )
        chart.update_traces(
            hovertemplate="<b>%{x}</b><br>%{fullData.name}: %{y:.2%}<extra></extra>",
        )
        st.plotly_chart(chart, use_container_width=True)

    st.markdown("### DETAILED MODEL EVALUATION")
    st.markdown(
        f"""
        <section class="model-production-reference">
            <div>
                <span>Production reference</span>
                <strong>{production.get('model_name', 'Logistic Regression')}</strong>
            </div>
            <div><span>Existing held-out accuracy</span><strong>{production_metrics.get('accuracy', 0) * 100:.2f}%</strong></div>
            <div><span>Precision</span><strong>{production_metrics.get('precision', 0) * 100:.2f}%</strong></div>
            <div><span>Recall</span><strong>{production_metrics.get('recall', 0) * 100:.2f}%</strong></div>
            <div><span>Status</span><strong>Active · 80/20 production period</strong></div>
        </section>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        "The table below shows experimental models only on the reserved 2025 final holdout."
    )
    detail_frame = final_frame.copy()
    if detail_frame.empty:
        detail_frame = validation_frame.copy()

    readable_columns = [
        ("model", "Model"),
        ("rows", "Rows"),
        ("accuracy", "Accuracy"),
        ("balanced_accuracy", "Balanced Accuracy"),
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("f1", "F1 Score"),
        ("roc_auc", "ROC-AUC"),
        ("brier", "Brier Score"),
        ("log_loss", "Log Loss"),
    ]
    available_detail_columns = [
        (key, label)
        for key, label in readable_columns
        if key in detail_frame.columns
    ]
    display_detail = detail_frame[[key for key, _ in available_detail_columns]].copy()
    display_detail.columns = [label for _, label in available_detail_columns]

    percent_columns = {
        "Accuracy",
        "Balanced Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC-AUC",
    }
    for column in percent_columns.intersection(display_detail.columns):
        display_detail[column] = display_detail[column].map(
            lambda value: f"{float(value) * 100:.2f}%"
        )
    for column in {"Brier Score", "Log Loss"}.intersection(display_detail.columns):
        display_detail[column] = display_detail[column].map(
            lambda value: f"{float(value):.4f}"
        )

    st.dataframe(
        display_detail,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### MODEL EVALUATION INSIGHTS")
    insight_cards: list[tuple[str, str]] = []

    highest_holdout = detail_frame.sort_values(
        "accuracy",
        ascending=False,
    ).iloc[0]
    insight_cards.append(
        (
            "HIGHEST HOLDOUT ACCURACY",
            f"{highest_holdout['model']} — "
            f"{highest_holdout['accuracy'] * 100:.2f}% "
            "on the available final holdout.",
        )
    )

    if "balanced_accuracy" in detail_frame.columns:
        highest_balanced = detail_frame.sort_values(
            "balanced_accuracy",
            ascending=False,
        ).iloc[0]
        insight_cards.append(
            (
                "HIGHEST BALANCED ACCURACY",
                f"{highest_balanced['model']} — "
                f"{highest_balanced['balanced_accuracy'] * 100:.2f}%.",
            )
        )

    baseline_row = detail_frame[
        detail_frame["model"].str.contains("Majority", case=False, na=False)
    ]
    if not baseline_row.empty and "accuracy" in baseline_row.columns:
        baseline_accuracy = float(baseline_row.iloc[0]["accuracy"])
        production_accuracy = float(production_metrics["accuracy"])
        insight_cards.append(
            (
                "PRODUCTION VS BASELINE",
                f"Production reports {production_accuracy * 100:.2f}% accuracy on "
                f"the existing chronological test period. The majority baseline "
                f"reports {baseline_accuracy * 100:.2f}% on the experimental 2025 "
                "holdout; these periods are not identical and are not directly comparable.",
            )
        )

    insight_html = "".join(
        f"""
        <article class="model-insight-card">
            <div class="model-insight-label">{label}</div>
            <p>{text}</p>
        </article>
        """
        for label, text in insight_cards
    )
    st.markdown(
        f'<section class="model-insights-grid">{insight_html}</section>',
        unsafe_allow_html=True,
    )

    st.markdown("### MODEL LIMITATIONS")
    st.markdown(
        """
        <section class="model-limitations">
            <div>
                <div class="ipl-eyebrow">DATA &amp; FEATURES</div>
                <p>Match-level summary data and five categorical features provide limited predictive signal.</p>
            </div>
            <div>
                <div class="ipl-eyebrow">VALIDATION</div>
                <p>Validation performance does not guarantee final holdout performance or future match results.</p>
            </div>
            <div>
                <div class="ipl-eyebrow">REPLACEMENT RULE</div>
                <p>An experimental model should replace production only after independent holdout validation and stable deployment checks.</p>
            </div>
            <div>
                <div class="ipl-eyebrow">PROBABILITY</div>
                <p>A match probability is a model estimate; it is not a calibrated guarantee or a result certainty.</p>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    inject_css()

    st.sidebar.markdown(
        '<div class="sidebar-brand"><span>I</span> IPL / INTELLIGENCE</div>',
        unsafe_allow_html=True,
    )
    page = st.sidebar.radio(
        "MODULE",
        [
            "Overview",
            "Match Prediction",
            "IPL Teams",
            "Players",
            "Player Performance Predictor",
            "Analytics",
            "Model Intelligence",
        ],
        label_visibility="collapsed",
    )

    history = load_history()
    player_history = load_player_history()
    experimental = load_experimental_artifact()

    try:
        production = load_model_artifact(DEFAULT_MODEL_PATH)
    except Exception as exc:
        st.error(f"Production model could not be loaded: {exc}")
        st.stop()

    if page == "Overview":
        render_overview(history, production)
    elif page == "Match Prediction":
        render_prediction(production, experimental, history)
    elif page == "IPL Teams":
        render_team_directory(player_history)
    elif page == "Players":
        render_players(player_history)
    elif page == "Player Performance Predictor":
        render_player_prediction()
    elif page == "Analytics":
        render_analytics(history)
    else:
        render_model_intelligence(production, experimental)

    st.markdown(
        '<footer class="ipl-footer">IPL / INTELLIGENCE · MATCH DATA THROUGH 03 JUN 2025</footer>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
