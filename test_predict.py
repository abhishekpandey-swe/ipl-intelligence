from __future__ import annotations

from predict import (
    DEFAULT_MODEL_PATH,
    InvalidMatchInputError,
    load_model_artifact,
    predict_match_winner,
)
from train_model import get_selected_model_metrics


def run_tests() -> None:
    artifact = load_model_artifact(DEFAULT_MODEL_PATH)
    teams = artifact.get("teams", [])
    venues = artifact.get("venues", [])
    metrics = get_selected_model_metrics(artifact)

    assert len(teams) >= 2, "Need at least 2 teams in trained artifact."
    assert len(venues) >= 1, "Need at least 1 venue in trained artifact."
    for key in ("accuracy", "precision", "recall"):
        assert key in metrics, f"Missing selected model metric '{key}'."

    t1, t2 = teams[0], teams[1]
    venue = venues[0]

    res_bat = predict_match_winner(
        team1=t1,
        team2=t2,
        toss_winner=t1,
        toss_decision="bat",
        venue=venue,
    )
    assert res_bat["batting_team"] == t1
    assert res_bat["bowling_team"] == t2
    assert abs((res_bat["batting_win_probability"] + res_bat["bowling_win_probability"]) - 1.0) < 1e-6

    res_field = predict_match_winner(
        team1=t1,
        team2=t2,
        toss_winner=t1,
        toss_decision="field",
        venue=venue,
    )
    assert res_field["batting_team"] == t2
    assert res_field["bowling_team"] == t1

    try:
        predict_match_winner(
            team1=t1,
            team2=t1,
            toss_winner=t1,
            toss_decision="bat",
            venue=venue,
        )
        raise AssertionError("Expected InvalidMatchInputError for identical teams.")
    except InvalidMatchInputError:
        pass

    try:
        predict_match_winner(
            team1=t1,
            team2=t2,
            toss_winner="Non Existent Team",
            toss_decision="bat",
            venue=venue,
        )
        raise AssertionError("Expected InvalidMatchInputError for invalid toss winner.")
    except InvalidMatchInputError:
        pass

    print("PASS: All prediction, metric lookup, and input validation checks succeeded.")
    print(
        f"Selected Model ({artifact.get('model_name')}): "
        f"Acc={metrics['accuracy']*100:.2f}%, Prec={metrics['precision']*100:.2f}%, Rec={metrics['recall']*100:.2f}%"
    )
    print(
        f"Sample prediction ({t1} vs {t2} at {venue}): "
        f"Winner={res_bat['predicted_winner']} ({res_bat['confidence']*100:.1f}%)"
    )


if __name__ == "__main__":
    run_tests()
