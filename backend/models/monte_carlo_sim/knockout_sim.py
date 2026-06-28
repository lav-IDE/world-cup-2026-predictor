from __future__ import annotations
import random
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from backend.models.monte_carlo_sim import tournament_structure as ts
from backend.models.monte_carlo_sim.sim_features import shootout_probability


@dataclass
class KnockoutMatchResult:
    match_id: str
    home_team: str
    away_team: str
    winner: str
    loser: str
    home_score: int
    away_score: int
    went_to_shootout: bool
    neutral: bool
    decided_by: str  # "model_agreement" or "elo_tiebreak"


def _decide_home_away(match_id: str, team_a: str, team_b: str) -> tuple[str, str, bool]:

    home_adv_team = ts.get_home_advantage_team(match_id, team_a, team_b)
    if home_adv_team == team_a:
        return team_a, team_b, False
    elif home_adv_team == team_b:
        return team_b, team_a, False
    else:
        return team_a, team_b, True


def _round_to_scoreline(home_score_pred: float, away_score_pred: float,
                         decided_winner: str, home_team: str, away_team: str
                         ) -> tuple[int, int]:

    home_int = max(0, round(home_score_pred))
    away_int = max(0, round(away_score_pred))

    if home_int == away_int:
        if decided_winner == home_team:
            home_int += 1
        else:
            away_int += 1

    return home_int, away_int


def _resolve_knockout_result(
    match_id: str,
    home_team: str,
    away_team: str,
    neutral: bool,
    prediction: dict,
    sim_feature_state,
) -> KnockoutMatchResult:
    home_win_renorm = prediction["home_win"] / (prediction["home_win"] + prediction["away_win"])
    classifier_winner = home_team if random.random() < home_win_renorm else away_team

    home_score_pred = prediction["home_score"]
    away_score_pred = prediction["away_score"]
    regressor_winner = home_team if home_score_pred > away_score_pred else away_team

    predicted_margin = abs(home_score_pred - away_score_pred)

    if classifier_winner == regressor_winner:
        decided_winner = classifier_winner
        decided_by = "model_agreement"
    else:
        home_exp, away_exp = sim_feature_state.elo_system.get_expected(
            home_team, away_team, neutral
        )
        decided_winner = home_team if random.random() < home_exp else away_team
        decided_by = "elo_tiebreak"
        predicted_margin = 0.0

    went_to_shootout = random.random() < shootout_probability(predicted_margin)

    if went_to_shootout:
        level_score = max(0, round(max(home_score_pred, away_score_pred)))
        home_score, away_score = level_score, level_score
        shootout_winner = decided_winner
    else:
        home_score, away_score = _round_to_scoreline(
            home_score_pred, away_score_pred, decided_winner, home_team, away_team
        )
        shootout_winner = None

    loser = away_team if decided_winner == home_team else home_team

    sim_feature_state.apply_match_result(
        date=pd.Timestamp(ts.KNOCKOUT_SCHEDULE[ts._match_id_to_int(match_id)]["date"]),
        home_team=home_team,
        away_team=away_team,
        home_score=home_score,
        away_score=away_score,
        neutral=neutral,
        tournament="FIFA World Cup",
        shootout_winner=shootout_winner,
    )

    return KnockoutMatchResult(
        match_id=match_id,
        home_team=home_team,
        away_team=away_team,
        winner=decided_winner,
        loser=loser,
        home_score=home_score,
        away_score=away_score,
        went_to_shootout=went_to_shootout,
        neutral=neutral,
        decided_by=decided_by,
    )


def _play_round(
    matchups: dict[str, tuple[str, str]],
    sim_feature_state,
    predictor,
) -> dict[str, KnockoutMatchResult]:

    match_ids = list(matchups.keys())
    decided = [_decide_home_away(mid, *matchups[mid]) for mid in match_ids]

    batch_input = [(home, away, neutral) for home, away, neutral in decided]
    predictions = predictor.predict_batch(batch_input, sim_feature_state)

    results = {}
    for match_id, (home_team, away_team, neutral), prediction in zip(
        match_ids, decided, predictions
    ):
        results[match_id] = _resolve_knockout_result(
            match_id, home_team, away_team, neutral, prediction, sim_feature_state
        )
    return results


def simulate_bracket(
    round_of_32_matchups: dict[str, tuple[str, str]],
    sim_feature_state,
    predictor,
) -> dict[str, KnockoutMatchResult]:

    results: dict[str, KnockoutMatchResult] = {}

    results.update(_play_round(round_of_32_matchups, sim_feature_state, predictor))

    r16_matchups = {
        match_id: (results[feeder_a].winner, results[feeder_b].winner)
        for match_id, (feeder_a, feeder_b) in ts.R16_FROM_R32.items()
    }
    results.update(_play_round(r16_matchups, sim_feature_state, predictor))

    qf_matchups = {
        match_id: (results[feeder_a].winner, results[feeder_b].winner)
        for match_id, (feeder_a, feeder_b) in ts.QF_FROM_R16.items()
    }
    results.update(_play_round(qf_matchups, sim_feature_state, predictor))

    sf_matchups = {
        match_id: (results[feeder_a].winner, results[feeder_b].winner)
        for match_id, (feeder_a, feeder_b) in ts.SF_FROM_QF.items()
    }
    results.update(_play_round(sf_matchups, sim_feature_state, predictor))

    final_matchups = {
        match_id: (results[feeder_a].winner, results[feeder_b].winner)
        for match_id, (feeder_a, feeder_b) in ts.FINAL_FROM_SF.items()
    }
    third_place_matchups = {
        match_id: (results[feeder_a].loser, results[feeder_b].loser)
        for match_id, (feeder_a, feeder_b) in ts.THIRD_PLACE_MATCH_FROM_SF.items()
    }
    results.update(_play_round({**final_matchups, **third_place_matchups},
                                sim_feature_state, predictor))

    return results