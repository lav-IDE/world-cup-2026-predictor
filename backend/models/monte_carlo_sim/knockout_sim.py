from __future__ import annotations
import random
from dataclasses import dataclass
from typing import Optional

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


def simulate_knockout_match(
    match_id: str,
    team_a: str,
    team_b: str,
    sim_feature_state,
    predictor,
) -> KnockoutMatchResult:
    home_team, away_team, neutral = _decide_home_away(match_id, team_a, team_b)

    prediction = predictor.predict(home_team, away_team, sim_feature_state, neutral)

    # Step 3: renormalize classifier probs excluding draw.
    home_win_renorm = prediction["home_win"] / (prediction["home_win"] + prediction["away_win"])
    classifier_winner = home_team if random.random() < home_win_renorm else away_team

    # Step 4: regressor-implied winner, directly from predicted scores.
    home_score_pred = prediction["home_score"]
    away_score_pred = prediction["away_score"]
    regressor_winner = home_team if home_score_pred > away_score_pred else away_team

    predicted_margin = abs(home_score_pred - away_score_pred)

    if classifier_winner == regressor_winner:
        decided_winner = classifier_winner
        decided_by = "model_agreement"
    else:
        # Step 5: Elo tie-break on genuine model disagreement.
        home_exp, away_exp = sim_feature_state.elo_system.get_expected(
            home_team, away_team, neutral
        )
        decided_winner = home_team if random.random() < home_exp else away_team
        decided_by = "elo_tiebreak"
        # A genuine model disagreement is treated as an effectively even
        # match for shootout-probability purposes (predicted_margin from
        # two disagreeing models isn't a trustworthy margin estimate).
        predicted_margin = 0.0

    # Step 6: shootout decision.
    went_to_shootout = random.random() < shootout_probability(predicted_margin)

    # Step 7: construct the actual scoreline.
    if went_to_shootout:
        # Scoreline must be level before shootout_winner resolves it --
        # use the rounded predicted scores but force equality, taking
        # the higher of the two rounded values as the level scoreline
        # (a tighter shootout-bound match is more plausible at a higher
        # scoreline than artificially forcing 0-0 regardless of the
        # model's own goal expectations).
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
        date=ts.KNOCKOUT_SCHEDULE[ts._match_id_to_int(match_id)]["date"],
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


def simulate_bracket(
    round_of_32_matchups: dict[str, tuple[str, str]],
    sim_feature_state,
    predictor,
) -> dict[str, KnockoutMatchResult]:

    results: dict[str, KnockoutMatchResult] = {}

    def _play(match_id: str, team_a: str, team_b: str) -> KnockoutMatchResult:
        result = simulate_knockout_match(match_id, team_a, team_b, sim_feature_state, predictor)
        results[match_id] = result
        return result

    # Round of 32
    for match_id, (team_a, team_b) in round_of_32_matchups.items():
        _play(match_id, team_a, team_b)

    # Round of 16 -- winners of R32 feed in
    for match_id, (feeder_a, feeder_b) in ts.R16_FROM_R32.items():
        team_a = results[feeder_a].winner
        team_b = results[feeder_b].winner
        _play(match_id, team_a, team_b)

    # Quarter-finals -- winners of R16 feed in
    for match_id, (feeder_a, feeder_b) in ts.QF_FROM_R16.items():
        team_a = results[feeder_a].winner
        team_b = results[feeder_b].winner
        _play(match_id, team_a, team_b)

    # Semi-finals -- winners of QF feed in
    for match_id, (feeder_a, feeder_b) in ts.SF_FROM_QF.items():
        team_a = results[feeder_a].winner
        team_b = results[feeder_b].winner
        _play(match_id, team_a, team_b)

    # Final -- winners of SF feed in
    for match_id, (feeder_a, feeder_b) in ts.FINAL_FROM_SF.items():
        team_a = results[feeder_a].winner
        team_b = results[feeder_b].winner
        _play(match_id, team_a, team_b)

    # Third Place Match -- LOSERS of SF feed in (not winners)
    for match_id, (feeder_a, feeder_b) in ts.THIRD_PLACE_MATCH_FROM_SF.items():
        team_a = results[feeder_a].loser
        team_b = results[feeder_b].loser
        _play(match_id, team_a, team_b)

    return results