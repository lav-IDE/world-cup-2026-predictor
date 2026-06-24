from __future__ import annotations
import copy
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Deque, Optional


# ---------------------------------------------------------------------------
# Incremental rolling stats
# ---------------------------------------------------------------------------

@dataclass
class MatchRecord:
    """One match from a single team's perspective — same shape as a row
    of recent_stats.get_team_history()'s output."""
    date: object  # pd.Timestamp or str — kept opaque, only used for ordering
    opponent: str
    goals_scored: int
    goals_conceded: int
    result: str  # "win" | "draw" | "loss"


class RollingStatsTracker:

    def __init__(self, window: int):
        self.window = window
        self._records: Deque[MatchRecord] = deque(maxlen=window)

    def seed(self, records: list[MatchRecord]) -> None:
        for r in records:
            self._records.append(r)

    def add_match(self, record: MatchRecord) -> None:
        self._records.append(record)

    def current_stats(self) -> Dict[str, Optional[float]]:
        if len(self._records) == 0:
            return {
                "avg_goals_scored": None,
                "avg_goals_conceded": None,
                "win_rate": None,
                "draw_rate": None,
                "loss_rate": None,
            }

        n = len(self._records)
        total_scored = sum(r.goals_scored for r in self._records)
        total_conceded = sum(r.goals_conceded for r in self._records)
        wins = sum(1 for r in self._records if r.result == "win")
        draws = sum(1 for r in self._records if r.result == "draw")
        losses = sum(1 for r in self._records if r.result == "loss")

        return {
            "avg_goals_scored": total_scored / n,
            "avg_goals_conceded": total_conceded / n,
            "win_rate": wins / n,
            "draw_rate": draws / n,
            "loss_rate": losses / n,
        }

    def __deepcopy__(self, memo):
        new = RollingStatsTracker(self.window)
        new._records = deque(copy.deepcopy(list(self._records), memo), maxlen=self.window)
        return new


def derive_match_records(
    date: object,
    home_team: str,
    away_team: str,
    home_score: int,
    away_score: int,
) -> tuple[MatchRecord, MatchRecord]:

    if home_score == away_score:
        winner = "Draw"
    elif home_score > away_score:
        winner = home_team
    else:
        winner = away_team

    def _result_for(team: str) -> str:
        if winner == "Draw":
            return "draw"
        return "win" if winner == team else "loss"

    home_record = MatchRecord(
        date=date,
        opponent=away_team,
        goals_scored=home_score,
        goals_conceded=away_score,
        result=_result_for(home_team),
    )
    away_record = MatchRecord(
        date=date,
        opponent=home_team,
        goals_scored=away_score,
        goals_conceded=home_score,
        result=_result_for(away_team),
    )
    return home_record, away_record


# ---------------------------------------------------------------------------
# Per-iteration simulation state
# ---------------------------------------------------------------------------

@dataclass
class SimFeatureState:
    elo_system: object  # your real EloSystem instance
    rolling_trackers: Dict[str, RollingStatsTracker] = field(default_factory=dict)

    def clone_for_iteration(self) -> "SimFeatureState":

        return SimFeatureState(
            elo_system=copy.deepcopy(self.elo_system),
            rolling_trackers={
                team: copy.deepcopy(tracker)
                for team, tracker in self.rolling_trackers.items()
            },
        )

    def get_features(self, team: str) -> dict:

        elo_rating = self.elo_system.ratings.get(
            team, self.elo_system.initial_rating
        )
        tracker = self.rolling_trackers.get(team)
        rolling = tracker.current_stats() if tracker else {
            "avg_goals_scored": None,
            "avg_goals_conceded": None,
            "win_rate": None,
            "draw_rate": None,
            "loss_rate": None,
        }
        return {"elo": elo_rating, **rolling}

    def apply_match_result(
        self,
        date: object,
        home_team: str,
        away_team: str,
        home_score: int,
        away_score: int,
        neutral: bool,
        tournament: str,
        shootout_winner: Optional[str] = None,
    ) -> None:

        # --- Elo update: build the row shape EloSystem.update_ratings expects ---
        row = {
            "home_team": home_team,
            "away_team": away_team,
            "date": date,
            "neutral": neutral,
            "home_score": home_score,
            "away_score": away_score,
            "shootout_winner": shootout_winner,
            "tournament": tournament,
        }
        self.elo_system.update_ratings(row)

        # --- Rolling stats update ---
        home_record, away_record = derive_match_records(
            date, home_team, away_team, home_score, away_score
        )
        self.rolling_trackers.setdefault(
            home_team, RollingStatsTracker(window=self._default_window())
        ).add_match(home_record)
        self.rolling_trackers.setdefault(
            away_team, RollingStatsTracker(window=self._default_window())
        ).add_match(away_record)

    def _default_window(self) -> int:
        for tracker in self.rolling_trackers.values():
            return tracker.window
        raise RuntimeError(
            "No rolling-stats window size available — SimFeatureState was "
            "constructed without any seeded trackers. Pass `window` "
            "explicitly via build_sim_feature_state() instead of "
            "constructing SimFeatureState directly."
        )


# ---------------------------------------------------------------------------
# Knockout shootout probability
# ---------------------------------------------------------------------------

import math

SHOOTOUT_BASE_RATE = 0.21  # probability of a shootout when predicted goal_diff == 0
SHOOTOUT_DECAY_SCALE = 1.0  # goals; TUNABLE, no historical basis


def shootout_probability(predicted_goal_diff: float) -> float:
    return SHOOTOUT_BASE_RATE * math.exp(
        -(predicted_goal_diff ** 2) / (2 * SHOOTOUT_DECAY_SCALE ** 2)
    )


def build_sim_feature_state(elo_system, window: int) -> SimFeatureState:
    state = SimFeatureState(elo_system=elo_system)
    for team in elo_system.ratings.keys():
        state.rolling_trackers[team] = RollingStatsTracker(window=window)
    return state