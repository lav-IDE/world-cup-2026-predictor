from __future__ import annotations
import random
from dataclasses import dataclass
from itertools import combinations

from backend.models.monte_carlo_sim import tournament_structure as ts


@dataclass
class GroupMatchResult:
    home_team: str
    away_team: str
    home_score: int
    away_score: int


@dataclass
class TeamStanding:
    team: str
    group: str
    points: int = 0
    goals_for: int = 0
    goals_against: int = 0
    wins: int = 0
    draws: int = 0
    losses: int = 0

    @property
    def goal_diff(self) -> int:
        return self.goals_for - self.goals_against


def simulate_group_match(home_team: str, away_team: str, date: str,
                          sim_feature_state, predictor) -> GroupMatchResult:
    prediction = predictor.predict(home_team, away_team, sim_feature_state, neutral=False)

    outcome = random.choices(
        ["home", "draw", "away"],
        weights=[prediction["home_win"], prediction["draw"], prediction["away_win"]],
    )[0]

    home_pred, away_pred = prediction["home_score"], prediction["away_score"]

    if outcome == "draw":
        level = max(0, round(max(home_pred, away_pred)))
        home_score, away_score = level, level
    else:
        home_score = max(0, round(home_pred))
        away_score = max(0, round(away_pred))
        if outcome == "home" and home_score <= away_score:
            home_score = away_score + 1
        elif outcome == "away" and away_score <= home_score:
            away_score = home_score + 1

    sim_feature_state.apply_match_result(
        date=date,
        home_team=home_team,
        away_team=away_team,
        home_score=home_score,
        away_score=away_score,
        neutral=False,
        tournament="FIFA World Cup",
        shootout_winner=None,
    )

    return GroupMatchResult(home_team, away_team, home_score, away_score)


def simulate_group(group_letter: str, sim_feature_state, predictor) -> list[GroupMatchResult]:
    teams = ts.GROUPS[group_letter]
    results = []
    for i, (team_a, team_b) in enumerate(combinations(teams, 2)):
        date = f"2026-06-{12 + i}"
        results.append(simulate_group_match(team_a, team_b, date, sim_feature_state, predictor))
    return results


def _build_standings(group_letter: str, match_results: list[GroupMatchResult]
                      ) -> dict[str, TeamStanding]:
    standings = {team: TeamStanding(team=team, group=group_letter)
                 for team in ts.GROUPS[group_letter]}

    for r in match_results:
        home, away = standings[r.home_team], standings[r.away_team]
        home.goals_for += r.home_score
        home.goals_against += r.away_score
        away.goals_for += r.away_score
        away.goals_against += r.home_score

        if r.home_score > r.away_score:
            home.points += 3
            home.wins += 1
            away.losses += 1
        elif r.home_score < r.away_score:
            away.points += 3
            away.wins += 1
            home.losses += 1
        else:
            home.points += 1
            away.points += 1
            home.draws += 1
            away.draws += 1

    return standings


def _head_to_head_points(team_a: str, team_b: str, match_results: list[GroupMatchResult]
                          ) -> tuple[int, int]:
    a_points = b_points = 0
    for r in match_results:
        if {r.home_team, r.away_team} != {team_a, team_b}:
            continue
        if r.home_score > r.away_score:
            winner = r.home_team
        elif r.home_score < r.away_score:
            winner = r.away_team
        else:
            a_points += 1
            b_points += 1
            continue
        if winner == team_a:
            a_points += 3
        else:
            b_points += 3
    return a_points, b_points


def _elo_rank_key(team: str, sim_feature_state):
    return -sim_feature_state.elo_system.ratings.get(
        team, sim_feature_state.elo_system.initial_rating
    )


def rank_group(group_letter: str, match_results: list[GroupMatchResult], sim_feature_state
               ) -> list[TeamStanding]:
    standings = _build_standings(group_letter, match_results)
    teams = list(standings.values())

    def primary_key(t: TeamStanding):
        return (-t.points, -t.goal_diff, -t.goals_for)

    teams.sort(key=primary_key)

    # Head-to-head only resolves a tied PAIR; a 3+ way tie skips straight to Elo for the whole block (silent-tiebreaker design decision).
    i = 0
    while i < len(teams):
        j = i
        while j + 1 < len(teams) and primary_key(teams[j + 1]) == primary_key(teams[i]):
            j += 1
        if j > i:
            tied = teams[i:j + 1]
            if len(tied) == 2:
                team_a, team_b = tied[0].team, tied[1].team
                a_pts, b_pts = _head_to_head_points(team_a, team_b, match_results)
                if a_pts != b_pts:
                    h2h_score = {team_a: a_pts, team_b: b_pts}
                    tied.sort(key=lambda t: h2h_score[t.team], reverse=True)
                else:
                    tied.sort(key=lambda t: _elo_rank_key(t.team, sim_feature_state))
            else:
                tied.sort(key=lambda t: _elo_rank_key(t.team, sim_feature_state))
            teams[i:j + 1] = tied
        i = j + 1

    return teams


def rank_third_place_teams(all_group_standings: dict[str, list[TeamStanding]],
                            sim_feature_state) -> list[TeamStanding]:
    third_place_teams = [standings[2] for standings in all_group_standings.values()]

    def key(t: TeamStanding):
        return (-t.points, -t.goal_diff, -t.goals_for, _elo_rank_key(t.team, sim_feature_state))

    third_place_teams.sort(key=key)
    return third_place_teams


def build_round_of_32_matchups(all_group_standings: dict[str, list[TeamStanding]],
                                qualifying_third_place: list[TeamStanding],
                                annex_c_lookup) -> dict[str, tuple[str, str]]:
    group_winner = {g: standings[0].team for g, standings in all_group_standings.items()}
    group_runner_up = {g: standings[1].team for g, standings in all_group_standings.items()}

    def resolve_slot(slot: str) -> str:
        rank, group = slot[0], slot[1]
        return group_winner[group] if rank == "1" else group_runner_up[group]

    matchups = {}
    for match_id, (slot_a, slot_b) in ts.FIXED_R32_PAIRINGS.items():
        matchups[match_id] = (resolve_slot(slot_a), resolve_slot(slot_b))

    qualifying_groups = frozenset(t.group for t in qualifying_third_place)
    third_place_assignments = ts.get_round_of_32_conditional_assignments(
        qualifying_groups, annex_c_lookup
    )
    third_place_team_by_group = {t.group: t.team for t in qualifying_third_place}

    for match_id, info in ts.CONDITIONAL_R32_SLOTS.items():
        group_winner_team = resolve_slot(info["group_winner"])
        third_place_group = third_place_assignments[match_id]
        third_place_team = third_place_team_by_group[third_place_group]
        matchups[match_id] = (group_winner_team, third_place_team)

    return matchups