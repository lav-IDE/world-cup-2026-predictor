from __future__ import annotations
from collections import defaultdict

from backend.models.monte_carlo_sim import tournament_structure as ts
from backend.models.monte_carlo_sim import groupstage_sim as gss
from backend.models.monte_carlo_sim import knockout_sim as ks

STAGES = ["group_winner", "reach_r32", "reach_r16", "reach_qf", "reach_sf", "reach_final", "champion"]


def _run_single_iteration(canonical_state, predictor, annex_c_lookup) -> dict[str, set[str]]:
    state = canonical_state.clone_for_iteration()

    reached = {stage: set() for stage in STAGES}

    all_group_standings = {}
    for group_letter in ts.GROUPS:
        match_results = gss.simulate_group(group_letter, state, predictor)
        ranked = gss.rank_group(group_letter, match_results, state)
        all_group_standings[group_letter] = ranked
        reached["group_winner"].add(ranked[0].team)

    third_place = gss.rank_third_place_teams(all_group_standings, state)
    qualifying_third_place = third_place[:8]

    for standings in all_group_standings.values():
        reached["reach_r32"].add(standings[0].team)
        reached["reach_r32"].add(standings[1].team)
    for t in qualifying_third_place:
        reached["reach_r32"].add(t.team)

    r32_matchups = gss.build_round_of_32_matchups(
        all_group_standings, qualifying_third_place, annex_c_lookup
    )
    bracket_results = ks.simulate_bracket(r32_matchups, state, predictor)

    for match_id in ts.R16_FROM_R32:
        feeder_a, feeder_b = ts.R16_FROM_R32[match_id]
        reached["reach_r16"].add(bracket_results[feeder_a].winner)
        reached["reach_r16"].add(bracket_results[feeder_b].winner)

    for match_id in ts.QF_FROM_R16:
        feeder_a, feeder_b = ts.QF_FROM_R16[match_id]
        reached["reach_qf"].add(bracket_results[feeder_a].winner)
        reached["reach_qf"].add(bracket_results[feeder_b].winner)

    for match_id in ts.SF_FROM_QF:
        feeder_a, feeder_b = ts.SF_FROM_QF[match_id]
        reached["reach_sf"].add(bracket_results[feeder_a].winner)
        reached["reach_sf"].add(bracket_results[feeder_b].winner)

    final_match_id = next(iter(ts.FINAL_FROM_SF))
    final_result = bracket_results[final_match_id]
    reached["reach_final"].add(final_result.home_team)
    reached["reach_final"].add(final_result.away_team)
    reached["champion"].add(final_result.winner)

    return reached


def run_simulation(canonical_state, predictor, annex_c_lookup, n_iterations: int
                    ) -> dict[str, dict[str, float]]:
    import time
    all_teams = [team for group in ts.GROUPS.values() for team in group]
    counts = {stage: defaultdict(int) for stage in STAGES}

    progress_interval = max(1, n_iterations // 10)

    for i in range(n_iterations):
        start = time.perf_counter()
        reached = _run_single_iteration(canonical_state, predictor, annex_c_lookup)
        elapsed = time.perf_counter() - start
 
        if i == 0:
            estimated_total_seconds = elapsed * n_iterations
            print(f"First iteration took {elapsed:.3f}s -- "
                  f"estimated total for {n_iterations} iterations: "
                  f"{estimated_total_seconds / 60:.1f} minutes")
 
        for stage in STAGES:
            for team in reached[stage]:
                counts[stage][team] += 1
 
        if (i + 1) % progress_interval == 0:
            print(f"Completed {i + 1}/{n_iterations} iterations")

    results = {}
    for team in all_teams:
        results[team] = {
            stage: counts[stage][team] / n_iterations for stage in STAGES
        }
    return results