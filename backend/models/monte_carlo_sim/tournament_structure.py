from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, FrozenSet, Optional
import csv


# ---------------------------------------------------------------------------
# Groups (confirmed post-playoff rosters)
# ---------------------------------------------------------------------------

GROUPS: Dict[str, List[str]] = {
    "A": ["Mexico", "South Africa", "South Korea", "Czechia"],
    "B": ["Canada", "Bosnia and Herzegovina", "Qatar", "Switzerland"],
    "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "D": ["United States", "Paraguay", "Australia", "Türkiye"],
    "E": ["Germany", "Curaçao", "Ivory Coast", "Ecuador"],
    "F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
    "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "H": ["Spain", "Cape Verde", "Saudi Arabia", "Uruguay"],
    "I": ["France", "Senegal", "Iraq", "Norway"],
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "K": ["Portugal", "DR Congo", "Uzbekistan", "Colombia"],
    "L": ["England", "Croatia", "Ghana", "Panama"],
}

ALL_GROUP_LETTERS: List[str] = sorted(GROUPS.keys())  # A..L


# ---------------------------------------------------------------------------
# Round of 32 — fixed pairings (Article 12.6)
# ---------------------------------------------------------------------------

FIXED_R32_PAIRINGS: Dict[str, Tuple[str, str]] = {
    "M73": ("2A", "2B"),
    "M75": ("1F", "2C"),
    "M76": ("1C", "2F"),
    "M78": ("2E", "2I"),
    "M83": ("2K", "2L"),
    "M84": ("1H", "2J"),
    "M86": ("1J", "2H"),
    "M88": ("2D", "2G"),
}

# ---------------------------------------------------------------------------
# Round of 32 — conditional pairings (Article 12.6 + Annex C)
# ---------------------------------------------------------------------------

CONDITIONAL_R32_SLOTS: Dict[str, dict] = {
    "M74": {"group_winner": "1E", "eligible_third_place_groups": frozenset("ABCDF")},
    "M77": {"group_winner": "1I", "eligible_third_place_groups": frozenset("CDFGH")},
    "M79": {"group_winner": "1A", "eligible_third_place_groups": frozenset("CEFHI")},
    "M80": {"group_winner": "1L", "eligible_third_place_groups": frozenset("EHIJK")},
    "M81": {"group_winner": "1D", "eligible_third_place_groups": frozenset("BEFIJ")},
    "M82": {"group_winner": "1G", "eligible_third_place_groups": frozenset("AEHIJ")},
    "M85": {"group_winner": "1B", "eligible_third_place_groups": frozenset("EFGIJ")},
    "M87": {"group_winner": "1K", "eligible_third_place_groups": frozenset("DEIJL")},
}

ALL_R32_MATCH_IDS: List[str] = [
    "M73", "M74", "M75", "M76", "M77", "M78", "M79", "M80",
    "M81", "M82", "M83", "M84", "M85", "M86", "M87", "M88",
]


# -------------------------------------------------------------
# # Annex C loader
# -------------------------------------------------------------

# that conditional slot corresponds to (per CONDITIONAL_R32_SLOTS above).
_GROUP_WINNER_HEADER_TO_MATCH_ID: Dict[str, str] = {
    "1A": "M79",
    "1B": "M85",
    "1D": "M81",
    "1E": "M74",
    "1G": "M82",
    "1I": "M77",
    "1K": "M87",
    "1L": "M80",
}


@dataclass(frozen=True)
class AnnexCRow:
    option: int
    # match_id -> third-place group letter assigned to that slot
    assignments: Dict[str, str]
    # the 8 group letters whose third-place teams qualify in this row
    qualifying_third_place_groups: FrozenSet[str]


def load_annex_c(csv_path: str) -> Dict[FrozenSet[str], AnnexCRow]:

    lookup: Dict[FrozenSet[str], AnnexCRow] = {}

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for raw_row in reader:
            option = int(raw_row["option"])

            assignments: Dict[str, str] = {}
            qualifying_groups = set()

            for header, match_id in _GROUP_WINNER_HEADER_TO_MATCH_ID.items():
                cell = raw_row[header].strip()
                third_place_group_letter = cell[1:] 
                assignments[match_id] = third_place_group_letter
                qualifying_groups.add(third_place_group_letter)

            if len(qualifying_groups) != 8:
                raise ValueError(
                    f"Annex C row option={option} does not have 8 distinct "
                    f"third-place groups in its values: {raw_row}. "
                    "Lookup-key assumption is invalid for this row — "
                    "stop and re-examine the CSV structure."
                )

            key = frozenset(qualifying_groups)
            row_obj = AnnexCRow(
                option=option,
                assignments=assignments,
                qualifying_third_place_groups=key,
            )

            if key in lookup:
                raise ValueError(
                    f"Duplicate qualifying-set key {sorted(key)} found at "
                    f"option={option} (already seen at "
                    f"option={lookup[key].option}). The qualifying-set-as-key "
                    "assumption is invalid — Annex C needs a different lookup "
                    "key (likely the literal 'option' row order combined with "
                    "some external rule for selecting which option applies)."
                )

            lookup[key] = row_obj

    return lookup


def get_round_of_32_conditional_assignments(
    qualifying_third_place_groups: FrozenSet[str],
    annex_c_lookup: Dict[FrozenSet[str], AnnexCRow],
) -> Dict[str, str]:

    if len(qualifying_third_place_groups) != 8:
        raise ValueError(
            f"Expected exactly 8 qualifying third-place groups, got "
            f"{len(qualifying_third_place_groups)}: {qualifying_third_place_groups}"
        )
    row = annex_c_lookup[qualifying_third_place_groups]
    return row.assignments


# ---------------------------------------------------------------------------
# Bracket progression: R32 -> R16 -> QF -> SF -> Final (Article 12.7-12.11)
# ---------------------------------------------------------------------------

R16_FROM_R32: Dict[str, Tuple[str, str]] = {
    "M89": ("M73", "M74"),
    "M90": ("M75", "M76"),
    "M91": ("M77", "M78"),
    "M92": ("M79", "M80"),
    "M93": ("M81", "M82"),
    "M94": ("M83", "M84"),
    "M95": ("M85", "M86"),
    "M96": ("M87", "M88"),
}

QF_FROM_R16: Dict[str, Tuple[str, str]] = {
    "M97": ("M89", "M90"),
    "M98": ("M91", "M92"),
    "M99": ("M93", "M94"),
    "M100": ("M95", "M96"),
}

SF_FROM_QF: Dict[str, Tuple[str, str]] = {
    "M101": ("M97", "M98"),
    "M102": ("M99", "M100"),
}

# ---------------------------------------------------------------------------
# Knockout match venues and host-country lookup
# ---------------------------------------------------------------------------

KNOCKOUT_SCHEDULE: Dict[int, dict] = {
    73: {"date": "2026-06-28", "venue": "Los Angeles Stadium", "location": "Inglewood, CA"},
    74: {"date": "2026-06-29", "venue": "Boston Stadium", "location": "Foxborough, MA"},
    75: {"date": "2026-06-29", "venue": "Monterrey Stadium", "location": "Guadalupe, MX"},
    76: {"date": "2026-06-29", "venue": "Houston Stadium", "location": "Houston, TX"},
    77: {"date": "2026-06-30", "venue": "New York New Jersey Stadium", "location": "East Rutherford, NJ"},
    78: {"date": "2026-06-30", "venue": "Dallas Stadium", "location": "Arlington, TX"},
    79: {"date": "2026-06-30", "venue": "Mexico City Stadium", "location": "Estadio Azteca, MX"},
    80: {"date": "2026-07-01", "venue": "San Francisco Bay Area Stadium", "location": "Santa Clara, CA"},
    81: {"date": "2026-07-01", "venue": "Seattle Stadium", "location": "Seattle, WA"},
    82: {"date": "2026-07-01", "venue": "Edmonton Stadium", "location": "Edmonton, CAN"},
    83: {"date": "2026-07-02", "venue": "Toronto Stadium", "location": "BMO Field, CAN"},
    84: {"date": "2026-07-02", "venue": "Los Angeles Stadium", "location": "Inglewood, CA"},
    85: {"date": "2026-07-02", "venue": "Vancouver Stadium", "location": "BC Place, CAN"},
    86: {"date": "2026-07-03", "venue": "Miami Stadium", "location": "Miami Gardens, FL"},
    87: {"date": "2026-07-03", "venue": "Kansas City Stadium", "location": "Kansas City, MO"},
    88: {"date": "2026-07-03", "venue": "Atlanta Stadium", "location": "Atlanta, GA"},
    89: {"date": "2026-07-04", "venue": "Houston Stadium", "location": "Houston, TX"},
    90: {"date": "2026-07-04", "venue": "Philadelphia Stadium", "location": "Philadelphia, PA"},
    91: {"date": "2026-07-05", "venue": "New York New Jersey Stadium", "location": "East Rutherford, NJ"},
    92: {"date": "2026-07-05", "venue": "Mexico City Stadium", "location": "Estadio Azteca, MX"},
    93: {"date": "2026-07-06", "venue": "Dallas Stadium", "location": "Arlington, TX"},
    94: {"date": "2026-07-06", "venue": "Seattle Stadium", "location": "Seattle, WA"},
    95: {"date": "2026-07-07", "venue": "Atlanta Stadium", "location": "Atlanta, GA"},
    96: {"date": "2026-07-07", "venue": "Vancouver Stadium", "location": "BC Place, CAN"},
    97: {"date": "2026-07-09", "venue": "Boston Stadium", "location": "Foxborough, MA"},
    98: {"date": "2026-07-10", "venue": "Los Angeles Stadium", "location": "Inglewood, CA"},
    99: {"date": "2026-07-11", "venue": "Miami Stadium", "location": "Miami Gardens, FL"},
    100: {"date": "2026-07-11", "venue": "Kansas City Stadium", "location": "Kansas City, MO"},
    101: {"date": "2026-07-14", "venue": "Dallas Stadium", "location": "Arlington, TX"},
    102: {"date": "2026-07-15", "venue": "Atlanta Stadium", "location": "Atlanta, GA"},
    103: {"date": "2026-07-18", "venue": "Miami Stadium", "location": "Hard Rock Stadium, FL"},
    104: {"date": "2026-07-19", "venue": "New York New Jersey Stadium", "location": "MetLife Stadium, NJ"},
}

# US state/territory postal codes that can appear in `location`'s suffix.
_US_STATE_CODES = frozenset({
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID",
    "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS",
    "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK",
    "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV",
    "WI", "WY", "DC",
})


def get_host_country(match_id: int) -> str:

    location = KNOCKOUT_SCHEDULE[match_id]["location"]
    suffix = location.split(",")[-1].strip()

    if suffix == "MX":
        return "Mexico"
    if suffix == "CAN":
        return "Canada"
    if suffix in _US_STATE_CODES:
        return "USA"
    raise ValueError(
        f"Could not classify host country for match {match_id}: "
        f"unrecognized location suffix '{suffix}' in '{location}'"
    )


HOST_COUNTRY_TEAM_NAMES: Dict[str, str] = {
    "USA": "United States",
    "Mexico": "Mexico",
    "Canada": "Canada",
}


def get_home_advantage_team(match_id: int, team_a: str, team_b: str) -> Optional[str]:

    host_country = get_host_country(match_id)
    host_team_name = HOST_COUNTRY_TEAM_NAMES[host_country]

    a_is_host = team_a == host_team_name
    b_is_host = team_b == host_team_name

    if a_is_host and b_is_host:
        raise ValueError(
            f"Match {match_id}: both participants ('{team_a}', '{team_b}') "
            f"resolve to the host team name '{host_team_name}' -- data error."
        )
    if a_is_host:
        return team_a
    if b_is_host:
        return team_b
    return None


# ---------------------------------------------------------------------------
# Bracket progression: R32 -> R16 -> QF -> SF -> Final/Third-Place
# ---------------------------------------------------------------------------

# (R16_FROM_R32, QF_FROM_R16, SF_FROM_QF defined above, unchanged)

THIRD_PLACE_MATCH_FROM_SF: Dict[str, Tuple[str, str]] = {
    # M103 is contested by the LOSERS of M101 and M102, not the winners.
    "M103": ("M101", "M102"),
}

FINAL_FROM_SF: Dict[str, Tuple[str, str]] = {
    # M104 is contested by the WINNERS of M101 and M102.
    "M104": ("M101", "M102"),
}