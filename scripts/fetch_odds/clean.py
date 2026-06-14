import argparse
import pandas as pd
import numpy as np
from pathlib import Path

# Team name normalisations (OddsPortal name -> canonical name)
NAME_MAP = {
    # USA variants
    "USA":                          "United States",
    "United States of America":     "United States",
    "US":                           "United States",
    # Korea
    "Korea Republic":               "South Korea",
    "Republic of Korea":            "South Korea",
    "Korea DPR":                    "North Korea",
    "DPR Korea":                    "North Korea",
    # Ivory Coast
    "Cote d'Ivoire":                "Ivory Coast",
    "Côte d'Ivoire":                "Ivory Coast",
    "Cote dIvoire":                 "Ivory Coast",
    # DR Congo
    "Congo DR":                     "DR Congo",
    "Democratic Republic of Congo": "DR Congo",
    "Congo, DR":                    "DR Congo",
    "D.R. Congo":                   "DR Congo",
    # Czech Republic
    "Czechia":                      "Czech Republic",
    # Bosnia
    "Bosnia":                       "Bosnia and Herzegovina",
    "Bosnia & Herzegovina":         "Bosnia and Herzegovina",
    # Cape Verde
    "Cape Verde Islands":           "Cape Verde",
    "Cape Verde Is.":               "Cape Verde",
    # Curacao
    "Curacao":                      "Curaçao",
    # Iran
    "IR Iran":                      "Iran",
    # Trinidad
    "Trinidad & Tobago":            "Trinidad and Tobago",
    # New Zealand 
    "New Zealand":                  "New Zealand",
    #China
    "China":                        "China PR",
    "Ireland":                      "Republic of Ireland"
}

TARGET_TEAMS = {
    "Algeria", "Argentina", "Australia", "Austria", "Belgium",
    "Bosnia and Herzegovina", "Brazil", "Canada", "Cape Verde",
    "Colombia", "Croatia", "Curaçao", "Czech Republic", "DR Congo",
    "Ecuador", "Egypt", "England", "France", "Germany", "Ghana",
    "Haiti", "Iran", "Iraq", "Ivory Coast", "Japan", "Jordan",
    "Mexico", "Morocco", "Netherlands", "New Zealand", "Norway",
    "Panama", "Paraguay", "Portugal", "Qatar", "Saudi Arabia",
    "Scotland", "Senegal", "South Africa", "South Korea", "Spain",
    "Sweden", "Switzerland", "Tunisia", "Turkey", "United States",
    "Uruguay", "Uzbekistan",
}

# Canonical tournament names - any scraped name not in this set gets flagged
KNOWN_TOURNAMENTS = {
    "FIFA World Cup",
    "FIFA World Cup qualification",
    "UEFA Euro",
    "UEFA Euro qualification",
    "UEFA Nations League",
    "Copa America",
    "Copa America qualification",
    "African Cup of Nations",
    "African Cup of Nations qualification",
    "AFC Asian Cup",
    "AFC Asian Cup qualification",
    "Gold Cup",
    "Gold Cup qualification",
    "CONCACAF Nations League",
    "CONCACAF Nations League qualification",
    "Confederations Cup",
    "International Friendly",
}


def normalise_team(name: str) -> str:
    if not isinstance(name, str):
        return name
    name = name.strip()
    return NAME_MAP.get(name, name)


def clean(
    input_path: str = "../../data/raw/odds_raw.csv",
    output_path: str = "../../data/processed/odds_clean.csv",
    filter_teams: bool = False,
    stats: bool = False,
):
    path = Path(input_path)
    if not path.exists():
        print(f"File not found: {input_path}")
        return

    print(f"Loading {input_path}...")
    df = pd.read_csv(input_path)
    print(f"  Raw rows: {len(df):,}")

    # --- Normalise column names ---------------------------------------------
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Ensure all expected columns exist
    for col in ["date", "tournament", "home_team", "away_team", "neutral",
                "home_odd", "draw_odd", "away_odd"]:
        if col not in df.columns:
            df[col] = np.nan

    # --- Team names ---------------------------------------------------------
    df["home_team"] = df["home_team"].apply(normalise_team)
    df["away_team"] = df["away_team"].apply(normalise_team)

    # --- Dates --------------------------------------------------------------
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    before = len(df)
    df = df.dropna(subset=["date"])
    if before - len(df):
        print(f"  Dropped {before - len(df):,} rows with unparseable dates")

    # --- Odds validation ----------------------------------------------------
    for col in ["home_odd", "draw_odd", "away_odd"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["home_odd", "draw_odd", "away_odd"]:
        mask = df[col].notna() & ((df[col] < 1.01) | (df[col] > 200))
        if mask.sum():
            print(f"  Nullifying {mask.sum()} implausible {col} values")
            df.loc[mask, col] = np.nan

    # --- Neutral venue ------------------------------------------------------
    df["neutral"] = df["neutral"].map(
        lambda x: True if str(x).lower() in ("true", "1", "yes") else False
    )

    # --- Unknown tournaments ------------------------------------------------
    unknown = set(df["tournament"].unique()) - KNOWN_TOURNAMENTS
    if unknown:
        print(f"  Unknown tournament labels (check manually): {unknown}")

    # --- Deduplication ------------------------------------------------------
    before = len(df)
    df = df.drop_duplicates(subset=["date", "home_team", "away_team", "tournament"])
    if before - len(df):
        print(f"  Removed {before - len(df):,} duplicate rows")

    # --- Filter to target teams --------------------------------------------
    if filter_teams:
        mask = df["home_team"].isin(TARGET_TEAMS) | df["away_team"].isin(TARGET_TEAMS)
        before = len(df)
        df = df[mask]
        print(f"  Filtered to target teams: {before - len(df):,} removed, {len(df):,} kept")

    # --- Sort ---------------------------------------------------------------
    df = df.sort_values(["tournament", "date"]).reset_index(drop=True)
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")

    # --- Save ---------------------------------------------------------------
    df.to_csv(output_path, index=False)
    print(f"\nSaved {len(df):,} rows -> {output_path}")

    # --- Stats --------------------------------------------------------------
    if stats:
        print(f"\n{'='*50}")
        print("Dataset statistics")
        print(f"{'='*50}")
        print(f"Total matches:       {len(df):,}")
        print(f"Date range:          {df['date'].min()} -> {df['date'].max()}")
        print(f"Matches with odds:   {df['home_odd'].notna().sum():,}")
        print(f"Neutral venue:       {df['neutral'].sum():,}")
        print(f"\nMatches per tournament:")
        for t, n in df["tournament"].value_counts().items():
            print(f"  {t:<50} {n:>5}")
        print(f"\nTop 20 teams by appearances:")
        teams = pd.concat([df["home_team"], df["away_team"]]).value_counts().head(20)
        for team, n in teams.items():
            flag = " <- target" if team in TARGET_TEAMS else ""
            print(f"  {team:<35} {n:>5}{flag}")

    return df


if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default=str(ROOT / "data" / "raw" / "odds_raw.csv")
    )

    parser.add_argument(
        "--output",
        default=str(ROOT / "data" / "processed" / "odds_clean.csv")
    )
    parser.add_argument("--filter-teams", action="store_true")
    parser.add_argument("--stats",        action="store_true")
    args = parser.parse_args()
    clean(args.input, args.output, args.filter_teams, args.stats)