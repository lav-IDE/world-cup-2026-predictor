import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  


def load_results():
    
    df = pd.read_csv(ROOT / "data/raw/results.csv")
    df = df.drop(columns=['city', 'country'])
    df["date"] = pd.to_datetime(df["date"])
    return df

def filter_years(df):

    df["date"] = pd.to_datetime(df["date"])
    return df[(df["date"].dt.year >= 1990) & (df["date"] < "2026-04-01")]


def add_winner(df):
    
    df["winner"] = "Draw"

    df.loc[
        df["home_score"]>df["away_score"],
        "winner"
    ] = df["home_team"]
    
    df.loc[
        df["home_score"]<df["away_score"],
        "winner"
    ] = df["away_team"]
    return df

def load_shootouts():
    df = pd.read_csv(ROOT / "data/raw/shootouts.csv")
    df = df.drop(columns='first_shooter')
    df["date"] = pd.to_datetime(df["date"])
    return df
    
    
def merge_shootouts(results_df, shootouts_df):
    merged = pd.merge(
        results_df,
        shootouts_df.rename(
            columns={"winner": "shootout_winner"}
        ),
        on=[
            "date",
            "home_team",
            "away_team"
        ],
        how="left"
    )
    return merged

def load_odds():
    df = pd.read_csv(ROOT / "data/processed/odds_clean.csv")
    return df




import pandas as pd

def merge_odds(results_df, odds_df):
    
    results_df['date'] = pd.to_datetime(results_df['date'])
    results_df = results_df.sort_values('date')
    
    odds_df['date'] = pd.to_datetime(odds_df['date'])
    odds_df = odds_df.sort_values('date')
    
    odds_subset = odds_df[['date', 'home_team', 'away_team', 'home_odd', 'draw_odd', 'away_odd']]
    
    merged_df = pd.merge_asof(
        results_df,
        odds_subset,
        on='date',
        by=['home_team', 'away_team'],
        tolerance=pd.Timedelta(days=1),
        direction='nearest'
    )
    
    return merged_df



