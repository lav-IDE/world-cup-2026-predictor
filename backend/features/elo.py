import pandas as pd
import numpy as np

class ELOModel:
    def __init__(
        self,
        initial_rating=1500,
        k_factor=20,
        home_advantage=50
    ):
        self.initial_rating = initial_rating
        self.k_factor = k_factor
        self.home_advantage = home_advantage
        self.ratings = {}
        self.last_played = {}
        self.history = []


    def get_rating(self, team, date):
        return self.ratings.get(team, self.initial_rating)
    
    
    def expected_score(self, rating_a, rating_b):
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
    
    
    def recency_weight(self, match_date, reference_date):
        years_ago = (reference_date - match_date).days / 365.25

        half_life_years = 25

        return 0.5 ** (years_ago / half_life_years)
     
       
    def match_weight(self, tournament):
        tournament = tournament.lower()
        
        if "friendly" in tournament:
            return 1.0
        elif "fifa world cup qualification" in tournament:
            return 1.2
        elif "fifa world cup" in tournament:
            return 3.0
        elif "qualif" in tournament:
            return 1.5
        elif any(t in tournament for t in ["uefa euro", "copa américa", "uefa nations league"]):
            return 2.5
        elif any(t in tournament for t in ["african cup of nations","concacaf nations league","gold cup"]):
            return 2.0
        elif any(t in tournament for t in ["waff", "gulf cup", "arab cup", "uncaf", "cafa", "asean"]):
            return 1.0
        elif any(t in tournament for t in ["asian games", "nehru cup", "king's cup", "merdeka", "millennium",
                                           "prime minister","cecafa", "amilcar cabral", "island games", "south pacific",
                                            "melanesia", "aff championship", "dynasty cup", "kirin cup",
                                            "baltic cup", "cfu caribbean", "south asian games",
                                            "southeast asian games", "usa cup", "korea cup"]):
            return 0.2  
        else:
            return 0.2
        
    def get_expected(self, home_team, away_team, neutral):
        advantage = 0 if neutral else self.home_advantage
        home_rating = self.ratings.get(home_team, self.initial_rating) + advantage
        away_rating = self.ratings.get(away_team, self.initial_rating)
        
        home_exp = self.expected_score(home_rating, away_rating)
        away_exp = self.expected_score(away_rating, home_rating)
        return home_exp, away_exp
    
    
    def actual_score(self, home_team, away_team, home_score, away_score, shootout_winner):
        if home_score > away_score:
            return 1.0, 0.0
        elif home_score < away_score:
            return 0.0, 1.0
        elif pd.notna(shootout_winner):
            if shootout_winner == home_team:
                return 0.6, 0.4
            else:
                return 0.4, 0.6
        else:
            return 0.5, 0.5    
    
    
    def update_ratings(self, row):
        home_team = row['home_team']
        away_team = row['away_team']
        date = row['date']

        home_exp, away_exp = self.get_expected(
            home_team, away_team,
            row['neutral']
        )

        
        
        home_act, away_act = self.actual_score(
            home_team, away_team,
            row['home_score'], row['away_score'],
            row['shootout_winner']
        )

        tournament_weight = self.match_weight(row["tournament"])
        reference_date = pd.Timestamp("2026-06-11")
        recency_weight = self.recency_weight(row["date"], reference_date)

        delta = (
            self.k_factor
            * tournament_weight
            * recency_weight
            * (home_act - home_exp)
        )

        self.history.append({
            "date": date,
            "home_team": home_team,
            "away_team": away_team,
            "home_elo": self.ratings.get(home_team, self.initial_rating),
            "away_elo": self.ratings.get(away_team, self.initial_rating),
        })

        self.ratings[home_team] = self.ratings.get(home_team, self.initial_rating) + delta
        self.ratings[away_team] = self.ratings.get(away_team, self.initial_rating) - delta

        self.last_played[home_team] = date
        self.last_played[away_team] = date
    
    
    def fit(self, df):
        df = df.sort_values('date').reset_index(drop=True)
        
        for _, row in df.iterrows():
            self.update_ratings(row)
        
        return self
    
    
    
    def compute_ratings(self):
        return (
        pd.DataFrame.from_dict(self.ratings, orient='index', columns=['rating'])
        .rename_axis('team')
        .reset_index()
        .sort_values('rating', ascending=False)
        .reset_index(drop=True)
        )
        
    def get_history(self):
        return pd.DataFrame(self.history)

    

