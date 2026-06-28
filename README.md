# FIFA World Cup 2026 Predictor 
Using custom ELO including time decay (giving recent matches more weightage), odds weight based on how the match turned out as compared to Bookmakers' odds before the match since they do a lot of analytics before providing the odds data, I have created a World cup predictor that runs on some data, processes it, runs ML algos and tells who it preferrably sees as the winner.


## Project Structure

```
world_cup_predictor/
│
├── backend/
│   ├── api/                                            # yet to be made
│   │  
│   ├── data/
│   │   └── preprocessing.py
│   │
│   ├── features/
│   │   ├── elo.py
│   │   ├── odds.py
│   │   └── recent_stats.py
│   │
│   ├── models/
│   │   └──artifacts
│   │   │   ├──classifier.json
│   │   │   ├──regressor.json
│   │   │   ├──home_score_regressor.json
│   │   │   └──away_score_regressor.json
│   │   │
│   │   └──monte_carlo_sim
│   │   │   ├──tournament_structure.py
│   │   │   ├──groupstage_sim.py
│   │   │   ├──knockout_sim.py
│   │   │   ├──predictor.py
│   │   │   └──sim_features.py
│   │   │
│   │   ├── data_split.py
│   │   ├── monte_carlo.py
│   │   ├── feature_matrix.py
│   │   ├── train_classifier.py
│   │   └── train_regressor.py
│   │
│   ├── config.py
│   └── app.py
│
├── data/
│   ├── raw/
│   │   └── .gitkeep
│   │
│   └── processed/
│       └── .gitkeep
│
├── scripts/
│   ├── fetch_odds/
│   │   ├── scraper.py
│   │   └── clean.py
│   │
│   ├── build_classifier.py
│   ├── build_dataset.py
│   ├── build_elo_ratings.py
│   ├── build_oddsfeature.py
│   ├── build_recentstats.py
│   ├── build_featurematrix.py
│   ├── build_regressor.py
│   └── build_simulation.py
│
├── frontend/                                                   # yet to be made
│
├── README.md
└── requirements.txt
```

## Datasets

- `results.csv` – historical international football results
- `odds_raw.csv` – scraped bookmaker odds
- `shootouts.csv` – penalty shootout outcomes
- `matches.csv` – final merged training dataset for ELOs
- `ro32.csv` - Round of 32 fixture buildup dataset

### Sources -

- Historic international football results - [Kaggle Dataset](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017)
- Odds data - Scraped from [Odds Portal](https://www.oddsportal.com/) (Site not available in India)
- Round of 32 dataset - Official FIFA document showing the mapping of each RO32 fixture (Annex-C).
