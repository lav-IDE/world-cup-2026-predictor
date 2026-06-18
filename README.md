# FIFA World Cup 2026 Predictor 
Using custom ELO including time decay (giving recent matches more weightage), odds weight based on how the match turned out as compared to Bookmakers' odds before the match since they do a lot of analytics before providing the odds data, I have created a World cup predictor that runs on some data, processes it, runs ML algos and tells who it preferrably sees as the winner.


## Project Structure

```
world_cup_predictor/
│
├── backend/
│   ├── api/
│   │  
│   ├── data/
│   │   ├── loaders.py
│   │   └── preprocessing.py
│   │
│   ├── features/
│   │   ├── elo.py
│   │   ├── odds.py
│   │   └── recent_stats.py
│   │
│   ├── models/
│   │   ├── predictor.py
│   │   ├── monte_carlo.py
│   │   └── feature_matrix.py
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
│   ├── build_dataset.py
│   ├── build_elo_ratings.py
│   ├── build_oddsfeature.py
│   ├── build_recentstats.py
│   └── train_model.py
│
├── frontend/
│
├── README.md
└── requirements.txt
```

## Datasets

- `results.csv` – historical international football results
- `odds_raw.csv` – scraped bookmaker odds
- `shootouts.csv` – penalty shootout outcomes
- `matches.csv` – final merged training dataset for ELOs

### Sources -

- Historic international football results - [Kaggle Dataset](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017)
- Odds data - Scraped from [Odds Portal](https://www.oddsportal.com/) (Site not available in India)