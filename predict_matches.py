"""
World Cup Match Predictor - Predict Upcoming Matches
-------------------------------------------------------
Run this AFTER train_model.py has been run successfully once.

Usage:
    Edit the MATCHES_TO_PREDICT list below with today's fixtures,
    then run: python predict_matches.py
"""

import pandas as pd
import pickle
from datetime import datetime

from feature_engineering import load_and_clean, build_feature_row

DATA_PATH = "data/results.csv"
MODEL_PATH = "model.pkl"
SCALER_PATH = "scaler.pkl"

FEATURE_COLUMNS = [
    'home_win_rate', 'away_win_rate',
    'home_avg_goals_scored', 'home_avg_goals_conceded',
    'away_avg_goals_scored', 'away_avg_goals_conceded',
    'form_diff', 'attack_vs_defense_home', 'attack_vs_defense_away',
    'h2h_home_win_rate', 'neutral_venue'
]

# EDIT THIS LIST with today's actual fixtures.
# Use exact team names as they appear in your results.csv
MATCHES_TO_PREDICT = [
    {"home_team": "Turkey", "away_team": "Paraguay", "neutral_venue": True},
    {"home_team": "Netherlands", "away_team": "Sweden", "neutral_venue": True},
    {"home_team": "Germany", "away_team": "Ivory Coast", "neutral_venue": True},
    {"home_team": "Ecuador", "away_team": "Curacao", "neutral_venue": True},
]


def check_team_exists(df, team_name):
    exists = ((df['home_team'] == team_name).any() or (df['away_team'] == team_name).any())
    return exists


def predict_match(model, scaler, df, home_team, away_team, neutral_venue=True):
    today = pd.Timestamp.now()

    feats = build_feature_row(df, home_team, away_team, today, neutral_venue=neutral_venue)
    X = pd.DataFrame([feats])[FEATURE_COLUMNS]
    X_scaled = scaler.transform(X)

    probs = model.predict_proba(X_scaled)[0]
    classes = model.classes_

    prob_dict = dict(zip(classes, probs))
    home_win_prob = prob_dict.get(1, 0)
    draw_prob = prob_dict.get(0, 0)
    away_win_prob = prob_dict.get(-1, 0)

    return {
        'home_team': home_team,
        'away_team': away_team,
        'home_win_prob': home_win_prob,
        'draw_prob': draw_prob,
        'away_win_prob': away_win_prob,
        'home_matches_played': feats['home_matches_played'],
        'away_matches_played': feats['away_matches_played'],
    }


def main():
    print("Loading model and historical data...")
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)

    df = load_and_clean(DATA_PATH)
    print(f"Loaded {len(df)} historical matches\n")

    print("=" * 70)
    print("PREDICTIONS FOR TODAY'S MATCHES")
    print("=" * 70)

    for match in MATCHES_TO_PREDICT:
        home, away = match['home_team'], match['away_team']

        for team in [home, away]:
            if not check_team_exists(df, team):
                print(f"\n  WARNING: '{team}' not found in historical data. "
                      f"Check spelling (e.g. 'Turkey' vs 'Türkiye', 'Ivory Coast' vs 'Côte d'Ivoire')")

        result = predict_match(model, scaler, df, home, away, match.get('neutral_venue', True))

        print(f"\n{result['home_team']} vs {result['away_team']}")
        print(f"  {result['home_team']} win: {result['home_win_prob']*100:.1f}%")
        print(f"  Draw:                {result['draw_prob']*100:.1f}%")
        print(f"  {result['away_team']} win: {result['away_win_prob']*100:.1f}%")
        print(f"  (based on {result['home_team']}'s last {result['home_matches_played']} matches, "
              f"{result['away_team']}'s last {result['away_matches_played']} matches)")

    print()
    print("=" * 70)
    print("Reminder: this model predicts decisive outcomes (win/loss) more")
    print("reliably than draws - see known limitation discussed earlier.")
    print("=" * 70)


if __name__ == "__main__":
    main()