"""
World Cup Match Predictor - Prediction Logger
-------------------------------------------------
This is the script that builds your real evidence over the tournament.

Two modes:
  1. LOG a prediction before a match happens (predicted probabilities)
  2. UPDATE the log with the actual result once the match finishes

Usage:
    Step 1 (before match): python log_prediction.py log
    Step 2 (after match):  python log_prediction.py update
"""

import pandas as pd
import pickle
import sys
import os
from datetime import datetime

from feature_engineering import load_and_clean, build_feature_row

DATA_PATH = "data/results.csv"
MODEL_PATH = "model.pkl"
SCALER_PATH = "scaler.pkl"
LOG_PATH = "predictions_log.csv"

FEATURE_COLUMNS = [
    'home_win_rate', 'away_win_rate',
    'home_avg_goals_scored', 'home_avg_goals_conceded',
    'away_avg_goals_scored', 'away_avg_goals_conceded',
    'form_diff', 'attack_vs_defense_home', 'attack_vs_defense_away',
    'h2h_home_win_rate', 'neutral_venue'
]

# EDIT THIS before each day's matches, then run: python log_prediction.py log
TODAYS_FIXTURES = [
    {"home_team": "Turkey", "away_team": "Paraguay", "neutral_venue": True},
    {"home_team": "Netherlands", "away_team": "Sweden", "neutral_venue": True},
    {"home_team": "Germany", "away_team": "Ivory Coast", "neutral_venue": True},
    {"home_team": "Ecuador", "away_team": "Curaçao", "neutral_venue": True},
]


def predict_match(model, scaler, df, home_team, away_team, neutral_venue=True):
    today = pd.Timestamp.now()
    feats = build_feature_row(df, home_team, away_team, today, neutral_venue=neutral_venue)
    X = pd.DataFrame([feats])[FEATURE_COLUMNS]
    X_scaled = scaler.transform(X)
    probs = model.predict_proba(X_scaled)[0]
    classes = model.classes_
    prob_dict = dict(zip(classes, probs))
    return {
        'home_win_prob': prob_dict.get(1, 0),
        'draw_prob': prob_dict.get(0, 0),
        'away_win_prob': prob_dict.get(-1, 0),
    }


def load_log():
    if os.path.exists(LOG_PATH):
        log = pd.read_csv(LOG_PATH)
        # Force these to object dtype so text like "Draw" can be written
        # into them later (empty/NaN columns default to float64 otherwise)
        log['actual_outcome'] = log['actual_outcome'].astype(object)
        log['correct'] = log['correct'].astype(object)
        return log
    return pd.DataFrame(columns=[
        'date_logged', 'home_team', 'away_team',
        'home_win_prob', 'draw_prob', 'away_win_prob',
        'predicted_outcome', 'actual_outcome', 'correct'
    ])


def log_todays_predictions():
    """Step 1: Run this BEFORE matches happen, to log what the model predicted."""
    print("Loading model and data...")
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    df = load_and_clean(DATA_PATH)

    log = load_log()
    new_rows = []

    for match in TODAYS_FIXTURES:
        home, away = match['home_team'], match['away_team']

        already_logged = ((log['home_team'] == home) & (log['away_team'] == away) &
                           (log['date_logged'] == datetime.now().strftime('%Y-%m-%d'))).any()
        if already_logged:
            print(f"Already logged: {home} vs {away} - skipping")
            continue

        probs = predict_match(model, scaler, df, home, away, match.get('neutral_venue', True))

        outcomes = {'Home Win': probs['home_win_prob'], 'Draw': probs['draw_prob'], 'Away Win': probs['away_win_prob']}
        predicted_outcome = max(outcomes, key=outcomes.get)

        new_rows.append({
            'date_logged': datetime.now().strftime('%Y-%m-%d'),
            'home_team': home,
            'away_team': away,
            'home_win_prob': round(probs['home_win_prob'], 3),
            'draw_prob': round(probs['draw_prob'], 3),
            'away_win_prob': round(probs['away_win_prob'], 3),
            'predicted_outcome': predicted_outcome,
            'actual_outcome': None,
            'correct': None
        })

        print(f"Logged: {home} vs {away} -> predicted {predicted_outcome} "
              f"(H:{probs['home_win_prob']*100:.0f}% D:{probs['draw_prob']*100:.0f}% A:{probs['away_win_prob']*100:.0f}%)")

    if new_rows:
        log = pd.concat([log, pd.DataFrame(new_rows)], ignore_index=True)
        log.to_csv(LOG_PATH, index=False)
        print(f"\nSaved {len(new_rows)} new predictions to {LOG_PATH}")
    else:
        print("\nNo new predictions to log.")


def update_with_results():
    """Step 2: Run this AFTER matches finish, to fill in actual results and score accuracy."""
    log = load_log()
    if len(log) == 0:
        print("No predictions logged yet. Run 'python log_prediction.py log' first.")
        return

    pending = log[log['actual_outcome'].isna()]
    if len(pending) == 0:
        print("No pending predictions to update.")
    else:
        print(f"\n{len(pending)} predictions awaiting results:\n")
        for idx, row in pending.iterrows():
            print(f"  [{idx}] {row['home_team']} vs {row['away_team']} (predicted: {row['predicted_outcome']})")

        print("\nFor each match above, enter the actual result.")
        print("Type: H (home win), D (draw), A (away win), or S (skip/not finished yet)\n")

        for idx, row in pending.iterrows():
            answer = input(f"{row['home_team']} vs {row['away_team']} - actual result (H/D/A/S): ").strip().upper()
            if answer == '' or answer == 'S':
                continue
            outcome_map = {'H': 'Home Win', 'D': 'Draw', 'A': 'Away Win'}
            if answer not in outcome_map:
                print(f"  Invalid input '{answer}', skipping this match (you can update it next time).")
                continue
            actual = outcome_map[answer]
            log.at[idx, 'actual_outcome'] = actual
            log.at[idx, 'correct'] = bool(actual == row['predicted_outcome'])

        log.to_csv(LOG_PATH, index=False)
        print(f"\nUpdated {LOG_PATH}")

    completed = log.dropna(subset=['actual_outcome'])
    if len(completed) > 0:
        accuracy = completed['correct'].mean()
        print(f"\n{'='*60}")
        print(f"RUNNING TOTAL: {len(completed)} matches scored so far")
        print(f"Accuracy: {accuracy*100:.1f}% ({completed['correct'].sum()}/{len(completed)} correct)")
        print(f"{'='*60}")


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ['log', 'update']:
        print("Usage:")
        print("  python log_prediction.py log     <- run BEFORE matches (logs predictions)")
        print("  python log_prediction.py update  <- run AFTER matches (records results, shows accuracy)")
    elif sys.argv[1] == 'log':
        log_todays_predictions()
    else:
        update_with_results()