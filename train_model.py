"""
World Cup Match Predictor - Model Training
--------------------------------------------
Run this after feature_engineering.py is saved in the same folder.

This script:
1. Loads historical match data
2. Builds features (using only pre-match information - no data leakage)
3. Trains 2 models and compares them
4. Saves the best model to disk for later use

NOTE: Step 2 (feature building) will take a few minutes since it scans
historical matches for every single match since 2000. This is expected.
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, log_loss

from feature_engineering import load_and_clean, build_training_set

DATA_PATH = "data/results.csv"
MODEL_OUTPUT_PATH = "model.pkl"
SCALER_OUTPUT_PATH = "scaler.pkl"

FEATURE_COLUMNS = [
    'home_win_rate', 'away_win_rate',
    'home_avg_goals_scored', 'home_avg_goals_conceded',
    'away_avg_goals_scored', 'away_avg_goals_conceded',
    'form_diff', 'attack_vs_defense_home', 'attack_vs_defense_away',
    'h2h_home_win_rate', 'neutral_venue'
]


def main():
    print("=" * 60)
    print("STEP 1: Loading historical match data...")
    print("=" * 60)
    df = load_and_clean(DATA_PATH)
    print(f"Loaded {len(df)} total matches from {df['date'].min().date()} to {df['date'].max().date()}")

    print()
    print("=" * 60)
    print("STEP 2: Building features (this will take a few minutes)...")
    print("=" * 60)
    feature_df = build_training_set(df, min_date='2000-01-01')

    feature_df.to_csv("data/features_built.csv", index=False)
    print(f"\nFeature building complete. {len(feature_df)} rows created.")
    print("Saved to data/features_built.csv (so we don't rebuild next time)")

    print()
    print("=" * 60)
    print("STEP 3: Preparing train/test split...")
    print("=" * 60)

    feature_df = feature_df[
        (feature_df['home_matches_played'] >= 3) &
        (feature_df['away_matches_played'] >= 3)
    ].copy()

   
    X = feature_df[FEATURE_COLUMNS]
    y = feature_df['result']  # 1 = home win, 0 = draw, -1 = away win

    # Safety net: drop any remaining rows with NaN features, just in case
    valid_rows = X.notna().all(axis=1)
    if (~valid_rows).sum() > 0:
        print(f"Dropping {(~valid_rows).sum()} rows with missing feature values")
    X = X[valid_rows]
    y = y[valid_rows]
    feature_df = feature_df[valid_rows]

    feature_df = feature_df.sort_values('date')
    split_idx = int(len(feature_df) * 0.85)

    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]

    print(f"Training set: {len(X_train)} matches")
    print(f"Test set: {len(X_test)} matches (most recent matches, unseen by model)")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print()
    print("=" * 60)
    print("STEP 4: Training models...")
    print("=" * 60)

    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
        probs = model.predict_proba(X_test_scaled)

        acc = accuracy_score(y_test, preds)
        ll = log_loss(y_test, probs)

        results[name] = {'model': model, 'accuracy': acc, 'log_loss': ll}

        print(f"\n{name}:")
        print(f"  Accuracy: {acc:.3f}")
        print(f"  Log Loss: {ll:.3f} (lower is better - measures probability quality)")
        print(classification_report(y_test, preds, target_names=['Away Win', 'Draw', 'Home Win']))

    print()
    print("=" * 60)
    print("STEP 5: Selecting best model...")
    print("=" * 60)

    best_name = min(results, key=lambda k: results[k]['log_loss'])
    best_model = results[best_name]['model']
    print(f"Best model: {best_name}")

    with open(MODEL_OUTPUT_PATH, 'wb') as f:
        pickle.dump(best_model, f)
    with open(SCALER_OUTPUT_PATH, 'wb') as f:
        pickle.dump(scaler, f)

    print(f"\nSaved model to {MODEL_OUTPUT_PATH}")
    print(f"Saved scaler to {SCALER_OUTPUT_PATH}")
    print()
    print("Baseline comparison: always predicting 'home win' would give "
          f"{(y_test == 1).mean():.3f} accuracy.")
    print("Our model should beat this baseline meaningfully to be considered useful.")


if __name__ == "__main__":
    main()