"""
World Cup Match Predictor - Feature Engineering
--------------------------------------------------
Expects a CSV with columns (standard Kaggle 'international football results' format):
    date, home_team, away_team, home_score, away_score, tournament, city, country, neutral

Builds team-level features for predicting match outcomes:
    - Recent form (last 10/20 matches win rate)
    - Goals scored/conceded average (last 2 years)
    - Head-to-head win percentage
    - Host advantage flag
    - Tournament stage (group vs knockout) - inferred from 'tournament' column

Output: a clean feature matrix ready for model training.
"""

import pandas as pd
import numpy as np
from datetime import datetime


def load_and_clean(filepath):
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])

    # Drop matches with missing scores entirely - these are unusable for
    # both training labels AND for computing other teams' form features.
    # A handful of very old/obscure matches in 150+ years of data have
    # missing scores; dropping them is safe and standard practice.
    before = len(df)
    df = df.dropna(subset=['home_score', 'away_score']).copy()
    dropped = before - len(df)
    if dropped > 0:
        print(f"Dropped {dropped} matches with missing scores (data quality cleanup)")

    df['home_score'] = df['home_score'].astype(int)
    df['away_score'] = df['away_score'].astype(int)

    df = df.sort_values('date').reset_index(drop=True)
    return df


def get_match_result(row):
    """1 = home win, 0 = draw, -1 = away win"""
    if row['home_score'] > row['away_score']:
        return 1
    elif row['home_score'] < row['away_score']:
        return -1
    else:
        return 0


def team_recent_form(df, team, before_date, n_matches=10):
    """Win rate, goals scored/conceded avg over last n matches before a given date."""
    mask = ((df['home_team'] == team) | (df['away_team'] == team)) & (df['date'] < before_date)
    recent = df[mask].sort_values('date').tail(n_matches)

    if len(recent) == 0:
        return {'win_rate': 0.5, 'avg_goals_scored': 1.0, 'avg_goals_conceded': 1.0, 'matches_played': 0}

    wins, goals_for, goals_against = 0, 0, 0
    for _, r in recent.iterrows():
        if r['home_team'] == team:
            goals_for += r['home_score']
            goals_against += r['away_score']
            if r['home_score'] > r['away_score']:
                wins += 1
        else:
            goals_for += r['away_score']
            goals_against += r['home_score']
            if r['away_score'] > r['home_score']:
                wins += 1

    n = len(recent)
    return {
        'win_rate': wins / n,
        'avg_goals_scored': goals_for / n,
        'avg_goals_conceded': goals_against / n,
        'matches_played': n
    }


def head_to_head(df, team_a, team_b, before_date):
    """Team A's win rate against Team B historically."""
    mask = (
        (((df['home_team'] == team_a) & (df['away_team'] == team_b)) |
         ((df['home_team'] == team_b) & (df['away_team'] == team_a))) &
        (df['date'] < before_date)
    )
    h2h = df[mask]
    if len(h2h) == 0:
        return 0.5  # no history -> neutral prior

    a_wins = 0
    for _, r in h2h.iterrows():
        if r['home_team'] == team_a and r['home_score'] > r['away_score']:
            a_wins += 1
        elif r['away_team'] == team_a and r['away_score'] > r['home_score']:
            a_wins += 1
    return a_wins / len(h2h)


def build_feature_row(df, home_team, away_team, match_date, neutral_venue=True):
    """Builds one row of features for a hypothetical/real match."""
    home_form = team_recent_form(df, home_team, match_date)
    away_form = team_recent_form(df, away_team, match_date)
    h2h_rate = head_to_head(df, home_team, away_team, match_date)

    return {
        'home_team': home_team,
        'away_team': away_team,
        'home_win_rate': home_form['win_rate'],
        'away_win_rate': away_form['win_rate'],
        'home_avg_goals_scored': home_form['avg_goals_scored'],
        'home_avg_goals_conceded': home_form['avg_goals_conceded'],
        'away_avg_goals_scored': away_form['avg_goals_scored'],
        'away_avg_goals_conceded': away_form['avg_goals_conceded'],
        'form_diff': home_form['win_rate'] - away_form['win_rate'],
        'attack_vs_defense_home': home_form['avg_goals_scored'] - away_form['avg_goals_conceded'],
        'attack_vs_defense_away': away_form['avg_goals_scored'] - home_form['avg_goals_conceded'],
        'h2h_home_win_rate': h2h_rate,
        'neutral_venue': int(neutral_venue),
        'home_matches_played': home_form['matches_played'],
        'away_matches_played': away_form['matches_played'],
    }


def build_training_set(df, min_date='2000-01-01'):
    """
    Walks through historical matches chronologically and builds a feature row
    for each match using only information available BEFORE that match date
    (this avoids data leakage - critical for time-series sports prediction).
    """
    df = df[df['date'] >= min_date].copy()
    rows = []

    for idx, match in df.iterrows():
        feats = build_feature_row(
            df, match['home_team'], match['away_team'], match['date'],
            neutral_venue=match.get('neutral', False)
        )
        feats['result'] = get_match_result(match)
        feats['date'] = match['date']
        rows.append(feats)

        if idx % 500 == 0:
            print(f"Processed {idx} matches...")

    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("This script is meant to be imported. See train_model.py for usage.")