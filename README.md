\# World Cup 2026 Match Predictor



A machine learning model that predicts FIFA World Cup match outcomes (home win / draw / away win) using historical international football data, with live tracking against the actual 2026 tournament as it happens.



\## Why I Built This



Most sports prediction tutorials use simple win/loss accuracy as the only metric, and many overfit to a handful of headline features. I wanted to build something that handles a real, messy dataset (150+ years of international football results) properly — with correct time-series validation, no data leakage, and honest reporting of where the model struggles.



I also wanted live evidence, not just a backtest. Since the FIFA World Cup 2026 is happening during development, I built a logging system that records the model's predictions before each match and checks them against real results once matches finish — so the project has a running, verifiable accuracy track record instead of a single claimed number.



\## How It Works



\*\*1. Feature Engineering\*\* (`feature\_engineering.py`)

For every match, the model only uses information that was available \*before\* that match was played — recent form (last 10 matches), average goals scored/conceded, and historical head-to-head record between the two teams. This avoids data leakage, a common mistake in sports prediction projects where future information accidentally leaks into training data and makes the model look artificially good.



\*\*2. Model Training\*\* (`train\_model.py`)

Trains and compares two models — Logistic Regression and Random Forest — on matches since 2000. Uses a chronological train/test split (not random), so the model is tested on its ability to predict \*future\* matches from \*past\* data, exactly how it would be used in practice. Model selection is based on log loss (which rewards well-calibrated probabilities), not just raw accuracy.



\*\*3. Prediction \& Live Tracking\*\* (`predict\_matches.py`, `log\_prediction.py`)

Generates win/draw/loss probabilities for any matchup. The logging script records predictions before matches happen and lets me fill in actual results afterward, building a real, growing accuracy record across the tournament.



\## Results



\- \*\*Logistic Regression accuracy:\*\* 56.8% on held-out test matches (most recent \~3,700 matches from a 25,000+ match dataset)

\- \*\*Baseline ("always pick home team"):\*\* 47.3% accuracy

\- The model beats the naive baseline by \*\*\~9.5 percentage points\*\*



\### Known Limitation: Draws



The model rarely predicts draws, even though draws make up a meaningful share of real outcomes (and have shown up frequently in the actual 2026 tournament). This is a known, well-documented difficulty in football prediction — draws are an emergent outcome of two evenly-matched teams rather than a distinct pattern the model can easily learn from form/goal-based features. I chose to report this honestly rather than hide it, and I'm treating it as a direction for future improvement (see below) rather than something to paper over.



\## Live Tracking



`predictions\_log.csv` contains real predictions logged before actual 2026 World Cup matches, along with the results once they finished. This is updated throughout the tournament — see the file for the current running accuracy.



\## Tech Stack



\- Python, pandas, NumPy

\- scikit-learn (Logistic Regression, Random Forest)

\- Data: international football results dataset (1872–2026), sourced from Kaggle



\## What I'd Improve Next



\- Add features specifically targeting draw prediction (e.g. closeness of recent form between two teams, low-scoring tendency)

\- Incorporate FIFA world rankings as an additional feature

\- Build a Streamlit dashboard for interactive predictions and visualizing the live tracking results

\- Explore calibration curves to verify predicted probabilities match real-world outcome frequencies



\## Setup



```bash

pip install pandas numpy scikit-learn



\# 1. Place results.csv in the data/ folder

\# 2. Train the model (takes \~10 min due to feature building on 49k+ matches)

python train\_model.py



\# 3. Predict a specific matchup

python predict\_matches.py



\# 4. Log predictions before matches, then update with results after

python log\_prediction.py log

python log\_prediction.py update

```



\## Data Source



Historical match data: \[International football results from 1872 to 2026](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017) (Kaggle)

