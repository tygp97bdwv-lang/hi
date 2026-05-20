#!/usr/bin/env python3
"""
Quick demo: generates realistic synthetic UFC data, trains the model,
and runs a sample prediction — no scraping or API key needed.

Usage:
  python demo.py
"""
import os
import sys
import random
import logging
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
logging.basicConfig(level=logging.WARNING)

from config import RAW_DIR, PROCESSED_DIR, MODEL_DIR, FIGHTERS_CSV
from features.feature_engineering import build_training_dataset, ALL_FEATURE_COLS
from model.train import train

random.seed(42)
np.random.seed(42)

STANCES = ["Orthodox", "Southpaw", "Switch"]
WEIGHT_CLASSES = ["Flyweight", "Bantamweight", "Featherweight", "Lightweight",
                  "Welterweight", "Middleweight", "Light Heavyweight", "Heavyweight"]

REAL_FIGHTERS = [
    # (name, slpm, str_acc, sapm, str_def, td_avg, td_acc, td_def, sub_avg, height_cm, reach_cm, stance, age, wins, losses)
    ("Jon Jones",        4.3, 0.57, 2.2, 0.65, 1.9, 0.43, 0.93, 0.9, 193, 215, "Orthodox",  37, 27, 1),
    ("Stipe Miocic",     7.6, 0.53, 5.9, 0.57, 1.5, 0.50, 0.69, 0.1, 193, 201, "Orthodox",  41, 20, 4),
    ("Khabib Nurmagomedov", 4.1, 0.49, 2.4, 0.64, 5.3, 0.47, 0.84, 1.8, 178, 178, "Orthodox", 35, 29, 0),
    ("Israel Adesanya",  6.7, 0.47, 3.1, 0.61, 0.3, 0.40, 0.76, 0.2, 193, 203, "Orthodox",  34, 24, 4),
    ("Alexander Volkanovski", 6.0, 0.56, 4.2, 0.54, 2.5, 0.49, 0.72, 0.8, 168, 182, "Orthodox", 35, 26, 3),
    ("Charles Oliveira",  5.4, 0.54, 5.2, 0.46, 2.1, 0.39, 0.60, 3.4, 178, 183, "Orthodox",  34, 34, 10),
    ("Kamaru Usman",     5.3, 0.56, 3.2, 0.60, 3.7, 0.57, 0.85, 0.4, 183, 193, "Orthodox",  37, 20, 3),
    ("Francis Ngannou",  4.9, 0.47, 4.7, 0.50, 1.6, 0.67, 0.58, 0.1, 193, 211, "Orthodox",  37, 17, 3),
    ("Valentina Shevchenko", 6.2, 0.54, 3.1, 0.62, 3.4, 0.57, 0.78, 1.1, 165, 168, "Southpaw", 36, 23, 4),
    ("Amanda Nunes",     7.0, 0.52, 5.3, 0.53, 2.3, 0.52, 0.68, 0.9, 170, 173, "Orthodox",  36, 22, 5),
    ("Conor McGregor",   5.4, 0.49, 4.7, 0.57, 0.7, 0.53, 0.63, 0.5, 175, 188, "Southpaw",  36, 22, 6),
    ("Max Holloway",     7.9, 0.47, 5.0, 0.55, 0.9, 0.35, 0.76, 0.2, 180, 175, "Orthodox",  32, 25, 7),
    ("Dustin Poirier",   6.0, 0.47, 5.0, 0.50, 1.7, 0.55, 0.64, 1.2, 175, 183, "Orthodox",  35, 30, 9),
    ("Leon Edwards",     5.0, 0.49, 3.3, 0.60, 1.2, 0.48, 0.71, 0.4, 183, 188, "Orthodox",  32, 21, 3),
    ("Sean O'Malley",    6.2, 0.59, 3.5, 0.62, 0.3, 0.31, 0.93, 0.1, 180, 183, "Orthodox",  29, 17, 1),
    ("Alex Pereira",     5.5, 0.53, 5.5, 0.50, 0.5, 0.29, 0.55, 0.1, 193, 203, "Orthodox",  36, 11, 2),
    ("Ciryl Gane",       6.3, 0.52, 4.4, 0.59, 1.3, 0.44, 0.82, 0.3, 196, 211, "Orthodox",  34, 13, 2),
    ("Tom Aspinall",     5.8, 0.56, 3.8, 0.60, 2.0, 0.67, 0.80, 1.4, 196, 208, "Orthodox",  31, 14, 3),
    ("Islam Makhachev",  4.3, 0.51, 2.6, 0.67, 4.7, 0.47, 0.87, 1.2, 175, 178, "Orthodox",  32, 25, 1),
    ("Sean Strickland",  8.0, 0.49, 6.0, 0.53, 1.0, 0.43, 0.60, 0.3, 185, 193, "Orthodox",  32, 28, 6),
    ("Ilia Topuria",     6.4, 0.58, 3.8, 0.64, 1.2, 0.50, 0.82, 1.8, 170, 175, "Orthodox",  27, 16, 0),
    ("Arman Tsarukyan",  5.8, 0.51, 3.9, 0.59, 4.1, 0.45, 0.72, 0.7, 170, 175, "Orthodox",  27, 21, 3),
    ("Paddy Pimblett",   5.2, 0.50, 5.1, 0.48, 2.0, 0.44, 0.62, 1.5, 175, 178, "Orthodox",  29, 21, 3),
    ("Dricus Du Plessis", 6.1, 0.52, 5.3, 0.54, 1.4, 0.48, 0.68, 0.8, 185, 193, "Orthodox", 30, 22, 2),
    ("Merab Dvalishvili", 7.2, 0.46, 5.6, 0.51, 7.5, 0.44, 0.67, 0.5, 168, 170, "Orthodox", 33, 17, 4),
    ("Shavkat Rakhmonov", 5.6, 0.56, 2.9, 0.64, 2.1, 0.61, 0.81, 1.2, 185, 193, "Orthodox", 29, 18, 0),
    ("Bo Nickal",        4.8, 0.55, 3.2, 0.60, 6.2, 0.70, 0.85, 1.4, 185, 188, "Orthodox", 27, 7, 0),
    ("Brendan Allen",    5.5, 0.51, 4.8, 0.54, 2.8, 0.52, 0.70, 1.6, 183, 190, "Orthodox", 28, 26, 5),
    ("Umar Nurmagomedov", 5.0, 0.52, 2.8, 0.66, 4.2, 0.50, 0.84, 0.9, 173, 178, "Orthodox", 27, 17, 0),
]


def _make_fighter_row(data: tuple) -> dict:
    name, slpm, str_acc, sapm, str_def, td_avg, td_acc, td_def, sub_avg, h, r, stance, age, w, l = data
    total = w if w > 0 else 1
    return {
        "name": name, "wins": w, "losses": l, "draws": 0,
        "slpm": slpm, "str_acc": str_acc, "sapm": sapm, "str_def": str_def,
        "td_avg": td_avg, "td_acc": td_acc, "td_def": td_def, "sub_avg": sub_avg,
        "height_cm": h, "reach_cm": r, "stance": stance, "age": age,
        "win_streak": random.randint(1, 5),
        "loss_streak": 0 if w > l else random.randint(1, 2),
        "ko_win_rate": round(random.uniform(0.2, 0.6), 2),
        "sub_win_rate": round(random.uniform(0.1, 0.4), 2),
        "dec_win_rate": round(random.uniform(0.2, 0.5), 2),
        "days_since_last_fight": random.randint(90, 400),
        "url": f"http://example.com/{name.lower().replace(' ', '-')}",
    }


def _generate_synthetic_fighters(n: int = 300) -> pd.DataFrame:
    """Generate synthetic fighters with plausible stat distributions."""
    rows = [_make_fighter_row(f) for f in REAL_FIGHTERS]

    for i in range(n):
        w = random.randint(8, 28)
        l = random.randint(1, 10)
        total = max(w, 1)
        ko = round(random.uniform(0.15, 0.55), 2)
        sub = round(random.uniform(0.10, 0.40), 2)
        dec = round(1.0 - ko - sub, 2)
        rows.append({
            "name": f"Fighter_{i:04d}",
            "wins": w, "losses": l, "draws": random.randint(0, 2),
            "slpm": round(np.random.normal(4.5, 1.5), 2),
            "str_acc": round(np.clip(np.random.normal(0.48, 0.07), 0.25, 0.75), 2),
            "sapm": round(np.random.normal(4.0, 1.5), 2),
            "str_def": round(np.clip(np.random.normal(0.55, 0.08), 0.30, 0.80), 2),
            "td_avg": round(np.clip(np.random.normal(2.0, 1.5), 0, 7), 2),
            "td_acc": round(np.clip(np.random.normal(0.44, 0.12), 0.10, 0.90), 2),
            "td_def": round(np.clip(np.random.normal(0.62, 0.14), 0.20, 0.95), 2),
            "sub_avg": round(np.clip(np.random.normal(0.8, 0.9), 0, 4), 2),
            "height_cm": round(np.random.normal(178, 8), 1),
            "reach_cm": round(np.random.normal(182, 8), 1),
            "stance": random.choice(STANCES),
            "age": round(np.random.normal(30, 4), 1),
            "win_streak": random.randint(0, 6),
            "loss_streak": random.randint(0, 3),
            "ko_win_rate": ko, "sub_win_rate": sub, "dec_win_rate": max(dec, 0.0),
            "days_since_last_fight": random.randint(60, 600),
            "url": f"http://example.com/fighter-{i}",
        })
    return pd.DataFrame(rows)


def _simulate_fight(f1: pd.Series, f2: pd.Series) -> str:
    """
    Simulate who wins based on stats. Higher SLpM+str_acc and str_def
    vs the opponent's sapm drives the outcome, with some randomness.
    """
    def score(a, b):
        s = 0.0
        s += (a.get("slpm", 4.5) * a.get("str_acc", 0.48)) - (b.get("sapm", 4.0) * (1 - b.get("str_def", 0.55)))
        s += a.get("td_avg", 2.0) * a.get("td_acc", 0.44) * (1 - b.get("td_def", 0.62))
        s += a.get("sub_avg", 0.8) * 0.3
        s += (a.get("win_streak", 0) - a.get("loss_streak", 0)) * 0.1
        s += (a.get("reach_cm", 182) - b.get("reach_cm", 182)) * 0.01
        return s

    s1 = score(f1, f2)
    s2 = score(f2, f1)
    # Convert to probability with noise
    diff = s1 - s2
    prob_f1 = 1 / (1 + np.exp(-diff * 0.8))
    prob_f1 = np.clip(prob_f1 + np.random.normal(0, 0.08), 0.05, 0.95)
    return f1["name"] if np.random.random() < prob_f1 else f2["name"]


def _generate_fights(fighters_df: pd.DataFrame, n: int = 3000) -> pd.DataFrame:
    rows = fighters_df.to_dict("records")
    fights = []
    for _ in range(n):
        f1, f2 = random.sample(rows, 2)
        winner = _simulate_fight(pd.Series(f1), pd.Series(f2))
        fights.append({
            "fighter_1": f1["name"],
            "fighter_2": f2["name"],
            "winner": winner,
            "method": random.choice(["KO/TKO", "Submission", "Decision - Unanimous"]),
            "event": "Demo Event",
            "url": "http://example.com/fight",
        })
    return pd.DataFrame(fights)


def run_demo():
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("Generating synthetic fighter data...")
    fighters_df = _generate_synthetic_fighters(300)
    fights_df = _generate_fights(fighters_df, 3000)

    print(f"  {len(fighters_df)} fighters, {len(fights_df)} fights generated")

    # Save fighters to disk so 'python main.py predict' can find them
    fighters_df.to_csv(FIGHTERS_CSV, index=False)

    print("\nTraining XGBoost model...")
    train(fighters_df, fights_df)

    print("\nRunning sample predictions...\n")
    from model.predict import predict_matchup
    from main import _print_prediction

    matchups = [
        ("Ilia Topuria", "Arman Tsarukyan"),
        ("Islam Makhachev", "Dustin Poirier"),
        ("Alex Pereira", "Israel Adesanya"),
        ("Jon Jones", "Tom Aspinall"),
    ]

    for a, b in matchups:
        result = predict_matchup(a, b, fighters_df=fighters_df, fetch_news=False)
        _print_prediction(result)

    print("Done! You can now run any matchup:")
    print('  python main.py predict "Ilia Topuria" "Arman Tsarukyan"')


if __name__ == "__main__":
    run_demo()
