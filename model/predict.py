"""
Prediction engine: loads model + scaler and returns win probabilities.
"""
import os
import logging
from typing import Optional

import joblib
import pandas as pd
import numpy as np

from config import MODEL_PATH, SCALER_PATH, FIGHTERS_CSV
from features.feature_engineering import ALL_FEATURE_COLS, build_diff_features
from scrapers.ufc_stats_scraper import scrape_fighter_profile, search_fighter
from scrapers.news_scraper import build_news_features, summarise_news

log = logging.getLogger(__name__)


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Train first: python main.py train"
        )
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


def _load_or_scrape_fighter(name: str, fighters_df: Optional[pd.DataFrame]) -> pd.Series:
    """Look up fighter from cache, or scrape live if not found."""
    if fighters_df is not None:
        cached = search_fighter(name, fighters_df)
        if cached is not None:
            log.info(f"Using cached profile for {name}")
            return cached

    log.info(f"Scraping live profile for {name}...")
    # Search the fighter index to get their URL
    from scrapers.ufc_stats_scraper import _get, UFC_STATS_BASE
    import urllib.parse
    search_url = (
        f"{UFC_STATS_BASE}/statistics/fighters/search?query={urllib.parse.quote(name)}"
    )
    soup = _get(search_url)
    if soup:
        link = soup.select_one("a.b-link.b-link_style_black")
        if link and link.get("href"):
            profile = scrape_fighter_profile(link["href"])
            if profile:
                return pd.Series(profile)

    raise ValueError(f"Fighter '{name}' not found. Check spelling or run 'python main.py scrape' first.")


def predict_matchup(
    fighter_a_name: str,
    fighter_b_name: str,
    fighters_df: Optional[pd.DataFrame] = None,
    fetch_news: bool = True,
) -> dict:
    """
    Predict the outcome of fighter_a vs fighter_b.

    Returns:
        {
            "fighter_a": str,
            "fighter_b": str,
            "prob_a_wins": float,
            "prob_b_wins": float,
            "predicted_winner": str,
            "confidence": str,  # "Low" / "Medium" / "High"
            "news_a": dict,
            "news_b": dict,
        }
    """
    model, scaler = load_model()

    # Load fighter stats
    if fighters_df is None and os.path.exists(FIGHTERS_CSV):
        fighters_df = pd.read_csv(FIGHTERS_CSV)

    fighter_a = _load_or_scrape_fighter(fighter_a_name, fighters_df)
    fighter_b = _load_or_scrape_fighter(fighter_b_name, fighters_df)

    # Fetch news
    news_a = build_news_features(fighter_a_name) if fetch_news else {}
    news_b = build_news_features(fighter_b_name) if fetch_news else {}

    # Build feature vector
    feature_cols = ALL_FEATURE_COLS
    feat_series = build_diff_features(fighter_a, fighter_b, news_a, news_b)

    X = np.array([[feat_series.get(c, 0.0) for c in feature_cols]])
    X_scaled = scaler.transform(X)

    proba = model.predict_proba(X_scaled)[0]
    prob_a = float(proba[1])
    prob_b = float(proba[0])

    margin = abs(prob_a - prob_b)
    if margin < 0.1:
        confidence = "Low"
    elif margin < 0.25:
        confidence = "Medium"
    else:
        confidence = "High"

    return {
        "fighter_a": fighter_a.get("name", fighter_a_name),
        "fighter_b": fighter_b.get("name", fighter_b_name),
        "prob_a_wins": round(prob_a, 4),
        "prob_b_wins": round(prob_b, 4),
        "predicted_winner": fighter_a.get("name", fighter_a_name) if prob_a > prob_b else fighter_b.get("name", fighter_b_name),
        "confidence": confidence,
        "news_a": news_a,
        "news_b": news_b,
        "features": feat_series.to_dict(),
    }
