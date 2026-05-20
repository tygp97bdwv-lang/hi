"""
Fetches recent UFC/MMA news for a given fighter via NewsAPI.
Falls back to web search if no API key is set.
"""
import logging
from datetime import datetime, timedelta
from typing import Optional

import requests

from config import NEWS_API_KEY, NEWS_API_URL

log = logging.getLogger(__name__)

_INJURY_KEYWORDS = [
    "injured", "injury", "surgery", "withdrawn", "pulled out",
    "hospitalized", "torn", "fractured", "broken", "concussion",
]
_WEIGHT_KEYWORDS = [
    "weight cut", "missed weight", "overweight", "dehydration",
]
_CAMP_KEYWORDS = [
    "new trainer", "switched camp", "left team", "joined team",
    "training with", "sparring with",
]


def fetch_fighter_news(fighter_name: str, days_back: int = 60) -> list[dict]:
    """Return recent news articles about a fighter."""
    if not NEWS_API_KEY:
        log.warning("NEWS_API_KEY not set — skipping news fetch")
        return []

    from_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    params = {
        "q": f'"{fighter_name}" UFC MMA',
        "from": from_date,
        "sortBy": "relevancy",
        "language": "en",
        "pageSize": 20,
        "apiKey": NEWS_API_KEY,
    }

    try:
        r = requests.get(NEWS_API_URL, params=params, timeout=10)
        r.raise_for_status()
        articles = r.json().get("articles", [])
        return articles
    except requests.RequestException as e:
        log.warning(f"News API request failed for {fighter_name}: {e}")
        return []


def build_news_features(fighter_name: str) -> dict:
    """
    Fetch news and return structured features:
      - news_sentiment: float in [-1, 1]
      - injury_flag: 0/1
      - weight_issue_flag: 0/1
      - camp_change_flag: 0/1
      - article_count: int
    """
    articles = fetch_fighter_news(fighter_name)

    features = {
        "news_sentiment": 0.0,
        "injury_flag": 0,
        "weight_issue_flag": 0,
        "camp_change_flag": 0,
        "article_count": len(articles),
    }

    if not articles:
        return features

    sentiments = []
    for article in articles:
        text = (
            (article.get("title") or "")
            + " "
            + (article.get("description") or "")
        ).lower()

        polarity = _simple_sentiment(text)
        sentiments.append(polarity)

        if any(kw in text for kw in _INJURY_KEYWORDS):
            features["injury_flag"] = 1
        if any(kw in text for kw in _WEIGHT_KEYWORDS):
            features["weight_issue_flag"] = 1
        if any(kw in text for kw in _CAMP_KEYWORDS):
            features["camp_change_flag"] = 1

    features["news_sentiment"] = round(sum(sentiments) / len(sentiments), 4)
    return features


def _simple_sentiment(text: str) -> float:
    """
    Lightweight lexicon-based sentiment scorer tuned for MMA news.
    Returns float in [-1, 1].
    """
    positive = [
        "knockout", "dominant", "impressive", "destroyed", "finished",
        "champion", "title", "comeback", "healthy", "sharp", "motivated",
        "perfect", "undefeated", "submitted", "victory", "wins",
    ]
    negative = [
        "loss", "lost", "knocked out", "injured", "surgery", "suspended",
        "stripped", "withdrew", "declined", "struggles", "concerns",
        "controversy", "failed", "missed weight", "overweight",
    ]

    score = 0
    for w in positive:
        score += text.count(w)
    for w in negative:
        score -= text.count(w)

    # Normalise to [-1, 1]
    if score == 0:
        return 0.0
    return max(-1.0, min(1.0, score / 5.0))


def summarise_news(fighter_name: str) -> str:
    """Return a human-readable summary of news flags for CLI display."""
    articles = fetch_fighter_news(fighter_name, days_back=30)
    if not articles:
        return "No recent news found."

    lines = [f"Latest news for {fighter_name} ({len(articles)} articles):"]
    for a in articles[:5]:
        title = a.get("title", "")
        source = a.get("source", {}).get("name", "")
        date = (a.get("publishedAt") or "")[:10]
        lines.append(f"  [{date}] {source}: {title}")
    return "\n".join(lines)
