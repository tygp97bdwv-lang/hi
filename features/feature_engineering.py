"""
Builds differential feature vectors from two fighter stat rows.
The model always sees (fighter_A - fighter_B) so label=1 means A wins.
"""
import pandas as pd
import numpy as np
from typing import Optional

from config import FIGHTER_STAT_COLS

# Columns used in the differential feature vector
DIFF_COLS = [f"{c}_diff" for c in FIGHTER_STAT_COLS]
NEWS_FEATURE_COLS = [
    "f1_news_sentiment", "f1_injury_flag", "f1_weight_issue_flag", "f1_camp_change_flag",
    "f2_news_sentiment", "f2_injury_flag", "f2_weight_issue_flag", "f2_camp_change_flag",
    "news_sentiment_diff",
]
ALL_FEATURE_COLS = DIFF_COLS + ["stance_adv"] + NEWS_FEATURE_COLS


def _stance_advantage(stance_a: Optional[str], stance_b: Optional[str]) -> int:
    """
    Orthodox vs Southpaw creates a known stylistic edge.
    +1 if A is southpaw vs orthodox, -1 if reversed, 0 otherwise.
    """
    if not stance_a or not stance_b:
        return 0
    a = str(stance_a).strip().upper()
    b = str(stance_b).strip().upper()
    if a == "SOUTHPAW" and b == "ORTHODOX":
        return 1
    if a == "ORTHODOX" and b == "SOUTHPAW":
        return -1
    return 0


def build_diff_features(
    fighter_a: pd.Series,
    fighter_b: pd.Series,
    news_a: Optional[dict] = None,
    news_b: Optional[dict] = None,
) -> pd.Series:
    """
    Returns a single feature vector: fighter_a stats minus fighter_b stats,
    plus stance advantage and news-derived features.
    """
    news_a = news_a or {}
    news_b = news_b or {}

    feature_dict = {}

    for col in FIGHTER_STAT_COLS:
        val_a = _safe_float(fighter_a.get(col))
        val_b = _safe_float(fighter_b.get(col))
        if val_a is not None and val_b is not None:
            feature_dict[f"{col}_diff"] = val_a - val_b
        else:
            feature_dict[f"{col}_diff"] = 0.0

    feature_dict["stance_adv"] = _stance_advantage(
        fighter_a.get("stance"), fighter_b.get("stance")
    )

    # News features
    feature_dict["f1_news_sentiment"] = news_a.get("news_sentiment", 0.0)
    feature_dict["f1_injury_flag"] = news_a.get("injury_flag", 0)
    feature_dict["f1_weight_issue_flag"] = news_a.get("weight_issue_flag", 0)
    feature_dict["f1_camp_change_flag"] = news_a.get("camp_change_flag", 0)

    feature_dict["f2_news_sentiment"] = news_b.get("news_sentiment", 0.0)
    feature_dict["f2_injury_flag"] = news_b.get("injury_flag", 0)
    feature_dict["f2_weight_issue_flag"] = news_b.get("weight_issue_flag", 0)
    feature_dict["f2_camp_change_flag"] = news_b.get("camp_change_flag", 0)

    feature_dict["news_sentiment_diff"] = (
        feature_dict["f1_news_sentiment"] - feature_dict["f2_news_sentiment"]
    )

    return pd.Series(feature_dict)


def build_training_dataset(
    fighters_df: pd.DataFrame,
    fights_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Join fight history with fighter stats to produce a training DataFrame.
    Each row is one fight, label=1 if fighter_1 won.
    We augment with the flipped version (swap fighters) to balance the dataset.
    """
    fighter_lookup = {
        row["name"].lower(): row
        for _, row in fighters_df.iterrows()
        if pd.notna(row.get("name"))
    }

    rows = []
    for _, fight in fights_df.iterrows():
        f1_name = str(fight.get("fighter_1", "")).lower()
        f2_name = str(fight.get("fighter_2", "")).lower()
        winner = str(fight.get("winner", "")).lower()

        f1 = fighter_lookup.get(f1_name)
        f2 = fighter_lookup.get(f2_name)
        if f1 is None or f2 is None:
            continue
        if winner not in (f1_name, f2_name):
            continue  # draw or no contest — skip

        label = 1 if winner == f1_name else 0
        feats = build_diff_features(f1, f2)
        feats["label"] = label
        rows.append(feats)

        # Augment: flip the matchup
        feats_flipped = build_diff_features(f2, f1)
        feats_flipped["label"] = 1 - label
        rows.append(feats_flipped)

    df = pd.DataFrame(rows)
    df = df.fillna(0.0)
    return df


def _safe_float(val) -> Optional[float]:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None
