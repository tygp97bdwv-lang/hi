"""
Train and evaluate the XGBoost fighter prediction model.
"""
import os
import logging

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
)
from xgboost import XGBClassifier

from config import FEATURES_CSV, MODEL_PATH, SCALER_PATH, MODEL_DIR
from features.feature_engineering import ALL_FEATURE_COLS, build_training_dataset

log = logging.getLogger(__name__)


def train(fighters_df: pd.DataFrame = None, fights_df: pd.DataFrame = None) -> XGBClassifier:
    """
    Build features from raw DataFrames (or load from cache), train XGBoost,
    evaluate with stratified K-fold CV, and persist model + scaler.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)

    if fighters_df is not None and fights_df is not None:
        log.info("Building training features from scraped data...")
        df = build_training_dataset(fighters_df, fights_df)
        df.to_csv(FEATURES_CSV, index=False)
    elif os.path.exists(FEATURES_CSV):
        log.info(f"Loading cached features from {FEATURES_CSV}")
        df = pd.read_csv(FEATURES_CSV)
    else:
        raise FileNotFoundError(
            "No features CSV found. Run the scraper first with: python main.py scrape"
        )

    feature_cols = [c for c in ALL_FEATURE_COLS if c in df.columns]
    X = df[feature_cols].fillna(0.0).values
    y = df["label"].values

    log.info(f"Training on {len(df)} samples with {len(feature_cols)} features")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = XGBClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring="accuracy")
    auc_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring="roc_auc")

    log.info(f"CV Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    log.info(f"CV AUC:      {auc_scores.mean():.3f} ± {auc_scores.std():.3f}")

    model.fit(X_scaled, y)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    log.info(f"Model saved to {MODEL_PATH}")

    _print_feature_importance(model, feature_cols)
    return model


def _print_feature_importance(model: XGBClassifier, feature_cols: list[str]) -> None:
    importances = model.feature_importances_
    ranked = sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True)
    print("\nTop 10 most important features:")
    for name, score in ranked[:10]:
        bar = "#" * int(score * 50)
        print(f"  {name:<30s} {bar} {score:.4f}")
