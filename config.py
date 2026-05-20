import os
from dotenv import load_dotenv

load_dotenv()

UFC_STATS_BASE = "http://ufcstats.com"
UFC_FIGHTERS_URL = f"{UFC_STATS_BASE}/statistics/fighters?char={{letter}}&page=all"
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")
NEWS_API_URL = "https://newsapi.org/v2/everything"

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")

FIGHTERS_CSV = os.path.join(RAW_DIR, "fighters.csv")
FIGHTS_CSV = os.path.join(RAW_DIR, "fights.csv")
FEATURES_CSV = os.path.join(PROCESSED_DIR, "features.csv")
MODEL_PATH = os.path.join(MODEL_DIR, "ufc_xgb_model.joblib")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")

SCRAPE_DELAY = 0.5  # seconds between requests

FIGHTER_STAT_COLS = [
    "slpm",        # significant strikes landed per minute
    "str_acc",     # striking accuracy %
    "sapm",        # significant strikes absorbed per minute
    "str_def",     # strike defense %
    "td_avg",      # takedowns per 15 min
    "td_acc",      # takedown accuracy %
    "td_def",      # takedown defense %
    "sub_avg",     # submission attempts per 15 min
    "height_cm",
    "reach_cm",
    "age",
    "win_streak",
    "loss_streak",
    "ko_win_rate",
    "sub_win_rate",
    "dec_win_rate",
]
