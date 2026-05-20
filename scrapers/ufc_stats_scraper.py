"""
Scrapes fighter profiles and fight history from ufcstats.com.
"""
import time
import re
import string
import logging
from datetime import datetime
from typing import Optional

import requests
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm

from config import UFC_STATS_BASE, UFC_FIGHTERS_URL, SCRAPE_DELAY, FIGHTERS_CSV, FIGHTS_CSV

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}


def _get(url: str, retries: int = 3) -> Optional[BeautifulSoup]:
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=15)
            r.raise_for_status()
            time.sleep(SCRAPE_DELAY)
            return BeautifulSoup(r.text, "lxml")
        except requests.RequestException as e:
            log.warning(f"Attempt {attempt+1}/{retries} failed for {url}: {e}")
            time.sleep(2 ** attempt)
    return None


def _parse_inches_to_cm(val: str) -> Optional[float]:
    """Convert '6' 2\"' or '74\"' style strings to centimeters."""
    if not val or val.strip() == "--":
        return None
    val = val.strip()
    m = re.match(r"(\d+)'\s*(\d+)\"", val)
    if m:
        return round((int(m.group(1)) * 12 + int(m.group(2))) * 2.54, 1)
    m = re.match(r'(\d+)"', val)
    if m:
        return round(int(m.group(1)) * 2.54, 1)
    return None


def _parse_pct(val: str) -> Optional[float]:
    if not val or val.strip() in ("--", ""):
        return None
    return float(val.strip().replace("%", "")) / 100.0


def _parse_float(val: str) -> Optional[float]:
    if not val or val.strip() in ("--", ""):
        return None
    try:
        return float(val.strip())
    except ValueError:
        return None


def scrape_fighter_list() -> list[dict]:
    """Fetch all fighter URLs from the A-Z fighter index."""
    fighters = []
    for letter in string.ascii_lowercase:
        url = UFC_FIGHTERS_URL.format(letter=letter)
        soup = _get(url)
        if not soup:
            continue
        rows = soup.select("table.b-statistics__table tbody tr")
        for row in rows:
            cols = row.find_all("td")
            if len(cols) < 2:
                continue
            link = cols[0].find("a")
            if not link:
                continue
            name = f"{link.text.strip()} {cols[1].text.strip()}".strip()
            href = link.get("href", "")
            if href:
                fighters.append({"name": name, "url": href})
    log.info(f"Found {len(fighters)} fighters")
    return fighters


def scrape_fighter_profile(url: str) -> dict:
    """Scrape stats from a single fighter's profile page."""
    soup = _get(url)
    if not soup:
        return {}

    data = {"url": url}

    name_tag = soup.select_one("span.b-content__title-highlight")
    data["name"] = name_tag.text.strip() if name_tag else ""

    record_tag = soup.select_one("span.b-content__title-record")
    if record_tag:
        record_text = record_tag.text.strip().replace("Record:", "").strip()
        parts = re.findall(r"\d+", record_text)
        data["wins"] = int(parts[0]) if len(parts) > 0 else 0
        data["losses"] = int(parts[1]) if len(parts) > 1 else 0
        data["draws"] = int(parts[2]) if len(parts) > 2 else 0

    # Physical attributes box
    info_items = soup.select("li.b-list__box-list-item")
    for item in info_items:
        text = item.text.strip()
        if "Height:" in text:
            data["height_cm"] = _parse_inches_to_cm(text.replace("Height:", "").strip())
        elif "Weight:" in text:
            pass  # weight class handled separately
        elif "Reach:" in text:
            data["reach_cm"] = _parse_inches_to_cm(text.replace("Reach:", "").strip())
        elif "STANCE:" in text:
            data["stance"] = text.replace("STANCE:", "").strip()
        elif "DOB:" in text:
            dob_str = text.replace("DOB:", "").strip()
            try:
                dob = datetime.strptime(dob_str, "%b %d, %Y")
                data["age"] = round((datetime.now() - dob).days / 365.25, 1)
            except ValueError:
                data["age"] = None

    # Career stats
    stat_labels = soup.select("li.b-list__box-list-item_type_block")
    for item in stat_labels:
        spans = item.find_all("i")
        if len(spans) < 2:
            continue
        label = spans[0].text.strip()
        value = spans[1].text.strip()
        if label == "SLpM:":
            data["slpm"] = _parse_float(value)
        elif label == "Str. Acc.:":
            data["str_acc"] = _parse_pct(value)
        elif label == "SApM:":
            data["sapm"] = _parse_float(value)
        elif label == "Str. Def:":
            data["str_def"] = _parse_pct(value)
        elif label == "TD Avg.:":
            data["td_avg"] = _parse_float(value)
        elif label == "TD Acc.:":
            data["td_acc"] = _parse_pct(value)
        elif label == "TD Def.:":
            data["td_def"] = _parse_pct(value)
        elif label == "Sub. Avg.:":
            data["sub_avg"] = _parse_float(value)

    # Win breakdown from fight history
    fight_rows = soup.select("table.b-fight-details__table tbody tr")
    ko_wins = sub_wins = dec_wins = total_wins = 0
    streak_val = 0
    streak_type = None
    win_streak = loss_streak = 0

    for row in fight_rows:
        cols = row.find_all("td")
        if len(cols) < 8:
            continue
        result = cols[0].text.strip().upper()
        method = cols[7].text.strip().upper() if len(cols) > 7 else ""

        if result == "WIN":
            total_wins += 1
            if "KO" in method or "TKO" in method:
                ko_wins += 1
            elif "SUB" in method:
                sub_wins += 1
            else:
                dec_wins += 1
            if streak_type is None or streak_type == "W":
                streak_type = "W"
                streak_val += 1
            else:
                if streak_type == "W":
                    win_streak = max(win_streak, streak_val)
                streak_type = "W"
                streak_val = 1
        elif result == "LOSS":
            if streak_type is None or streak_type == "L":
                streak_type = "L"
                streak_val += 1
            else:
                if streak_type == "L":
                    loss_streak = max(loss_streak, streak_val)
                streak_type = "L"
                streak_val = 1

    if streak_type == "W":
        win_streak = streak_val
    elif streak_type == "L":
        loss_streak = streak_val

    data["win_streak"] = win_streak
    data["loss_streak"] = loss_streak
    data["ko_win_rate"] = round(ko_wins / total_wins, 3) if total_wins > 0 else 0.0
    data["sub_win_rate"] = round(sub_wins / total_wins, 3) if total_wins > 0 else 0.0
    data["dec_win_rate"] = round(dec_wins / total_wins, 3) if total_wins > 0 else 0.0

    # Last fight date
    date_cells = soup.select("td.b-fight-details__table-col:nth-child(9)")
    if date_cells:
        try:
            last_date = datetime.strptime(date_cells[0].text.strip(), "%b. %d, %Y")
            data["days_since_last_fight"] = (datetime.now() - last_date).days
        except ValueError:
            data["days_since_last_fight"] = None

    return data


def scrape_all_fighters(limit: Optional[int] = None) -> pd.DataFrame:
    """Scrape full fighter roster and profiles. Saves to CSV."""
    fighter_list = scrape_fighter_list()
    if limit:
        fighter_list = fighter_list[:limit]

    records = []
    for f in tqdm(fighter_list, desc="Scraping fighter profiles"):
        profile = scrape_fighter_profile(f["url"])
        if profile:
            records.append(profile)

    df = pd.DataFrame(records)
    df.to_csv(FIGHTERS_CSV, index=False)
    log.info(f"Saved {len(df)} fighter profiles to {FIGHTERS_CSV}")
    return df


def scrape_fight_results() -> pd.DataFrame:
    """Scrape the event list and individual fight results for training data."""
    events_url = f"{UFC_STATS_BASE}/statistics/events/completed?page=all"
    soup = _get(events_url)
    if not soup:
        return pd.DataFrame()

    event_links = [
        a["href"] for a in soup.select("a.b-link.b-link_style_black")
        if "/event-details/" in a.get("href", "")
    ]
    log.info(f"Found {len(event_links)} events")

    fights = []
    for event_url in tqdm(event_links, desc="Scraping events"):
        event_soup = _get(event_url)
        if not event_soup:
            continue

        event_name_tag = event_soup.select_one("span.b-content__title-highlight")
        event_name = event_name_tag.text.strip() if event_name_tag else ""

        fight_links = [
            a["href"] for a in event_soup.select("a.b-flag")
            if "/fight-details/" in a.get("href", "")
        ]

        for fight_url in fight_links:
            fight_data = _scrape_fight_detail(fight_url, event_name)
            if fight_data:
                fights.append(fight_data)

    df = pd.DataFrame(fights)
    df.to_csv(FIGHTS_CSV, index=False)
    log.info(f"Saved {len(df)} fights to {FIGHTS_CSV}")
    return df


def _scrape_fight_detail(url: str, event_name: str) -> Optional[dict]:
    soup = _get(url)
    if not soup:
        return None

    fighters = soup.select("a.b-link.b-fight-details__person-link")
    if len(fighters) < 2:
        return None

    result_tags = soup.select("i.b-fight-details__person-status")
    if len(result_tags) < 2:
        return None

    f1_name = fighters[0].text.strip()
    f2_name = fighters[1].text.strip()
    f1_result = result_tags[0].text.strip().upper()

    winner = f1_name if f1_result == "W" else (f2_name if f1_result == "L" else None)

    method_tag = soup.select_one("i.b-fight-details__text-item_first i:nth-of-type(2)")
    method = method_tag.text.strip() if method_tag else ""

    return {
        "fighter_1": f1_name,
        "fighter_2": f2_name,
        "winner": winner,
        "method": method,
        "event": event_name,
        "url": url,
    }


def search_fighter(name: str, df: pd.DataFrame) -> Optional[pd.Series]:
    """Fuzzy-match a fighter name from the scraped DataFrame."""
    name_lower = name.lower()
    exact = df[df["name"].str.lower() == name_lower]
    if not exact.empty:
        return exact.iloc[0]
    partial = df[df["name"].str.lower().str.contains(name_lower, na=False)]
    if not partial.empty:
        return partial.iloc[0]
    return None
