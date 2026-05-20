#!/usr/bin/env python3
"""
UFC Fighter Prediction CLI

Usage:
  python main.py scrape                        # Scrape all UFC fighter profiles
  python main.py train                         # Train the XGBoost model
  python main.py predict "Jon Jones" "Stipe Miocic"
  python main.py news "Jon Jones"              # Show recent news for a fighter
  python main.py info "Jon Jones"              # Show fighter stats from cache
"""
import sys
import os
import logging

import pandas as pd
from colorama import Fore, Style, init as colorama_init
from tabulate import tabulate

colorama_init(autoreset=True)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(__file__))

from config import FIGHTERS_CSV, FIGHTS_CSV


def cmd_scrape(args):
    from scrapers.ufc_stats_scraper import scrape_all_fighters, scrape_fight_results

    limit = int(args[0]) if args else None
    print(f"{Fore.CYAN}Scraping UFC fighter profiles{' (limit: ' + str(limit) + ')' if limit else ''}...")
    fighters_df = scrape_all_fighters(limit=limit)
    print(f"{Fore.GREEN}Scraped {len(fighters_df)} fighters")

    print(f"{Fore.CYAN}Scraping fight results for training data...")
    fights_df = scrape_fight_results()
    print(f"{Fore.GREEN}Scraped {len(fights_df)} fights")
    return fighters_df, fights_df


def cmd_train(args):
    from model.train import train

    fighters_df = None
    fights_df = None

    if os.path.exists(FIGHTERS_CSV) and os.path.exists(FIGHTS_CSV):
        fighters_df = pd.read_csv(FIGHTERS_CSV)
        fights_df = pd.read_csv(FIGHTS_CSV)

    print(f"{Fore.CYAN}Training XGBoost model...")
    train(fighters_df, fights_df)
    print(f"{Fore.GREEN}Model trained and saved.")


def cmd_predict(args):
    if len(args) < 2:
        print(f"{Fore.RED}Usage: python main.py predict \"Fighter A\" \"Fighter B\"")
        sys.exit(1)

    from model.predict import predict_matchup

    fighter_a = args[0]
    fighter_b = args[1]

    fighters_df = None
    if os.path.exists(FIGHTERS_CSV):
        fighters_df = pd.read_csv(FIGHTERS_CSV)

    print(f"\n{Fore.CYAN}Analysing matchup: {fighter_a} vs {fighter_b}...")
    result = predict_matchup(fighter_a, fighter_b, fighters_df=fighters_df)

    _print_prediction(result)

    # Show news summaries if available
    from scrapers.news_scraper import NEWS_API_KEY
    if NEWS_API_KEY:
        from scrapers.news_scraper import summarise_news
        print(f"\n{Fore.YELLOW}{summarise_news(fighter_a)}")
        print(f"\n{Fore.YELLOW}{summarise_news(fighter_b)}")


def _print_prediction(result: dict):
    a = result["fighter_a"]
    b = result["fighter_b"]
    pa = result["prob_a_wins"] * 100
    pb = result["prob_b_wins"] * 100
    winner = result["predicted_winner"]
    conf = result["confidence"]

    conf_color = {
        "High": Fore.GREEN,
        "Medium": Fore.YELLOW,
        "Low": Fore.RED,
    }.get(conf, Fore.WHITE)

    bar_width = 40
    a_bars = int(pa / 100 * bar_width)
    b_bars = bar_width - a_bars

    print(f"\n{'='*60}")
    print(f"  {Fore.WHITE}{Style.BRIGHT}{a:<25} vs  {b}")
    print(f"{'='*60}")
    print(f"  {Fore.CYAN}{a:<25} {Fore.GREEN}{'█' * a_bars}{Fore.RED}{'█' * b_bars} {Fore.CYAN}{b}")
    print(f"  {'':25} {pa:>5.1f}%  {pb:>5.1f}%")
    print()
    print(f"  Predicted Winner : {Style.BRIGHT}{Fore.GREEN}{winner}")
    print(f"  Confidence       : {conf_color}{conf}")

    # Flags
    na = result.get("news_a", {})
    nb = result.get("news_b", {})
    flags = []
    if na.get("injury_flag"):
        flags.append(f"{Fore.RED}⚠  {a} has recent injury news")
    if nb.get("injury_flag"):
        flags.append(f"{Fore.RED}⚠  {b} has recent injury news")
    if na.get("weight_issue_flag"):
        flags.append(f"{Fore.YELLOW}⚠  {a} has weight cut concerns")
    if nb.get("weight_issue_flag"):
        flags.append(f"{Fore.YELLOW}⚠  {b} has weight cut concerns")

    if flags:
        print(f"\n  Risk Flags:")
        for f in flags:
            print(f"    {f}")

    # Key stat diffs
    feats = result.get("features", {})
    key_stats = [
        ("SLpM diff", "slpm_diff"),
        ("Str Acc diff", "str_acc_diff"),
        ("Str Def diff", "str_def_diff"),
        ("TD Avg diff", "td_avg_diff"),
        ("TD Def diff", "td_def_diff"),
        ("Reach diff", "reach_cm_diff"),
        ("Win streak diff", "win_streak_diff"),
        ("Age diff", "age_diff"),
    ]
    table = []
    for label, key in key_stats:
        val = feats.get(key)
        if val is not None:
            favor = f"→ {a}" if val > 0 else (f"→ {b}" if val < 0 else "Even")
            table.append([label, f"{val:+.2f}", favor])

    if table:
        print(f"\n  Key Stat Differentials (positive = favours {a}):")
        print(tabulate(table, headers=["Stat", "Diff", "Favours"], tablefmt="simple",
                       colalign=("left", "right", "left")))
    print(f"{'='*60}\n")


def cmd_news(args):
    if not args:
        print(f"{Fore.RED}Usage: python main.py news \"Fighter Name\"")
        sys.exit(1)
    from scrapers.news_scraper import summarise_news, build_news_features
    name = args[0]
    print(summarise_news(name))
    features = build_news_features(name)
    print(f"\nDerived features: {features}")


def cmd_info(args):
    if not args:
        print(f"{Fore.RED}Usage: python main.py info \"Fighter Name\"")
        sys.exit(1)
    if not os.path.exists(FIGHTERS_CSV):
        print(f"{Fore.RED}No fighter data found. Run: python main.py scrape")
        sys.exit(1)
    fighters_df = pd.read_csv(FIGHTERS_CSV)
    from scrapers.ufc_stats_scraper import search_fighter
    name = args[0]
    row = search_fighter(name, fighters_df)
    if row is None:
        print(f"{Fore.RED}Fighter '{name}' not found in cache.")
        sys.exit(1)
    print(f"\n{Style.BRIGHT}{row.get('name', name)}")
    display_cols = [
        "wins", "losses", "draws", "height_cm", "reach_cm", "stance", "age",
        "slpm", "str_acc", "sapm", "str_def",
        "td_avg", "td_acc", "td_def", "sub_avg",
        "win_streak", "loss_streak", "ko_win_rate", "sub_win_rate", "dec_win_rate",
    ]
    table = [(col, row.get(col, "N/A")) for col in display_cols if col in row.index]
    print(tabulate(table, headers=["Stat", "Value"], tablefmt="simple"))


COMMANDS = {
    "scrape": cmd_scrape,
    "train": cmd_train,
    "predict": cmd_predict,
    "news": cmd_news,
    "info": cmd_info,
}

HELP_TEXT = """
UFC Fighter Prediction Model
━━━━━━━━━━━━━━━━━━━━━━━━━━
Commands:
  scrape [limit]            Scrape UFC fighter profiles and fight history
  train                     Train the XGBoost prediction model
  predict "A" "B"           Predict outcome of fighter A vs fighter B
  news "Fighter Name"       Show recent news for a fighter
  info "Fighter Name"       Show cached stats for a fighter

Quick start:
  1. Copy .env.example to .env and add your NEWS_API_KEY (optional)
  2. python main.py scrape
  3. python main.py train
  4. python main.py predict "Jon Jones" "Ciryl Gane"
"""


def main():
    args = sys.argv[1:]
    if not args or args[0] in ("-h", "--help", "help"):
        print(HELP_TEXT)
        return

    cmd = args[0].lower()
    cmd_args = args[1:]

    if cmd not in COMMANDS:
        print(f"{Fore.RED}Unknown command: {cmd}")
        print(HELP_TEXT)
        sys.exit(1)

    COMMANDS[cmd](cmd_args)


if __name__ == "__main__":
    main()
