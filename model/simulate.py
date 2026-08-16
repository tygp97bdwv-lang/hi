"""
Round-by-round fight simulator.

Generates a stat-driven, play-by-play narrative of a simulated fight between
two fighters, using their cached stats to weight strike volume/accuracy,
takedown success, and finish probability each round. Purely for entertainment
on top of the win-probability model — it does not affect predict_matchup.
"""
import random
from typing import Optional

import numpy as np
import pandas as pd

MAX_ROUNDS = 3

STRIKE_VERBS = [
    "lands a sharp jab", "connects with a straight right", "cracks a leg kick",
    "finds a clean 1-2", "lands a body kick", "clips with an elbow in the clinch",
    "scores with a counter hook", "lands a spinning back kick", "tags with a jab-cross",
    "opens up with a combination",
]
MISS_VERBS = [
    "swings and misses", "gets checked on a leg kick", "eats a counter",
    "overcommits on a combo", "is stuffed at the fence", "whiffs on a spinning attack",
]
TD_VERBS = [
    "changes levels and secures a takedown", "drags the fight to the mat",
    "scores a trip takedown against the cage",
]
TD_DEF_VERBS = [
    "stuffs the takedown attempt", "sprawls and scrambles back to the feet",
    "defends well against the cage",
]
SUB_VERBS = [
    "hunts for a guillotine", "works for an arm-triangle from top position",
    "threatens a rear-naked choke", "locks in a tight kimura attempt",
]
FLAVOR = [
    "The crowd is on its feet.", "Both fighters exchange a nod of respect.",
    "The corner is shouting instructions.", "A tense feeling-out moment follows.",
    "The pace slows as both fighters reset.", "The energy in the arena is electric.",
]


def _safe(f: pd.Series, key: str, default: float) -> float:
    val = f.get(key) if hasattr(f, "get") else None
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return default
    return float(val)


def _round_score(f: pd.Series, opp: pd.Series) -> dict:
    """Weighted round-level stats used to pick simulated events."""
    return {
        "strike_rate": _safe(f, "slpm", 4.5),
        "str_acc": _safe(f, "str_acc", 0.48),
        "str_def_opp": _safe(opp, "str_def", 0.55),
        "td_avg": _safe(f, "td_avg", 1.5),
        "td_acc": _safe(f, "td_acc", 0.44),
        "td_def_opp": _safe(opp, "td_def", 0.62),
        "sub_avg": _safe(f, "sub_avg", 0.6),
        "ko_win_rate": _safe(f, "ko_win_rate", 0.3),
        "sub_win_rate": _safe(f, "sub_win_rate", 0.2),
    }


def _pick_finish(name_a: str, name_b: str, score_a: dict, score_b: dict, rnd: int) -> Optional[dict]:
    """Roll for a finish this round. Higher finish-rate fighters land it more often."""
    base = 0.05 + rnd * 0.03  # finishes get more likely as the fight wears on
    ko_chance_a = base * (0.5 + score_a["ko_win_rate"])
    ko_chance_b = base * (0.5 + score_b["ko_win_rate"])
    sub_chance_a = base * 0.6 * (0.5 + score_a["sub_win_rate"])
    sub_chance_b = base * 0.6 * (0.5 + score_b["sub_win_rate"])

    roll = random.random()
    cum = 0.0
    for chance, fighter, method in (
        (ko_chance_a, name_a, "KO/TKO"),
        (ko_chance_b, name_b, "KO/TKO"),
        (sub_chance_a, name_a, "Submission"),
        (sub_chance_b, name_b, "Submission"),
    ):
        cum += chance
        if roll < cum:
            return {"winner": fighter, "method": method}
    return None


def simulate_fight(
    fighter_a: pd.Series,
    fighter_b: pd.Series,
    prob_a_wins: float = 0.5,
    rounds: int = MAX_ROUNDS,
    seed: Optional[int] = None,
) -> dict:
    """
    Simulate a round-by-round fight and return a play-by-play log plus outcome.

    Returns:
        {
            "log": [{"round": int, "events": [str, ...]}, ...],
            "winner": str,
            "method": str,           # "KO/TKO" | "Submission" | "Decision"
            "finish_round": int | None,
            "round_scores": {round: {name_a: pts, name_b: pts}},
        }
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    name_a = fighter_a.get("name", "Fighter A")
    name_b = fighter_b.get("name", "Fighter B")
    score_a = _round_score(fighter_a, fighter_b)
    score_b = _round_score(fighter_b, fighter_a)

    log = []
    round_scores = {}
    finish = None

    for rnd in range(1, rounds + 1):
        events = [f"— Round {rnd} —"]
        pts_a = pts_b = 0

        n_exchanges = random.randint(4, 6)
        for _ in range(n_exchanges):
            actor, other, s_actor, s_other = (
                (name_a, name_b, score_a, score_b) if random.random() < 0.5
                else (name_b, name_a, score_b, score_a)
            )
            roll = random.random()

            landed_chance = min(0.85, s_actor["str_acc"] + (1 - s_other["str_def_opp"]) * 0.3)
            td_chance = min(0.4, s_actor["td_avg"] / 15 * (1 - s_other["td_def_opp"]))
            sub_chance = min(0.15, s_actor["sub_avg"] / 20)

            if roll < td_chance:
                if random.random() < 0.35:
                    events.append(f"{other} {random.choice(TD_DEF_VERBS)}.")
                    if actor == name_a:
                        pts_b += 1
                    else:
                        pts_a += 1
                else:
                    events.append(f"{actor} {random.choice(TD_VERBS)}.")
                    if actor == name_a:
                        pts_a += 2
                    else:
                        pts_b += 2
            elif roll < td_chance + sub_chance:
                events.append(f"{actor} {random.choice(SUB_VERBS)}!")
                if actor == name_a:
                    pts_a += 1
                else:
                    pts_b += 1
            elif roll < td_chance + sub_chance + landed_chance:
                events.append(f"{actor} {random.choice(STRIKE_VERBS)}.")
                if actor == name_a:
                    pts_a += 1
                else:
                    pts_b += 1
            else:
                events.append(f"{actor} {random.choice(MISS_VERBS)}.")

            if random.random() < 0.15:
                events.append(random.choice(FLAVOR))

        round_scores[rnd] = {name_a: pts_a, name_b: pts_b}

        finish = _pick_finish(name_a, name_b, score_a, score_b, rnd)
        if finish:
            events.append(f"🏆 {finish['winner']} finishes the fight by {finish['method']} in round {rnd}!")
            log.append({"round": rnd, "events": events})
            break

        log.append({"round": rnd, "events": events})

    if finish:
        winner, method, finish_round = finish["winner"], finish["method"], log[-1]["round"]
    else:
        total_a = sum(r[name_a] for r in round_scores.values())
        total_b = sum(r[name_b] for r in round_scores.values())
        # Blend simulated round scoring with the model's win probability so
        # the decision is consistent with the predicted favourite.
        total_a += prob_a_wins * 3
        total_b += (1 - prob_a_wins) * 3
        winner = name_a if total_a >= total_b else name_b
        method = "Decision"
        finish_round = None

    return {
        "log": log,
        "winner": winner,
        "method": method,
        "finish_round": finish_round,
        "round_scores": round_scores,
    }
