
from __future__ import annotations
import math
import numpy as np
import random
from typing import Dict, List, Tuple, Any


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def estimate_theta_eap(
    responses: List[Dict[str, Any]],
    theta_min: float = -4.0,
    theta_max: float = 6.0,
    theta_step: float = 0.05,
    prior_mean: float = 0.0,
    prior_sd: float = 1.5,
) -> Tuple[float, float]:
    """EAP estimate for a Rasch-like model.

    Item b-values in this prototype are expert-estimated, not empirically calibrated.
    The output is therefore IRT-like and useful for prototyping, but not a validated
    psychometric score.
    """
    if not responses:
        return prior_mean, prior_sd

    grid = np.arange(theta_min, theta_max + theta_step, theta_step)
    log_prior = -0.5 * ((grid - prior_mean) / prior_sd) ** 2
    log_likelihood = np.zeros_like(grid, dtype=float)

    for r in responses:
        b = float(r["b_value"])
        y = int(r["correct"])
        p = 1.0 / (1.0 + np.exp(-(grid - b)))
        p = np.clip(p, 1e-9, 1 - 1e-9)
        log_likelihood += y * np.log(p) + (1 - y) * np.log(1 - p)

    log_post = log_prior + log_likelihood
    log_post -= np.max(log_post)
    post = np.exp(log_post)
    post /= np.sum(post)

    theta = float(np.sum(grid * post))
    var = float(np.sum(((grid - theta) ** 2) * post))
    se = math.sqrt(max(var, 1e-9))
    return theta, se


def select_next_item(
    items: List[Dict[str, Any]],
    administered_ids: List[str],
    theta: float,
    precision_mode: bool = False,
) -> Dict[str, Any] | None:
    """Select a near-optimal not-yet-administered item with stochastic variation.

    The previous prototype always selected the single closest b-value, which made
    repeated test runs feel deterministic. This version draws randomly from the
    closest candidates, while still keeping item difficulty near the current theta.
    """
    administered = set(administered_ids)
    remaining = [it for it in items if it["item_id"] not in administered]
    if not remaining:
        return None

    def distance(it: Dict[str, Any]) -> float:
        return abs(float(it["b_value"]) - theta)

    ranked = sorted(remaining, key=lambda it: (distance(it), it["item_id"]))
    # Keep psychometric targeting, but add variation. In precision mode be stricter.
    top_k = 4 if precision_mode else 7
    max_window = 0.45 if precision_mode else 0.85
    best_dist = distance(ranked[0])
    pool = [it for it in ranked[:top_k] if distance(it) <= best_dist + max_window]
    if not pool:
        pool = ranked[:1]

    # Weight closer items higher, but do not make the choice deterministic.
    weights = [1.0 / (0.15 + distance(it)) for it in pool]
    return random.choices(pool, weights=weights, k=1)[0]


def select_routing_items(items: List[Dict[str, Any]], n: int = 10) -> List[Dict[str, Any]]:
    """Random stratified routing set across the current itembank.

    The bank is sorted by expert-estimated difficulty and split into n strata.
    One item is sampled per stratum, then the order is shuffled. This keeps the
    routing phase difficulty-balanced but no longer identical across test runs.
    """
    sorted_items = sorted(items, key=lambda it: float(it["b_value"]))
    if n >= len(sorted_items):
        out = list(sorted_items)
        random.shuffle(out)
        return out

    strata = np.array_split(sorted_items, n)
    selected = []
    used_ids = set()
    for group in strata:
        candidates = [dict(x) for x in group.tolist() if x["item_id"] not in used_ids]
        if candidates:
            chosen = random.choice(candidates)
            selected.append(chosen)
            used_ids.add(chosen["item_id"])

    # In case of empty strata or duplicates, fill from unused items.
    if len(selected) < n:
        remaining = [it for it in sorted_items if it["item_id"] not in used_ids]
        random.shuffle(remaining)
        selected.extend(remaining[: n - len(selected)])

    random.shuffle(selected)
    return selected[:n]


def phase1_should_stop(
    responses: List[Dict[str, Any]],
    theta_history: List[float],
    routing_items: int,
    min_adaptive_items: int,
    stability_window: int,
    theta_stability_range: float,
    max_items: int,
) -> Tuple[bool, str]:
    n = len(responses)
    if n >= max_items:
        return True, "max_items_reached"

    adaptive_n = max(0, n - routing_items)
    if adaptive_n < min_adaptive_items:
        return False, "min_adaptive_items_not_reached"

    if len(theta_history) >= stability_window:
        window = theta_history[-stability_window:]
        if max(window) - min(window) <= theta_stability_range:
            return True, "theta_stable"

    return False, "continue"


def gate_decision(theta: float, se: float, precision_items_used: int, gate_cfg: Dict[str, Any]) -> str:
    if theta >= gate_cfg["phase_2_gate_theta"] and se <= gate_cfg["phase_2_gate_se"]:
        return "open"
    if theta >= gate_cfg["phase_2_gate_theta"] and se > gate_cfg["phase_2_gate_se"] and precision_items_used < gate_cfg["max_precision_items"]:
        return "precision_block"
    return "closed"


def summarize_dimension_performance(responses: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for r in responses:
        dim = r["dimension"]
        out.setdefault(dim, {"n": 0, "correct": 0, "mean_b": 0.0})
        out[dim]["n"] += 1
        out[dim]["correct"] += int(r["correct"])
        out[dim]["mean_b"] += float(r["b_value"])
    for dim, vals in out.items():
        vals["accuracy"] = vals["correct"] / vals["n"] if vals["n"] else None
        vals["mean_b"] = vals["mean_b"] / vals["n"] if vals["n"] else None
    return out
