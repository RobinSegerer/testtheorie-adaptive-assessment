from __future__ import annotations

import json
import os
import re
import random
import copy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

from src.scoring import (
    estimate_theta_eap,
    gate_decision,
    phase1_should_stop,
    select_next_item,
    select_routing_items,
    summarize_dimension_performance,
)
from src.item_generator import generate_phase1_item_with_openai
from src.openai_judge import score_with_openai

ROOT = Path(__file__).parent
DATA = ROOT / "data"
PROMPTS = ROOT / "prompts"
RESULTS = ROOT / "results"

MODES = [
    "Volltest: Routing → Phase 1 → Phase 2",
    "Nur Phase 1 (mit Online-Items)",
    "Nur Phase 2 (mit Online-Items)",
]

PHASE1_CFG = {
    "routing_items": 10,
    "min_adaptive_items": 12,
    "stability_window": 8,
    "theta_stability_range": 0.35,
    "max_items": 45,
    "theta_grid_min": -4.0,
    "theta_grid_max": 6.0,
    "theta_grid_step": 0.05,
    "prior_mean": 0.0,
    "prior_sd": 1.5,
}
GATE_CFG = {
    "phase_2_gate_theta": 3.0,
    "phase_2_gate_se": 0.35,
    "max_precision_items": 3,
}
PHASE2_CFG = {"max_cases": 3}
OPENAI_MODEL = "gpt-4.1-mini"

st.set_page_config(page_title="Testtheorie-Kompetenztest", layout="centered")


@st.cache_data
def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_items() -> List[Dict[str, Any]]:
    return load_json(DATA / "itembank_phase_1.json")["itembank_phase_1"]


@st.cache_data
def load_cases() -> List[Dict[str, Any]]:
    return load_json(DATA / "phase2_cases.json")["phase2_cases"]


@st.cache_data
def load_prompt(filename: str) -> str:
    return (PROMPTS / filename).read_text(encoding="utf-8")


def get_api_key() -> Optional[str]:
    try:
        key = st.secrets.get("OPENAI_API_KEY", None)
        if key:
            return str(key)
    except Exception:
        pass
    key = os.getenv("OPENAI_API_KEY")
    return key or None


def append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def reset_state(mode: Optional[str] = None) -> None:
    keep_mode = mode if mode is not None else st.session_state.get("mode", MODES[0])
    keep_p1_online = st.session_state.get("full_phase1_online", False)
    keep_p2_online = st.session_state.get("full_phase2_online", False)
    keep_phase1_only_online = st.session_state.get("phase1_only_online", True)
    keep_phase2_only_online = st.session_state.get("phase2_only_online", True)
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    st.session_state.mode = keep_mode
    st.session_state.full_phase1_online = keep_p1_online
    st.session_state.full_phase2_online = keep_p2_online
    st.session_state.phase1_only_online = keep_phase1_only_online
    st.session_state.phase2_only_online = keep_phase2_only_online
    st.session_state.stage = "phase1" if keep_mode != "Nur Phase 2 (mit Online-Items)" else "phase2"

    st.session_state.responses = []
    st.session_state.theta = 0.0
    st.session_state.se = 1.5
    st.session_state.theta_history = []
    st.session_state.administered_ids = []
    st.session_state.current_item = None
    st.session_state.last_phase1_feedback = None
    st.session_state.phase1_complete = False
    st.session_state.stop_reason = None
    st.session_state.routing_queue = select_routing_items(load_items(), PHASE1_CFG["routing_items"])
    st.session_state.precision_items_used = 0
    st.session_state.online_failures = 0

    st.session_state.phase2_case_index = 0
    st.session_state.current_case = None
    st.session_state.used_case_ids = []
    st.session_state.phase2_scores = []
    st.session_state.phase2_answer_text = ""
    st.session_state.last_phase2_feedback = None
    st.session_state.test_complete = False


def ensure_state() -> None:
    if "mode" not in st.session_state:
        reset_state(MODES[0])


def estimate() -> None:
    theta, se = estimate_theta_eap(
        st.session_state.responses,
        theta_min=PHASE1_CFG["theta_grid_min"],
        theta_max=PHASE1_CFG["theta_grid_max"],
        theta_step=PHASE1_CFG["theta_grid_step"],
        prior_mean=PHASE1_CFG["prior_mean"],
        prior_sd=PHASE1_CFG["prior_sd"],
    )
    st.session_state.theta = theta
    st.session_state.se = se
    st.session_state.theta_history.append(theta)


def phase1_routing_count_for_mode(mode: str) -> int:
    # In beiden Phase-1-Modi beginnt der Test mit einer festen, aber zufällig
    # stratifizierten Routingphase. Danach wird adaptiv weitergemessen.
    if mode in {"Volltest: Routing → Phase 1 → Phase 2", "Nur Phase 1 (mit Online-Items)"}:
        return PHASE1_CFG["routing_items"]
    return 0


def phase1_uses_online(mode: str) -> bool:
    # Online-Items werden nie im Routing verwendet, sondern erst nach der
    # Routingphase im adaptiven Teil.
    if mode == "Volltest: Routing → Phase 1 → Phase 2":
        return bool(st.session_state.get("full_phase1_online", False))
    if mode == "Nur Phase 1 (mit Online-Items)":
        return bool(st.session_state.get("phase1_only_online", True))
    return False


def phase1_online_only(mode: str) -> bool:
    return False


def phase2_uses_online(mode: str) -> bool:
    if mode == "Volltest: Routing → Phase 1 → Phase 2":
        return bool(st.session_state.get("full_phase2_online", False))
    if mode == "Nur Phase 2 (mit Online-Items)":
        return bool(st.session_state.get("phase2_only_online", True))
    return False


def generate_online_item(target_b: float) -> Optional[Dict[str, Any]]:
    api_key = get_api_key()
    if not api_key:
        st.error("Kein OpenAI-Key im Hintergrund gefunden.")
        return None
    system_prompt = load_prompt("phase1_item_generator_system_prompt.txt")
    last_error: Optional[Exception] = None
    for _ in range(3):
        try:
            return generate_phase1_item_with_openai(
                system_prompt=system_prompt,
                target_b=target_b,
                target_dimension=None,
                administered_items=st.session_state.responses,
                api_key=api_key,
                model=OPENAI_MODEL,
                quarantine_path=RESULTS / "generated_items_quarantine.jsonl",
                failure_log_path=RESULTS / "generation_failures.jsonl",
            )
        except Exception as exc:
            last_error = exc
            append_jsonl(RESULTS / "generation_failures.jsonl", {
                "event": "phase1_generation_attempt_failed",
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "error": str(exc),
                "target_b": target_b,
            })
    st.session_state.online_failures += 1
    st.error(f"Item konnte nicht generiert werden: {last_error}")
    return None


def get_next_phase1_item() -> Optional[Dict[str, Any]]:
    mode = st.session_state.mode
    n_done = len(st.session_state.responses)
    routing_count = phase1_routing_count_for_mode(mode)

    # Feste Routingphase: immer aus JSON, außer im Online-only-Modus.
    if routing_count and n_done < routing_count:
        return st.session_state.routing_queue[n_done]

    # Online-only: nie JSON nutzen.
    if phase1_online_only(mode):
        return generate_online_item(st.session_state.theta)

    # Volltest mit Online-Phase-1: nach Routing Online versuchen, bei Fehler Bankitem.
    if phase1_uses_online(mode):
        item = generate_online_item(st.session_state.theta)
        if item is not None:
            return item
        if st.session_state.online_failures >= 2:
            st.warning("Online-Generierung mehrfach fehlgeschlagen; Fallback auf Itembank.")

    # Standard: feste JSON-Itembank adaptiv.
    return select_next_item(load_items(), st.session_state.administered_ids, st.session_state.theta)


def balance_option_order(item: Dict[str, Any], item_index: int) -> Dict[str, Any]:
    """Return a copy of the item with the correct option rotated across A-D.

    This prevents answer-position artifacts in Phase 1 while preserving the
    item text, options, rationale and scoring logic. The order is fixed once
    the item is loaded into session_state.
    """
    if item.get("option_order_balanced"):
        return item
    item = copy.deepcopy(item)
    options = item.get("options", {})
    old_correct_key = item.get("correct_key")
    if old_correct_key not in options or len(options) != 4:
        return item

    target_correct_key = ["A", "B", "C", "D"][item_index % 4]
    correct_text = options[old_correct_key]
    distractor_texts = [v for k, v in options.items() if k != old_correct_key]

    rng = random.Random(f"{item.get('item_id', 'item')}-{item_index}-option-balance")
    rng.shuffle(distractor_texts)

    new_options: Dict[str, str] = {}
    distractor_iter = iter(distractor_texts)
    for key in ["A", "B", "C", "D"]:
        if key == target_correct_key:
            new_options[key] = correct_text
        else:
            new_options[key] = next(distractor_iter)

    item["options"] = new_options
    item["correct_key"] = target_correct_key
    item["original_correct_key"] = old_correct_key
    item["option_order_balanced"] = True
    return item


def maybe_load_phase1_item() -> None:
    if st.session_state.phase1_complete or st.session_state.current_item is not None:
        return
    item = get_next_phase1_item()
    if item is None:
        st.session_state.phase1_complete = True
        st.session_state.stop_reason = "no_item_available"
    else:
        st.session_state.current_item = balance_option_order(item, len(st.session_state.responses))


def submit_phase1_answer(answer_key: str) -> None:
    item = st.session_state.current_item
    if not item:
        return
    correct = int(answer_key == item["correct_key"])
    response = {
        "item_id": item["item_id"],
        "dimension": item["dimension"],
        "b_value": float(item["b_value"]),
        "correct": correct,
        "answer_key": answer_key,
        "correct_key": item["correct_key"],
        "generated_online": bool(item.get("generated_online", False)),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    st.session_state.responses.append(response)
    st.session_state.administered_ids.append(item["item_id"])
    st.session_state.current_item = None
    estimate()

    st.session_state.last_phase1_feedback = {
        "is_correct": bool(correct),
        "selected": answer_key,
        "correct_key": item["correct_key"],
        "correct_text": item["options"].get(item["correct_key"], ""),
        "rationale": item.get("rationale", ""),
        "theta": st.session_state.theta,
        "se": st.session_state.se,
    }

    mode = st.session_state.mode
    decision = gate_decision(st.session_state.theta, st.session_state.se, st.session_state.precision_items_used, GATE_CFG)

    # Im Volltest springt Phase 2 sobald das Gate offen ist.
    if mode == "Volltest: Routing → Phase 1 → Phase 2" and decision == "open":
        st.session_state.phase1_complete = True
        st.session_state.stop_reason = "phase2_gate_open"
        st.session_state.stage = "phase2"
        return

    routing_count = phase1_routing_count_for_mode(mode)
    should_stop, reason = phase1_should_stop(
        responses=st.session_state.responses,
        theta_history=st.session_state.theta_history,
        routing_items=routing_count,
        min_adaptive_items=PHASE1_CFG["min_adaptive_items"],
        stability_window=PHASE1_CFG["stability_window"],
        theta_stability_range=PHASE1_CFG["theta_stability_range"],
        max_items=PHASE1_CFG["max_items"],
    )
    if should_stop:
        st.session_state.phase1_complete = True
        st.session_state.stop_reason = reason
        if mode == "Volltest: Routing → Phase 1 → Phase 2" and decision in {"open", "precision_block"}:
            st.session_state.stage = "phase2"


def render_sidebar_explainer() -> None:
    st.markdown("### Kurz erklärt")
    st.caption(
        "**Phase 1**: kurze MC-/Single-best-answer-Items. Der Test schätzt nach jeder Antwort θ "
        "(Fähigkeit) und SE (Unsicherheit). Antworten werden sofort als richtig/falsch mit Begründung zurückgemeldet."
    )
    st.caption(
        "**Routing**: Startblock mit zufällig gezogenen Items aus verschiedenen Schwierigkeitsbereichen. "
        "Das stabilisiert die erste θ-Schätzung und verhindert immer gleiche Startitems."
    )
    st.caption(
        "**Adaptivität**: Nach dem Routing werden Items nahe der aktuellen θ-Schätzung ausgewählt. "
        "Wenn Online-Items aktiv sind, werden diese Items passend zu θ generiert; sonst kommen sie aus der Itembank."
    )
    st.caption(
        "**Sprung zu Phase 2**: Im Volltest wird nach jeder Phase-1-Antwort geprüft, ob das Gate offen ist. "
        "Aktuell: θ ≥ 3.00 und SE ≤ 0.35. Dann startet das Expert:innenmodul."
    )
    st.caption(
        "**Phase 2**: offene Fallstudien mit rubriziertem Scoring. Online aktiv bedeutet: Fallstudien werden neu generiert; "
        "das Scoring nutzt ebenfalls den hinterlegten OpenAI-Key."
    )


def render_phase1_status() -> None:
    n = len(st.session_state.responses)
    routing_count = phase1_routing_count_for_mode(st.session_state.mode)
    adaptive_n = max(0, n - routing_count)
    decision = gate_decision(st.session_state.theta, st.session_state.se, st.session_state.precision_items_used, GATE_CFG)
    status = {
        "θ": f"{st.session_state.theta:.2f}",
        "SE": f"{st.session_state.se:.2f}",
        "Items": n,
        "Routing": f"{min(n, routing_count)}/{routing_count}",
        "Adaptiv": adaptive_n,
        "Phase-2-Gate": decision,
    }
    st.dataframe(pd.DataFrame([status]), hide_index=True, use_container_width=True)


def render_last_phase1_feedback() -> None:
    fb = st.session_state.get("last_phase1_feedback")
    if not fb:
        return
    if fb["is_correct"]:
        st.success(f"Richtig. Neue Schätzung: θ = {fb['theta']:.2f}, SE = {fb['se']:.2f}")
    else:
        st.error(f"Falsch. Richtig wäre {fb['correct_key']}: {fb['correct_text']}. Neue Schätzung: θ = {fb['theta']:.2f}, SE = {fb['se']:.2f}")
    if fb.get("rationale"):
        st.info(f"Begründung: {fb['rationale']}")


def render_phase1() -> None:
    st.subheader("Phase 1: adaptives Screening")
    render_phase1_status()
    render_last_phase1_feedback()

    if st.session_state.phase1_complete:
        st.success(f"Phase 1 beendet: {st.session_state.stop_reason}")
        if st.session_state.responses:
            st.write("Kurzprofil")
            st.dataframe(pd.DataFrame(summarize_dimension_performance(st.session_state.responses)).T, use_container_width=True)
        if st.session_state.mode == "Volltest: Routing → Phase 1 → Phase 2":
            if st.button("Phase 2 starten", type="primary"):
                st.session_state.stage = "phase2"
                st.rerun()
        return

    maybe_load_phase1_item()
    item = st.session_state.current_item
    if item is None:
        st.warning("Kein weiteres Item verfügbar.")
        return

    source = "online generiert" if item.get("generated_online") else "Itembank"
    routing_count = phase1_routing_count_for_mode(st.session_state.mode)
    phase_label = "Routing" if len(st.session_state.responses) < routing_count else "Adaptiv"
    st.markdown(f"**{phase_label}-Item** · b ≈ {float(item['b_value']):.2f} · {source}")
    st.write(item["stem"])
    choice = st.radio(
        "Antwort",
        list(item["options"].keys()),
        format_func=lambda k: f"{k}: {item['options'][k]}",
        index=None,
        key=f"ans_{item['item_id']}_{len(st.session_state.responses)}",
    )
    if choice is None:
        st.caption("Bitte eine Antwort auswählen.")

    if st.button("Antwort senden", type="primary", disabled=(choice is None)):
        submit_phase1_answer(choice)
        st.rerun()


def extract_json(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not m:
            raise
        return json.loads(m.group(0))


def looks_english(text: str) -> bool:
    if not isinstance(text, str):
        return False
    lower = " " + text.lower() + " "
    markers = [" the ", " and ", " because ", " therefore ", " which ", " this ", " should ", " validity ", " reliability ", " test score "]
    return sum(m in lower for m in markers) >= 2


def validate_generated_case(case: Dict[str, Any]) -> Dict[str, Any]:
    if "case_id" not in case:
        case["case_id"] = "P2_GEN_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    case.setdefault("title", "Online generierte Fallstudie")
    case.setdefault("target_level", "expert")
    dims = case.get("rubric_dimensions", [])
    if not isinstance(dims, list) or len(dims) < 3:
        raise ValueError("Generated case must contain at least 3 rubric_dimensions.")
    for d in dims:
        if "dimension_id" not in d or "label" not in d:
            raise ValueError("Rubric dimension missing dimension_id or label.")
        if set(d.get("score_anchors", {}).keys()) != {"0", "1", "2", "3"}:
            raise ValueError("Rubric dimension must include score_anchors 0-3.")
        if set(d.get("few_shot_anchors", {}).keys()) != {"0", "1", "2", "3"}:
            raise ValueError("Rubric dimension must include few_shot_anchors 0-3.")
    # Language guard: Fallstudie und Rubriken müssen deutsch sein.
    texts = [case.get("title", ""), case.get("case_text", "")]
    for d in dims:
        texts.append(d.get("label", ""))
        texts.extend(str(v) for v in d.get("score_anchors", {}).values())
        texts.extend(str(v) for v in d.get("few_shot_anchors", {}).values())
    if any(looks_english(t) for t in texts):
        raise ValueError("Generated case failed validation: language_not_german")
    case["generated_online"] = True
    return case


def phase2_current_level_hint() -> str:
    if not st.session_state.phase2_scores:
        return "expert"
    totals = []
    for result in st.session_state.phase2_scores:
        summ = result.get("score_summary", {})
        if summ.get("max_score"):
            totals.append(summ.get("total_score", 0) / summ.get("max_score"))
    mean_score = sum(totals) / len(totals) if totals else 0.0
    return "reviewer_like" if mean_score >= 0.75 else "expert"


def generate_online_case() -> Optional[Dict[str, Any]]:
    api_key = get_api_key()
    if not api_key:
        st.error("Kein OpenAI-Key im Hintergrund gefunden.")
        return None
    try:
        from openai import OpenAI
    except Exception:
        st.error("OpenAI-Paket fehlt. Bitte requirements installieren.")
        return None

    target_level = phase2_current_level_hint()
    system = (
        "Du bist ein strenger Psychometrie-Dozent. Erzeuge genau eine schwierige offene Fallstudie "
        "zur psychologischen Testtheorie/Testentwicklung. Gib nur ein valides JSON-Objekt zurück. "
        "Die Aufgabe soll Expert:innen-/Reviewer-Kompetenz prüfen, nicht bloß Faktenwissen. "
        "WICHTIG: Schreibe die gesamte Fallstudie, alle Rubrikdimensionen, alle Score-Anker und alle Few-Shot-Anker ausschließlich auf Deutsch. "
        "Englische Fachbegriffe sind nur als etablierte Abkürzungen/Termini erlaubt, z. B. IRT, CAT, DIF oder True Score/wahrer Wert. "
        "Keine englischen Sätze, keine englischen Aufgabenstellungen, keine englischen Rubriktexte."
    )
    user = {
        "target_level": target_level,
        "required_schema": {
            "case_id": "P2_GEN_unique_id",
            "title": "string",
            "target_level": "expert_or_reviewer_like",
            "case_text": "string",
            "rubric_dimensions": [
                {
                    "dimension_id": "short_snake_case",
                    "label": "string",
                    "score_anchors": {"0": "...", "1": "...", "2": "...", "3": "..."},
                    "few_shot_anchors": {"0": "...", "1": "...", "2": "...", "3": "..."},
                }
            ],
        },
        "constraints": [
            "mindestens 3 Rubrikdimensionen",
            "jede Dimension hat Score-Anker und Few-Shot-Anker für 0,1,2,3",
            "Themen: Validität, Reliabilität, IRT/CAT, Fairness/DIF, lokale Abhängigkeit, Normierung oder Testentwicklung",
            "keine Lösung in die Fallstudie schreiben",
            "alle Felder ausschließlich auf Deutsch formulieren",
            "keine englischen Sätze oder Antwortanker verwenden",
        ],
    }
    try:
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": json.dumps(user, ensure_ascii=False)}],
            temperature=0.25,
            response_format={"type": "json_object"},
        )
        text = resp.choices[0].message.content or "{}"
        case = validate_generated_case(extract_json(text))
        append_jsonl(RESULTS / "generated_phase2_cases.jsonl", {"created_at": datetime.now().isoformat(timespec="seconds"), "case": case})
        return case
    except Exception as exc:
        append_jsonl(RESULTS / "phase2_generation_failures.jsonl", {"created_at": datetime.now().isoformat(timespec="seconds"), "error": str(exc)})
        st.error(f"Fallstudie konnte nicht generiert werden: {exc}")
        return None


def select_fixed_phase2_case() -> Optional[Dict[str, Any]]:
    cases = load_cases()
    unused = [c for c in cases if c.get("case_id") not in st.session_state.used_case_ids]
    if not unused:
        return None
    level = phase2_current_level_hint()
    if level == "reviewer_like":
        reviewer = [c for c in unused if c.get("target_level") == "reviewer_like"]
        if reviewer:
            return reviewer[0]
    return unused[0]


def load_current_phase2_case() -> None:
    if st.session_state.current_case is not None:
        return
    if len(st.session_state.phase2_scores) >= PHASE2_CFG["max_cases"]:
        st.session_state.test_complete = True
        return
    if phase2_uses_online(st.session_state.mode):
        st.session_state.current_case = generate_online_case()
    else:
        st.session_state.current_case = select_fixed_phase2_case()
    if st.session_state.current_case is None:
        st.session_state.test_complete = True


def score_phase2_answer(case: Dict[str, Any], answer: str) -> Optional[Dict[str, Any]]:
    api_key = get_api_key()
    if not api_key:
        st.error("Kein OpenAI-Key im Hintergrund gefunden. Phase-2-Scoring ist ohne Key nicht verfügbar.")
        return None
    try:
        result = score_with_openai(
            case=case,
            candidate_answer=answer,
            system_prompt=load_prompt("phase2_judge_system_prompt.txt"),
            api_key=api_key,
            model=OPENAI_MODEL,
        )
        append_jsonl(RESULTS / "phase2_scores.jsonl", {"created_at": datetime.now().isoformat(timespec="seconds"), "result": result})
        return result
    except Exception as exc:
        st.error(f"Scoring fehlgeschlagen: {exc}")
        return None


def render_last_phase2_feedback() -> None:
    fb = st.session_state.get("last_phase2_feedback")
    if not fb:
        return
    summary = fb.get("score_summary", {})
    if summary:
        st.success(f"Phase-2-Score: {summary.get('total_score')}/{summary.get('max_score')} · Profil: {summary.get('profile_level')}")
    scoring = fb.get("scoring_results", {})
    if scoring:
        rows = []
        for dim, res in scoring.items():
            rows.append({"Dimension": dim, "Score": res.get("score"), "Begründung": res.get("rationale", "")})
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


def render_phase2() -> None:
    st.subheader("Phase 2: rubriziertes Expert:innenmodul")
    st.caption(f"Fall {len(st.session_state.phase2_scores) + 1} von maximal {PHASE2_CFG['max_cases']} · Quelle: {'online' if phase2_uses_online(st.session_state.mode) else 'feste Fallstudien'}")
    render_last_phase2_feedback()

    if st.session_state.test_complete:
        st.success("Phase 2 beendet.")
        if st.session_state.phase2_scores:
            rows = []
            for i, result in enumerate(st.session_state.phase2_scores, 1):
                summ = result.get("score_summary", {})
                rows.append({"Fall": i, "Score": summ.get("total_score"), "Max": summ.get("max_score"), "Profil": summ.get("profile_level")})
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        return

    load_current_phase2_case()
    case = st.session_state.current_case
    if case is None:
        st.warning("Keine Fallstudie verfügbar.")
        return

    source = "online generiert" if case.get("generated_online") else "feste Fallstudie"
    st.markdown(f"**{case.get('title', 'Fallstudie')}** · {source}")
    st.write(case["case_text"])

    # The text area key includes the current case index and case_id so that the
    # previous answer is not carried over when a new Phase-2 case is loaded.
    answer_key = f"phase2_answer_{len(st.session_state.phase2_scores)}_{case.get('case_id', 'case')}"
    answer = st.text_area("Antwort", height=220, key=answer_key)

    if st.button("Antwort bewerten", type="primary", disabled=not bool(answer.strip())):
        result = score_phase2_answer(case, answer)
        if result:
            st.session_state.phase2_scores.append(result)
            st.session_state.used_case_ids.append(case.get("case_id"))
            st.session_state.last_phase2_feedback = result
            st.session_state.current_case = None
            st.session_state.phase2_answer_text = ""
            if len(st.session_state.phase2_scores) >= PHASE2_CFG["max_cases"]:
                st.session_state.test_complete = True
            st.rerun()


def main() -> None:
    ensure_state()
    st.title("Testtheorie-Kompetenztest")
    st.caption("Konzept und Prototyp: Dr. Robin Segerer")

    with st.sidebar:
        st.markdown("## Menü")
        mode = st.radio("Testmodus", MODES, index=MODES.index(st.session_state.mode))
        if mode != st.session_state.mode:
            reset_state(mode)
            st.rerun()

        if st.session_state.mode == "Volltest: Routing → Phase 1 → Phase 2":
            st.caption("Fester Startblock → adaptive Phase 1 → automatischer Sprung zu Phase 2, sobald das Gate offen ist.")
            st.session_state.full_phase1_online = st.checkbox(
                "Phase 1 nach Routing online generieren",
                value=st.session_state.get("full_phase1_online", False),
                help="Aus: adaptive Items kommen aus der Itembank. An: nach dem Routing werden passende Items online generiert; bei Fehlern greift die Itembank als Fallback.",
            )
            st.session_state.full_phase2_online = st.checkbox(
                "Phase 2 online generieren",
                value=st.session_state.get("full_phase2_online", False),
                help="Aus: feste Fallstudien. An: Fallstudien werden online generiert.",
            )
        elif st.session_state.mode == "Nur Phase 1 (mit Online-Items)":
            st.caption("Routing → adaptive Phase 1. Kein Sprung zu Phase 2.")
            st.session_state.phase1_only_online = st.checkbox(
                "Nach Routing online generieren",
                value=st.session_state.get("phase1_only_online", True),
                help="Aus: adaptive Items kommen aus der Itembank. An: nach dem Routing werden passende Items online generiert.",
            )
        elif st.session_state.mode == "Nur Phase 2 (mit Online-Items)":
            st.caption("Direkt offene Fallstudien. Phase 1 wird übersprungen.")
            st.session_state.phase2_only_online = st.checkbox(
                "Fallstudien online generieren",
                value=st.session_state.get("phase2_only_online", True),
                help="Aus: feste Fallstudien. An: Fallstudien werden online generiert.",
            )

        if st.button("Neu starten"):
            reset_state(st.session_state.mode)
            st.rerun()

        st.divider()
        st.caption("OpenAI-Key: " + ("gefunden" if get_api_key() else "nicht gefunden"))
        st.divider()
        render_sidebar_explainer()

    if st.session_state.stage == "phase1":
        render_phase1()
    else:
        render_phase2()


if __name__ == "__main__":
    main()

st.sidebar.markdown("---")
st.sidebar.caption("© Dr. Robin Segerer")
st.sidebar.caption("Prototyp · nicht empirisch kalibriert")
