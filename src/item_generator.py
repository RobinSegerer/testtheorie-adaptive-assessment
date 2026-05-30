from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _extract_json(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def _looks_english(text: str) -> bool:
    """Lightweight language guard: reject obvious English generations, not technical abbreviations."""
    if not isinstance(text, str):
        return False
    lower = " " + text.lower() + " "
    english_markers = [
        " the ", " and ", " because ", " therefore ", " which ", " what ", " why ",
        " validity ", " reliability ", " item difficulty ", " test score ",
        " this test ", " participants ", " students ", " should be ", " is the ",
    ]
    return sum(marker in lower for marker in english_markers) >= 2


def validate_generated_item(item: Dict[str, Any]) -> Tuple[bool, List[str]]:
    errors: List[str] = []
    required = [
        "dimension", "source_anchor", "b_value", "b_basis", "b_confidence",
        "cognitive_process", "item_type", "stem", "options", "correct_key",
        "rationale", "calibration_status"
    ]
    for key in required:
        if key not in item:
            errors.append(f"missing:{key}")

    options = item.get("options")
    if not isinstance(options, dict) or set(options.keys()) != {"A", "B", "C", "D"}:
        errors.append("options_must_be_A_B_C_D")
    else:
        for k, v in options.items():
            if not isinstance(v, str) or len(v.strip()) < 2:
                errors.append(f"option_{k}_empty")

    if item.get("correct_key") not in {"A", "B", "C", "D"}:
        errors.append("correct_key_invalid")

    if item.get("item_type") not in {"multiple_choice", "single_best_answer", "interpretation_item"}:
        errors.append("item_type_invalid")

    # Guard against a common LLM artifact: the correct option is uniquely the
    # longest and most elaborated. This is especially problematic in routing.
    if isinstance(options, dict) and item.get("correct_key") in {"A", "B", "C", "D"}:
        lens = {k: len(str(v).strip()) for k, v in options.items()}
        ck = item.get("correct_key")
        correct_len = lens.get(ck, 0)
        other_lens = [v for k, v in lens.items() if k != ck]
        if other_lens and correct_len == max(lens.values()):
            if correct_len - max(other_lens) > 28 or correct_len > 1.45 * max(1, sum(other_lens) / len(other_lens)):
                errors.append("correct_option_length_cue")

    try:
        b = float(item.get("b_value"))
        if not -6 <= b <= 8:
            errors.append("b_value_out_of_safety_range")
    except Exception:
        errors.append("b_value_not_numeric")

    if not isinstance(item.get("stem"), str) or len(item.get("stem", "").strip()) < 30:
        errors.append("stem_too_short")

    if not isinstance(item.get("rationale"), str) or len(item.get("rationale", "").strip()) < 30:
        errors.append("rationale_too_short")

    # Language guard: generated items must be German.
    lang_fields = [item.get("stem", ""), item.get("rationale", ""), item.get("generation_notes", "")]
    if isinstance(item.get("options"), dict):
        lang_fields.extend(str(v) for v in item.get("options", {}).values())
    if any(_looks_english(str(field)) for field in lang_fields):
        errors.append("language_not_german")

    # Force experimental flags regardless of model output.
    item["b_basis"] = "llm_expert_estimate"
    item["b_confidence"] = "low"
    item["calibration_status"] = "experimental_generated"
    item["generated_online"] = True

    return not errors, errors


def _append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def build_generation_user_prompt(
    target_b: float,
    target_dimension: Optional[str],
    administered_items: List[Dict[str, Any]],
) -> str:
    administered_topics = []
    for r in administered_items[-12:]:
        administered_topics.append({
            "item_id": r.get("item_id"),
            "dimension": r.get("dimension"),
            "b_value": r.get("b_value"),
            "correct": r.get("correct"),
        })
    return json.dumps({
        "target_b": round(float(target_b), 2),
        "target_dimension": target_dimension or "choose a relevant underrepresented Phase-1 domain",
        "already_administered_recent_items": administered_topics,
        "instruction": "Erzeuge genau ein neues geschlossenes Item nahe target_b. Schreibe das gesamte Item ausschließlich auf Deutsch. Vermeide die Wiederholung kürzlich verwendeter Themen. Gib ausschließlich valides JSON zurück."
    }, ensure_ascii=False, indent=2)


def generate_phase1_item_with_openai(
    system_prompt: str,
    target_b: float,
    target_dimension: Optional[str],
    administered_items: List[Dict[str, Any]],
    api_key: Optional[str],
    model: str,
    item_id_prefix: str = "P1_GEN",
    quarantine_path: Optional[Path] = None,
    failure_log_path: Optional[Path] = None,
) -> Dict[str, Any]:
    try:
        from openai import OpenAI
    except Exception as exc:
        raise RuntimeError("Das Python-Paket 'openai' ist nicht installiert. Bitte 'pip install -r requirements.txt' ausführen.") from exc

    resolved_key = api_key or os.getenv("OPENAI_API_KEY")
    if not resolved_key:
        raise RuntimeError("Kein OpenAI-API-Key verfügbar; Online-Itemgenerierung wird übersprungen.")

    client = OpenAI(api_key=resolved_key)
    user_prompt = build_generation_user_prompt(target_b, target_dimension, administered_items)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        text = response.choices[0].message.content or "{}"
    except TypeError:
        response = client.chat.completions.create(model=model, messages=messages, temperature=0.2)
        text = response.choices[0].message.content or "{}"

    try:
        item = _extract_json(text)
        ok, errors = validate_generated_item(item)
        if not ok:
            raise ValueError("Generated item failed validation: " + "; ".join(errors))

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        item["item_id"] = f"{item_id_prefix}_{timestamp}"
        item["created_at"] = datetime.now().isoformat(timespec="seconds")
        item["model"] = model

        if quarantine_path:
            _append_jsonl(quarantine_path, {"event": "accepted_generated_item", "item": item})
        return item
    except Exception as exc:
        if failure_log_path:
            _append_jsonl(failure_log_path, {
                "event": "generation_failed",
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "target_b": target_b,
                "target_dimension": target_dimension,
                "model": model,
                "error": str(exc),
                "raw_text": text,
            })
        raise
