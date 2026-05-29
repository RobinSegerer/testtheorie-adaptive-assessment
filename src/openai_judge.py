from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Optional

from .prompt_builder import build_phase2_judge_prompt


def _extract_json(text: str) -> Dict[str, Any]:
    """Parse a JSON object from a model response, with a guarded fallback."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def _validate_scoring_result(result: Dict[str, Any], case: Dict[str, Any]) -> Dict[str, Any]:
    """Add simple validation metadata without hiding the original response."""
    expected = {dim["dimension_id"] for dim in case.get("rubric_dimensions", [])}
    scoring_results = result.get("scoring_results", {})
    observed = set(scoring_results.keys())

    validation_errors = []
    missing = expected - observed
    extra = observed - expected
    if missing:
        validation_errors.append(f"Missing rubric dimensions: {sorted(missing)}")
    if extra:
        validation_errors.append(f"Unexpected rubric dimensions: {sorted(extra)}")

    total_score = 0
    max_score = 3 * len(expected)
    for dim_id in expected & observed:
        score = scoring_results.get(dim_id, {}).get("score")
        if not isinstance(score, int) or score < 0 or score > 3:
            validation_errors.append(f"Invalid score for {dim_id}: {score!r}")
        else:
            total_score += score

    if "score_summary" not in result:
        level = "not_yet_expert"
        if max_score and total_score >= 0.85 * max_score:
            level = "reviewer_like"
        elif max_score and total_score >= 0.70 * max_score:
            level = "expert"
        elif max_score and total_score >= 0.50 * max_score:
            level = "advanced"
        result["score_summary"] = {
            "total_score": total_score,
            "max_score": max_score,
            "profile_level": level,
            "phase_2_score_is_not_irt_theta": True,
        }

    result["validation"] = {
        "valid_schema_basic": not validation_errors,
        "errors": validation_errors,
    }
    return result


def score_with_openai(
    case: Dict[str, Any],
    candidate_answer: str,
    system_prompt: str,
    api_key: Optional[str] = None,
    model: str = "gpt-4.1-mini",
) -> Dict[str, Any]:
    """Automatic Phase-2 scoring via OpenAI.

    The LLM score is rubriziertes Scoring, not empirical IRT calibration.
    API key resolution order:
    1. explicit api_key argument,
    2. OPENAI_API_KEY environment variable,
    3. OpenAI SDK default handling.
    """
    try:
        from openai import OpenAI
    except Exception as exc:
        raise RuntimeError("Das Python-Paket 'openai' ist nicht installiert. Bitte 'pip install -r requirements.txt' ausführen.") from exc

    resolved_key = api_key or os.getenv("OPENAI_API_KEY")
    if not resolved_key:
        raise RuntimeError(
            "Kein OpenAI-API-Key gefunden. Setze OPENAI_API_KEY als Umgebungsvariable "
            "oder trage ihn in .streamlit/secrets.toml ein."
        )

    client = OpenAI(api_key=resolved_key)
    full_prompt = build_phase2_judge_prompt(system_prompt, case, candidate_answer)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a strict psychometric scoring engine. "
                "Return one valid JSON object only. Do not include markdown."
            ),
        },
        {"role": "user", "content": full_prompt},
    ]

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,
            response_format={"type": "json_object"},
        )
        text = response.choices[0].message.content or "{}"
    except TypeError:
        # Fallback for SDK/API combinations without JSON mode.
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,
        )
        text = response.choices[0].message.content or "{}"

    result = _extract_json(text)
    result["model"] = model
    result["case_id"] = case.get("case_id")
    return _validate_scoring_result(result, case)
