
from __future__ import annotations
import json
from typing import Dict, Any


def build_phase2_judge_prompt(system_prompt: str, case: Dict[str, Any], candidate_answer: str) -> str:
    return f"""{system_prompt}

CASE_ID:
{case["case_id"]}

CASE_TITLE:
{case["title"]}

CASE_TEXT:
{case["case_text"]}

RUBRIC_DIMENSIONS:
{json.dumps(case["rubric_dimensions"], ensure_ascii=False, indent=2)}

CANDIDATE_ANSWER:
{candidate_answer}
"""
