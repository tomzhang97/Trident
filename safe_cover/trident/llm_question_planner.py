from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from .chain_builder import strict_json_call


@dataclass
class QuestionPlan:
    question_type: str
    # list of required fact slots, each is {slot, entity, constraints?}
    required_facts: List[Dict[str, Any]]


class LLMQuestionPlanner:
    """
    SINGLE LLM call per question:
      - determine question type
      - determine required fact slots and their anchored entity
    No hypothesis phrasing allowed here.
    """

    def __init__(self, llm: Any, max_facts: int = 3, max_new_tokens: int = 200):
        self.llm = llm
        self.max_facts = max_facts
        self.max_new_tokens = max_new_tokens

    def plan(self, question: str) -> QuestionPlan:
        prompt = f"""
You are a planner for a retrieval+certification system.

Task:
1) Identify the question_type: one of ["WHO","WHAT","WHEN","WHERE","HOW_MANY","YES_NO","OTHER"].
2) Identify the minimum required facts needed to answer.
Each required fact MUST include:
- slot: a short canonical slot name like "director", "mother", "capital", "award", "spouse", "birth_date"
- entity: the anchored entity string from the question (film/person/place/etc.)
Optional:
- constraints: short text constraints if needed

Rules:
- Do NOT write hypotheses or natural language explanations.
- Keep required_facts length <= {self.max_facts}.
- If uncertain, output an empty required_facts list.

Return JSON:
{{
  "question_type": "...",
  "required_facts": [
    {{ "slot": "...", "entity": "...", "constraints": "..." }},
    ...
  ]
}}

Question:
{question}
""".strip()

        obj, _raw = strict_json_call(self.llm, prompt, max_new_tokens=self.max_new_tokens)

        qt = str(obj.get("question_type", "OTHER")).upper().strip()
        rf = obj.get("required_facts", [])
        if not isinstance(rf, list):
            rf = []

        # hard sanitize
        cleaned: List[Dict[str, Any]] = []
        for x in rf[: self.max_facts]:
            if not isinstance(x, dict):
                continue
            slot = str(x.get("slot", "")).strip().lower()
            ent = str(x.get("entity", "")).strip()
            if not slot or not ent:
                continue
            cleaned.append(
                {
                    "slot": slot,
                    "entity": ent,
                    "constraints": str(x.get("constraints", "")).strip(),
                }
            )

        return QuestionPlan(question_type=qt, required_facts=cleaned)