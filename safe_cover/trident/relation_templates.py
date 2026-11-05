from __future__ import annotations
from typing import Dict

# IMPORTANT:
# - hypothesis must be stable across runs (calibration depends on this)
# - keep it short, declarative, MNLI-friendly
# - always anchor the object/entity in the hypothesis text

RELATION_SLOT_TO_TEMPLATE: Dict[str, Dict[str, str]] = {
    "director": {
        "relation_kind": "DIRECTOR",
        "relation_pid": "P57",
        "relation_schema_source": "frozen_slot_templates",
        "relation_schema_version": "v1",
        "predicate": "was directed by",
        "hypothesis": "The passage states who directed {entity}.",
        "answer_role": "subject",
        "anchor_policy": "ANY",
    },
    "mother": {
        "relation_kind": "MOTHER",
        "relation_pid": "P25",
        "relation_schema_source": "frozen_slot_templates",
        "relation_schema_version": "v1",
        "predicate": "mother of",
        "hypothesis": "The passage states the mother of {entity}.",
        "answer_role": "object",
        "anchor_policy": "ANY",
    },
    "capital": {
        "relation_kind": "CAPITAL",
        "relation_pid": "P36",
        "relation_schema_source": "frozen_slot_templates",
        "relation_schema_version": "v1",
        "predicate": "capital of",
        "hypothesis": "The passage states the capital of {entity}.",
        "answer_role": "object",
        "anchor_policy": "ANY",
    },
    "award": {
        "relation_kind": "AWARD",
        "relation_pid": "P166",
        "relation_schema_source": "frozen_slot_templates",
        "relation_schema_version": "v1",
        "predicate": "won",
        "hypothesis": "The passage states what award {entity} won.",
        "answer_role": "object",
        "anchor_policy": "ANY",
    },
    "spouse": {
        "relation_kind": "SPOUSE",
        "relation_pid": "P26",
        "relation_schema_source": "frozen_slot_templates",
        "relation_schema_version": "v1",
        "predicate": "spouse of",
        "hypothesis": "The passage states the spouse of {entity}.",
        "answer_role": "object",
        "anchor_policy": "ANY",
    },
    "birth_date": {
        "relation_kind": "BIRTH_DATE",
        "relation_pid": "P569",
        "relation_schema_source": "frozen_slot_templates",
        "relation_schema_version": "v1",
        "predicate": "was born on",
        "hypothesis": "The passage states when {entity} was born.",
        "answer_role": "object",
        "anchor_policy": "ANY",
    },
    "death_date": {
        "relation_kind": "DEATH_DATE",
        "relation_pid": "P570",
        "relation_schema_source": "frozen_slot_templates",
        "relation_schema_version": "v1",
        "predicate": "died on",
        "hypothesis": "The passage states when {entity} died.",
        "answer_role": "object",
        "anchor_policy": "ANY",
    },
}
