#!/usr/bin/env python3
"""
trident/facets.py
UPDATES:
- Added generic BRIDGE_HOP type.
- Generates only adjacent hops (no aggregate).
- Works for 2, 3, 4+ title chains.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

_PUNCT_FIX_RE = re.compile(r"\s+([,.;:?!])")
_MULTI_SPACE_RE = re.compile(r"\s{2,}")
_APOS_S_RE = re.compile(r"\s+'s\b", re.IGNORECASE)
_BAD_CHARS_RE = re.compile(r"[\u200b\u200c\u200d\uFEFF]")

def _clean_text(s: str) -> str:
    s = (s or "").strip()
    s = _BAD_CHARS_RE.sub("", s)
    s = _APOS_S_RE.sub("'s", s)
    s = _PUNCT_FIX_RE.sub(r"\1", s)
    s = _MULTI_SPACE_RE.sub(" ", s)
    return s

def _ensure_sentence(s: str) -> str:
    s = _clean_text(s)
    if not s: return s
    if s[-1] not in ".?!": s += "."
    return s

def _safe_phrase(s: str) -> str:
    s = _clean_text(s)
    s = s.strip("`\"' ")
    return _clean_text(s)

def _looks_like_junk_entity(mention: str) -> bool:
    """
    Filter junk ENTITY facets that waste cover budget and dilute certification.

    Rejects:
    - Generic nouns: "film", "movie", "song", "album", "book", "episode", "series", etc.
    - Partial punctuation entities: ", Blind Shaft", " Film", etc.
    - WH-leading contamination: "Are North Marion High School"
    - Single-letter or very short non-informative tokens
    """
    m = _safe_phrase(mention)
    if not m: return True

    # CRITICAL FIX: Filter leading/trailing punctuation
    # Rejects: ", Blind Shaft", " Film", "Film,", etc.
    if m[0] in ",.;:!?'\"-" or m[-1] in ",.;:!?'\"":
        return True

    ml = m.lower()

    # CRITICAL FIX: Expanded stoplist of generic nouns
    # These waste cover budget and create false positives for "required facet" logic
    generic_nouns = {
        "first", "second", "third", "which", "what", "who", "when", "where", "one", "two",
        "this", "that", "the", "a", "an", "and", "or",
        # Media types
        "film", "movie", "song", "album", "book", "episode", "series", "season", "game",
        "show", "documentary", "video", "track", "record", "novel", "story", "play",
        # Generic places/things
        "city", "town", "country", "place", "location", "area", "region",
        "thing", "person", "people", "man", "woman", "group", "team", "company",
        # Generic roles
        "director", "actor", "actress", "writer", "author", "artist", "musician",
        "singer", "player", "coach", "politician", "scientist", "president",
    }

    if ml in generic_nouns: return True

    # CRITICAL FIX: Filter WH-leading contamination
    # Rejects: "Are North Marion High School", "Is John", "The Film", etc.
    wh_prefixes = {"are", "is", "was", "were", "the", "a", "an", "can", "will", "would", "could"}
    first_token = ml.split()[0] if ml.split() else ""
    if first_token in wh_prefixes:
        return True

    # Single token that's very short and non-informative
    if len(m.split()) == 1 and len(m) <= 3: return True

    # Bare numbers
    if re.fullmatch(r"\d+(\.\d+)?", ml): return True

    return False

# WH-words and generic terms that should not be treated as real RELATION endpoints
_RELATION_WH_WORDS = {"who", "what", "which", "when", "where", "why", "how", "whom", "whose"}
_RELATION_GENERIC = {"the", "a", "an", "of", "to", "in", "on", "for", "and", "or", "is", "are",
                     "was", "were", "be", "been", "being", "have", "has", "had", "do", "does",
                     "did", "will", "would", "could", "should", "may", "might", "must", "shall",
                     "director", "film", "movie", "actor", "actress", "person", "thing", "place",
                     "name", "title", "author", "writer", "book", "song", "album", "band", "group"}

def _dedupe_strings(xs: List[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for x in xs:
        x = _safe_phrase(x)
        if not x: continue
        key = x.lower()
        if key in seen: continue
        seen.add(key)
        out.append(x)
    return out


def _dedupe_entities_by_longest_span(mentions: List[str]) -> List[str]:
    """
    Deduplicate entity mentions by keeping only the longest span.

    If 'Michael' and 'Michael Doeberl' both exist, drop 'Michael'.
    This prevents partial matches from covering wrong passages.

    Example:
        Input: ['Michael', 'Michael Doeberl', 'Edward', 'Edward Dearle']
        Output: ['Michael Doeberl', 'Edward Dearle']
    """
    if not mentions:
        return []

    # Normalize and dedupe exact matches first
    normalized = []
    seen_lower = set()
    for m in mentions:
        m = _safe_phrase(m)
        if not m:
            continue
        m_lower = m.lower()
        if m_lower not in seen_lower:
            seen_lower.add(m_lower)
            normalized.append(m)

    if not normalized:
        return []

    # Sort by length descending so we check longer spans first
    by_length = sorted(normalized, key=lambda x: len(x), reverse=True)

    # Keep only mentions that are not substrings of a longer mention
    result = []
    for m in by_length:
        m_lower = m.lower()
        # Check if this mention is a substring of any already-kept mention
        is_substring = False
        for kept in result:
            kept_lower = kept.lower()
            # Check if m is a proper substring (not equal)
            if m_lower != kept_lower and m_lower in kept_lower:
                is_substring = True
                break
        if not is_substring:
            result.append(m)

    return result


class FacetType(Enum):
    ENTITY = "ENTITY"
    RELATION = "RELATION"
    TEMPORAL = "TEMPORAL"
    NUMERIC = "NUMERIC"
    BRIDGE_HOP = "BRIDGE_HOP" # Generic for all hops
    BRIDGE_HOP1 = "BRIDGE_HOP1" # Legacy/Specific
    BRIDGE_HOP2 = "BRIDGE_HOP2" # Legacy/Specific
    BRIDGE = "BRIDGE"
    COMPARISON = "COMPARISON"
    CAUSAL = "CAUSAL"
    PROCEDURAL = "PROCEDURAL"

    @classmethod
    def core_types(cls) -> list:
        return [cls.ENTITY, cls.RELATION, cls.TEMPORAL, cls.NUMERIC, cls.BRIDGE_HOP]


def instantiate_facets(
    facets: List["Facet"],
    bindings: Dict[str, str]
) -> List["Facet"]:
    """
    Replace symbolic placeholders in facet templates with concrete bound entities.

    This is the REQUIRED facet instantiation step for hop-2 facets in
    two-pass Certified Adaptive Safe-Cover.

    Example:
        bindings = {"DIRECTOR_RESULT": "Xawery Żuławski"}

        Input facet template:
            {"subject": "[DIRECTOR_RESULT]", "predicate": "'s mother", ...}

        Output facet template:
            {"subject": "Xawery Żuławski", "predicate": "'s mother", ...}

    CRITICAL: The output facets are NEW objects. Original facets are not mutated.

    Args:
        facets: List of facets (possibly containing placeholders like [VAR_RESULT])
        bindings: Dict mapping variable names to bound entity values
                  e.g., {"DIRECTOR_RESULT": "Xawery Żuławski"}

    Returns:
        List of new facets with placeholders replaced by concrete values
    """
    import os
    debug = os.environ.get("TRIDENT_DEBUG_INSTANTIATE", "0") == "1"

    if not bindings:
        return list(facets)

    new_facets = []
    for facet in facets:
        # Build new template with placeholders replaced
        old_tpl = facet.template or {}
        new_tpl = dict(old_tpl)  # Shallow copy

        # Track if we made any replacements
        replaced = False

        # Fields that may contain placeholders
        placeholder_fields = ["subject", "object", "hypothesis", "entity1", "entity2", "bridge_entity"]

        for field in placeholder_fields:
            if field not in new_tpl:
                continue
            value = str(new_tpl[field] or "")

            # Replace each binding
            for var_name, entity_value in bindings.items():
                placeholder = f"[{var_name}]"
                if placeholder in value:
                    new_value = value.replace(placeholder, entity_value)
                    new_tpl[field] = new_value
                    replaced = True

                    if debug:
                        print(f"[INSTANTIATE] {facet.facet_id} field={field}: "
                              f"'{value}' -> '{new_value}'")

        # Also update metadata to mark as instantiated
        old_meta = facet.metadata or {}
        new_meta = dict(old_meta)
        if replaced:
            new_meta["instantiated"] = True
            new_meta["bindings_applied"] = list(bindings.keys())

        # ================================================================
        # PARENT NORMALIZATION: Normalize MOTHER/FATHER to PARENT for scoring
        # ================================================================
        # Evidence often says "X is the son of Y" which doesn't contain "mother".
        # NLI fails to entail "mother of X" from "son of Y" reliably.
        # Fix: Score as PARENT (hypothesis: "The passage states a parent of X")
        # and gate anchors include {mother, father, son of, daughter of, parent}.
        # Keep outer_relation_type for reporting/analysis.
        outer_rel = new_tpl.get("outer_relation_type", "")
        if outer_rel in ("MOTHER", "FATHER"):
            new_tpl["scoring_relation_kind"] = "PARENT"
            if debug:
                print(f"[INSTANTIATE] {facet.facet_id} normalized {outer_rel} -> scoring_relation_kind=PARENT")

        # Create new facet with updated template
        # Note: Facet is frozen=True, so we create a new instance
        new_facet = Facet(
            facet_id=facet.facet_id,
            facet_type=facet.facet_type,
            template=new_tpl,
            weight=facet.weight,
            metadata=new_meta,
            required=facet.required,  # Preserve required flag
        )
        new_facets.append(new_facet)

    return new_facets


def is_wh_question(question: str) -> bool:
    """Check if question is a WH-question (who, what, where, when, which, how)."""
    q_lower = question.lower().strip()
    wh_starters = ('who ', 'what ', 'where ', 'when ', 'which ', 'how ', 'whose ')
    return q_lower.startswith(wh_starters) or any(f' {w}' in q_lower[:30] for w in ['who', 'what', 'where', 'when', 'which'])


def mark_required_facets(
    facets: List["Facet"],
    question: str,
    require_relation: bool = False,
) -> List["Facet"]:
    """
    Mark RELATION facets as required for WH-questions.

    For questions like "Who is X?" or "Where was Y born?", the RELATION facet
    MUST be covered for a valid answer. Without it, the answer is unreliable.

    Args:
        facets: List of facets
        question: The original question

    Returns:
        List of facets with required flags set appropriately
    """
    if not (require_relation or is_wh_question(question)):
        return facets

    import os
    debug = os.environ.get("TRIDENT_DEBUG_REQUIRED", "0") == "1"

    new_facets = []
    for facet in facets:
        # Mark RELATION facets as required for WH-questions
        if facet.facet_type == FacetType.RELATION:
            if not facet.required:
                # Create new facet with required=True
                new_facet = Facet(
                    facet_id=facet.facet_id,
                    facet_type=facet.facet_type,
                    template=facet.template,
                    weight=facet.weight,
                    metadata=facet.metadata,
                    required=True,
                )
                if debug:
                    print(f"[REQUIRED] Marking RELATION facet as required: {facet.facet_id}")
                new_facets.append(new_facet)
            else:
                new_facets.append(facet)
        else:
            new_facets.append(facet)

    return new_facets

@dataclass(frozen=True)
class Facet:
    facet_id: str
    facet_type: FacetType
    template: Dict[str, Any]
    weight: float = 1.0
    metadata: Dict[str, Any] = None
    required: bool = False  # If True, this facet MUST be covered for certification

    def __post_init__(self):
        if self.metadata is None:
            object.__setattr__(self, "metadata", {})
        ft = object.__getattribute__(self, "facet_type")
        if isinstance(ft, str):
            object.__setattr__(self, "facet_type", FacetType(ft))

    def to_hypothesis(self, passage_text: Optional[str] = None) -> str:
        ft = self.facet_type
        tpl = self.template

        if ft == FacetType.ENTITY:
            mention = _safe_phrase(tpl.get("mention", ""))
            if passage_text:
                # Try to extract context like "X is a Y" or "X, a Y"
                pat1 = re.search(rf"\b{re.escape(mention)}\s+(?:is|was|are|were)\s+(?:an?|the)\s+([^,.]+)", passage_text, re.IGNORECASE)
                if pat1 and len(pat1.group(1).split()) < 10:
                    return _ensure_sentence(f"{mention} is {pat1.group(1)}")
                pat2 = re.search(rf"\b{re.escape(mention)}\s*[,(]\s*(?:an?|the)\s+([^,)]+)", passage_text, re.IGNORECASE)
                if pat2 and len(pat2.group(1).split()) < 10:
                    return _ensure_sentence(f"{mention} is {pat2.group(1)}")
            # Simple fallback: just check if the passage contains the entity
            # Previous hypothesis was too strict ("identifies unambiguously") for NLI
            return _ensure_sentence(f'The passage mentions "{mention}".')

        if ft == FacetType.NUMERIC:
            entity = _safe_phrase(tpl.get("entity", ""))
            value = _safe_phrase(tpl.get("value", ""))
            unit = _safe_phrase(tpl.get("unit", ""))
            attr = _safe_phrase(tpl.get("attribute", ""))
            val_str = f"{value} {unit}".strip()
            if entity and attr: return _ensure_sentence(f"The passage states that {entity}'s {attr} is {val_str}")
            return _ensure_sentence(f"The passage binds the value {val_str} to a specific property")

        if ft == FacetType.RELATION:
            custom_hyp = tpl.get("custom_hypothesis")
            if custom_hyp:
                return _ensure_sentence(str(custom_hyp))

            subject = _safe_phrase(tpl.get("subject", ""))
            predicate = _safe_phrase(tpl.get("predicate", ""))
            obj = _safe_phrase(tpl.get("object", ""))
            relation_kind = _safe_phrase(tpl.get("relation_kind", ""))
            is_wh_subject = bool((self.metadata or {}).get("is_wh_subject"))

            # WH-subject: use existential hypothesis entailed by typical passive phrasing
            if is_wh_subject:
                anchor = obj or subject or "the topic"
                if relation_kind.upper() == "DIRECTOR" or "direct" in predicate.lower():
                    return _ensure_sentence(f"{anchor} was directed by someone")
                if relation_kind.upper() == "MOTHER":
                    return _ensure_sentence(f"Someone is the mother of {anchor}")
                if relation_kind.upper() == "FATHER":
                    return _ensure_sentence(f"Someone is the father of {anchor}")
                if relation_kind.upper() == "PARENT":
                    return _ensure_sentence(f"Someone is a parent of {anchor}")
                return _ensure_sentence(f"Someone is related to {anchor}")

            anchor = obj or subject or "the topic"
            core = predicate if predicate else "is related to"
            return _ensure_sentence(f"The passage states that {anchor} {core}.")

        if ft == FacetType.TEMPORAL:
            time = tpl.get('time', '')
            event = tpl.get('event', '')
            if event: return _ensure_sentence(f"The passage states that {event} happened in {time}")
            return _ensure_sentence(f"The passage mentions {time} in relation to a specific event")

        if ft == FacetType.COMPARISON:
            e1 = tpl.get('entity1', '')
            e2 = tpl.get('entity2', '')
            attr = tpl.get('attribute', 'different')
            return _ensure_sentence(f"The passage compares {e1} and {e2} regarding {attr}")

        # BRIDGE HYPOTHESES (Unified)
        if "BRIDGE_HOP" in ft.value:
            e1 = tpl.get("entity1", "")
            eb = tpl.get("bridge_entity", "")
            rel = tpl.get("relation", "")
            if "includes guest" in rel:
                return _ensure_sentence(f"The passage explicitly states that {e1} includes guest appearances from {eb}")
            return _ensure_sentence(f"The passage explicitly mentions a direct factual connection between {e1} and {eb}")

        if ft == FacetType.CAUSAL:
            return _ensure_sentence(f"The passage explains the cause or reason for the event")

        if ft == FacetType.PROCEDURAL:
            return _ensure_sentence(f"The passage describes a procedure, step, or method")

        return _ensure_sentence(str(self.template))

    def get_keywords(self) -> List[str]:
        """Extract keywords from facet template for lexical matching."""
        keywords = []
        tpl = self.template
        ft = self.facet_type

        if ft == FacetType.ENTITY:
            if tpl.get("mention"):
                keywords.append(tpl["mention"])

        elif ft == FacetType.NUMERIC:
            if tpl.get("entity"):
                keywords.append(tpl["entity"])
            if tpl.get("value"):
                keywords.append(str(tpl["value"]))

        elif ft == FacetType.RELATION:
            if tpl.get("subject"):
                keywords.append(tpl["subject"])
            if tpl.get("object"):
                keywords.append(tpl["object"])

        elif ft == FacetType.TEMPORAL:
            if tpl.get("time"):
                keywords.append(tpl["time"])
            if tpl.get("event"):
                keywords.append(tpl["event"])

        elif ft == FacetType.COMPARISON:
            if tpl.get("entity1"):
                keywords.append(tpl["entity1"])
            if tpl.get("entity2"):
                keywords.append(tpl["entity2"])

        elif "BRIDGE_HOP" in ft.value or ft == FacetType.BRIDGE:
            if tpl.get("entity1"):
                keywords.append(tpl["entity1"])
            if tpl.get("bridge_entity"):
                keywords.append(tpl["bridge_entity"])
            elif tpl.get("entity2"):
                keywords.append(tpl["entity2"])

        elif ft == FacetType.CAUSAL:
            # Extract any entities from template
            for key in ["cause", "effect", "entity"]:
                if tpl.get(key):
                    keywords.append(tpl[key])

        elif ft == FacetType.PROCEDURAL:
            # Extract procedure-related terms
            for key in ["action", "step", "method"]:
                if tpl.get(key):
                    keywords.append(tpl[key])

        return [kw for kw in keywords if kw]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "facet_id": self.facet_id,
            "facet_type": self.facet_type.value,
            "template": self.template,
            "weight": self.weight,
            "metadata": self.metadata,
            "required": self.required,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Facet":
        return cls(
            facet_id=data["facet_id"],
            facet_type=FacetType(data["facet_type"]),
            template=data["template"],
            weight=data.get("weight", 1.0),
            metadata=data.get("metadata"),
            required=data.get("required", False),
        )

class FacetMiner:

    def __init__(self, config: Any, llm: Optional[Any] = None):
        self.config = config

        # Optional LLM for query-time facet planning / relation classification.
        # Kept None-safe: all LLM-driven behavior is gated by config/env flags.
        self.llm = llm

        # NLP dependency removed for stability; keep attribute for compatibility.
        self.nlp = None

    def _relation_facet_builder(
        self,
        facet_id: str,
        template: Dict[str, Any],
        *,
        weight: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
        text: Optional[str] = None,
        normalize: bool = True,
    ) -> Optional[Facet]:
        """Construct a RELATION facet with WH detection and placeholder gating."""

        def _is_placeholder(x: str) -> bool:
            x = (x or "").strip().lower()
            return (not x) or (x in {"?", "person", "someone", "something"}) or x in {
                "who",
                "what",
                "where",
                "when",
                "which",
                "whom",
                "whose",
                "why",
                "how",
            }

        tpl = dict(template or {})

        subject_str = str(tpl.get("subject", ""))
        object_str = str(tpl.get("object", ""))
        if _is_placeholder(subject_str) and _is_placeholder(object_str):
            return None

        subj = subject_str.strip().lower()
        meta = dict(metadata or {})
        is_wh_subject = meta.get("is_wh_subject")
        if is_wh_subject is None:
            is_wh_subject = subj in {
                "who",
                "what",
                "where",
                "when",
                "which",
                "whom",
                "whose",
                "why",
                "how",
            }

        meta["is_wh_subject"] = bool(is_wh_subject)
        if text:
            meta.setdefault("debug_text", text)
        if meta["is_wh_subject"]:
            tpl["anchor_policy"] = "ANY"

        return Facet(
            facet_id=facet_id,
            facet_type=FacetType.RELATION,
            template=tpl,
            weight=weight,
            metadata=meta,
        )

    def _build_relation_facet(
        self,
        facet_id: str,
        template: Dict[str, Any],
        *,
        weight: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
        text: Optional[str] = None,
        normalize: bool = True,
    ) -> Optional[Facet]:
        """Wrapper to keep backward compatibility with callers expecting this name."""

        return self._relation_facet_builder(
            facet_id,
            template,
            weight=weight,
            metadata=metadata,
            text=text,
            normalize=normalize,
        )

    def extract_facets(
        self,
        query: str,
        supporting_facts: Optional[List[Tuple[str, int]]] = None,
    ) -> List[Facet]:
        query = _clean_text(query)

        facets: List[Facet] = []
        facets.extend(self._extract_entity_facets(query))
        facets.extend(self._extract_relation_facets(query))
        facets.extend(self._extract_temporal_facets(query))
        facets.extend(self._extract_numeric_facets(query))
        facets.extend(self._extract_comparison_facets(query))
        facets.extend(self._extract_causal_facets(query))

        return self._deduplicate_facets(facets)

    # ---- entity ----
    def _extract_or_entities(self, query: str) -> List[str]:
        q = query.strip()
        m = re.search(r"\bfirst\s+(.+?)\s+or\s+(.+?)\s*\?$", q, flags=re.IGNORECASE)
        if not m: m = re.search(r"(.+?)\s+or\s+(.+?)\s*\?$", q, flags=re.IGNORECASE)
        if not m: return []
        a, b = m.group(1).strip(), m.group(2).strip()
        a = re.sub(r"^(which|what|who|where|when|why|how)\b.*?\b(first|more|less|better|worse|earlier|later)\b", "", a, flags=re.IGNORECASE).strip()
        return [x for x in [a, b] if _safe_phrase(x) and not _looks_like_junk_entity(x)]

    def _extract_entity_facets(self, query: str) -> List[Facet]:
        mentions: List[str] = []
        mentions.extend(self._extract_or_entities(query))
        if not mentions:
            mentions.extend(re.findall(r"\b[A-Z][A-Za-z0-9'\-’]+(?:\s+[A-Z][A-Za-z0-9'\-’]+)*\b", query))
        # CRITICAL: Use longest-span dedup to prevent "Michael" when "Michael Doeberl" exists
        # This prevents partial matches from covering wrong passages
        mentions = [m for m in _dedupe_entities_by_longest_span(mentions) if not _looks_like_junk_entity(m)]
        return [Facet(facet_id=f"entity_{i}", facet_type=FacetType.ENTITY, template={"mention": _safe_phrase(m)}) for i, m in enumerate(mentions)]

    # ---- relation (planned or heuristic) ----
    def _extract_relation_facets(self, query: str) -> List[Facet]:
        q = query.lower()
        facets: List[Facet] = []

        ents = self._extract_entity_facets(query)

        def _choose_anchor(entity_facets: List[Facet]) -> str:
            mentions: List[str] = []
            for e in entity_facets:
                mention = _safe_phrase((e.template or {}).get("mention", ""))
                if mention:
                    mentions.append(mention)

            if not mentions:
                return ""

            generic_singletons = {"film", "movie", "person", "someone", "something", "director"}

            def _score(m: str) -> Tuple[int, int]:
                tokens = m.split()
                if len(tokens) == 1 and tokens[0].lower() in generic_singletons:
                    return (-1, len(m))
                return (1 if len(tokens) > 1 else 0, len(m))

            mentions.sort(key=_score, reverse=True)
            return mentions[0]

        anchor = _choose_anchor(ents)

        def _matches_mother(question_text: str) -> bool:
            return bool(
                re.search(r"\bwho\b[^?]*\bmother of\b", question_text)
                or re.search(r"\bmother of\b[^?]*\bwho\b", question_text)
            )

        def _matches_director(question_text: str) -> bool:
            return bool(
                re.search(r"\bwho\b[^?]*\bdirector of\b", question_text)
                or re.search(r"\bdirector of\b[^?]*\bwho\b", question_text)
                or re.search(r"\bwho\b[^?]*\bdirected\b", question_text)
            )

        nested_parent_match = re.search(
            r"\b(mother|father|parent|son|daughter|child|spouse|wife|husband)\s+of\s+the\s+director\s+of\b",
            q,
        )

        if anchor and nested_parent_match:
            relation_token = nested_parent_match.group(1).lower()
            relation_map = {
                "mother": "MOTHER",
                "father": "FATHER",
                "parent": "PARENT",
                "son": "CHILD",
                "daughter": "CHILD",
                "child": "CHILD",
                "spouse": "SPOUSE",
                "wife": "SPOUSE",
                "husband": "SPOUSE",
            }

            outer_relation = relation_map.get(relation_token, "PARENT")
            predicate_text = f"{relation_token} of" if relation_token not in {"son", "daughter", "child"} else "child of"

            hop1 = self._relation_facet_builder(
                "rel_director_hop1",
                template={
                    "relation_kind": "DIRECTOR",
                    "subject": "?",
                    "object": anchor,
                    "predicate": "directed",
                    "answer_role": "subject",
                    "inner_relation_type": "DIRECTOR",
                    "compositional": True,
                    "hop": 1,
                },
                metadata={"is_wh_subject": True},
            )

            hop2 = self._relation_facet_builder(
                f"rel_{relation_token}_of_director_hop2",
                template={
                    "relation_kind": outer_relation,
                    "subject": "?",
                    "object": "[DIRECTOR_RESULT]",
                    "bridge_entity": "[DIRECTOR_RESULT]",
                    "predicate": predicate_text,
                    "answer_role": "subject",
                    "outer_relation_type": outer_relation,
                    "inner_relation_type": "DIRECTOR",
                    "hop": 2,
                },
                metadata={"is_wh_subject": True},
            )

            if hop1:
                facets.append(hop1)
            if hop2:
                facets.append(hop2)

            return facets

        elif anchor and _matches_mother(q):
            rel = self._relation_facet_builder(
                "rel_mother",
                template={
                    "relation_kind": "MOTHER",
                    "subject": "?",
                    "object": anchor,
                    "predicate": "mother",
                    "answer_role": "subject",
                },
                metadata={"is_wh_subject": True},
            )
            if rel:
                facets.append(rel)

        if anchor and _matches_director(q):
            rel = self._relation_facet_builder(
                "rel_director",
                template={
                    "relation_kind": "DIRECTOR",
                    "subject": "?",
                    "object": anchor,
                    "predicate": "directed",
                    "answer_role": "subject",
                },
                metadata={"is_wh_subject": True},
            )
            if rel:
                facets.append(rel)

        return facets

    # ---- temporal ----
    def _extract_temporal_facets(self, query: str) -> List[Facet]:
        facets: List[Facet] = []
        patterns = [r"\b(19\d{2}|20\d{2})\b", r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b"]
        for pat in patterns:
            for m in re.finditer(pat, query, flags=re.IGNORECASE):
                time_text = _safe_phrase(m.group(0))
                event = _safe_phrase(self._extract_temporal_event(query, m.span()))
                facets.append(Facet(facet_id=f"temporal_{len(facets)}", facet_type=FacetType.TEMPORAL, template={"time": time_text, "event": event}))
        return facets

    # ---- numeric ----
    def _extract_numeric_facets(self, query: str) -> List[Facet]:
        facets: List[Facet] = []
        numeric_pattern = r"\b(\d+(?:\.\d+)?)\s*(%|percent|million|billion|thousand|hundred)?\b"
        for m in re.finditer(numeric_pattern, query, flags=re.IGNORECASE):
            value = _safe_phrase(m.group(1))
            unit = _safe_phrase(m.group(2) or "")
            if not value: continue

            entity, attribute = "", ""

            if not entity and not attribute:
                q_lower = query.lower()
                if "population" in q_lower: attribute = "population"
                elif "born" in q_lower: attribute = "birth year"
                elif "length" in q_lower: attribute = "length"
                elif "area" in q_lower: attribute = "area"

            facets.append(Facet(facet_id=f"numeric_{len(facets)}", facet_type=FacetType.NUMERIC, template={"value": value, "unit": unit, "entity": entity, "attribute": attribute}))
        return facets

    # ---- comparison ----
    def _extract_comparison_facets(self, query: str) -> List[Facet]:
        facets: List[Facet] = []
        for k in ["more", "less", "better", "worse", "higher", "lower", "earlier", "later"]:
            pat = rf"(.+?)\s+(?:is|was|are|were)?\s*{re.escape(k)}\s+than\s+(.+?)(?:\?|$)"
            for m in re.finditer(pat, query, flags=re.IGNORECASE):
                facets.append(Facet(facet_id=f"comparison_{len(facets)}", facet_type=FacetType.COMPARISON, template={"entity1": m.group(1), "entity2": m.group(2), "attribute": k}))
        return facets

    # ---- causal / procedural ----
    def _extract_causal_facets(self, query: str) -> List[Facet]:
        facets: List[Facet] = []
        q = query.lower()

        # P2-6 FIX: Only generate CAUSAL facets for explicit causal questions
        # Strict patterns: must start with "why" or explicitly mention cause/reason
        explicit_causal_patterns = [
            r"^\s*why\b",  # Starts with "why"
            r"\bbecause\b",  # Mentions "because"
            r"\bcaused\s+by\b",  # Mentions "caused by"
            r"\bwhat\s+(?:was|is|are)\s+the\s+caus(?:e|es)\b",  # "what was the cause"
            r"\bwhat\s+led\s+to\b",  # "what led to"
            r"\bwhat\s+resulted\s+in\b",  # "what resulted in"
            r"\breason\s+for\b",  # "reason for"
        ]

        # P2-6: Only create CAUSAL facet if explicitly causal
        if any(re.search(p, q) for p in explicit_causal_patterns):
            # Create as non-required by default (required=False is default)
            facets.append(Facet(facet_id="causal_0", facet_type=FacetType.CAUSAL, template={"trigger": "causal"}, required=False))

        return facets

    def _extract_temporal_event(self, query: str, time_span: Tuple[int, int]) -> str:
        start = max(0, time_span[0] - 30)
        end = min(len(query), time_span[1] + 30)
        return query[start:end].replace(query[time_span[0]:time_span[1]], "").strip(" ,;:-")

    def _deduplicate_facets(self, facets: List[Facet]) -> List[Facet]:
        seen: Set[str] = set()
        unique: List[Facet] = []
        for f in facets:
            # P0-4: Drop facets with empty hypothesis text at mining time
            hypothesis = f.to_hypothesis()
            if not hypothesis or not hypothesis.strip():
                continue  # Skip facets with empty text

            sig = f"{f.facet_type.value}:{_clean_text(hypothesis).lower()}"
            if sig in seen: continue
            seen.add(sig)
            unique.append(f)
        return unique