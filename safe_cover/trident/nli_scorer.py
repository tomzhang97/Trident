"""NLI/Cross-encoder scoring module.
UPDATES:
- Short token fallback: If phrase is "US" (no long tokens), falls back to exact match.
- Robust normalization syntax.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
from collections import OrderedDict
import hashlib
import os
import re
import unicodedata

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from .facets import Facet, FacetType
from .candidates import Passage
from .config import NLIConfig
from .relation_schema import get_default_registry

@dataclass
class NLIScore:
    entailment_score: float
    contradiction_score: float
    neutral_score: float
    final_score: float

def _sha1(text: str) -> str:
    return hashlib.sha1(
        (text or "").encode("utf-8"), usedforsecurity=False
    ).hexdigest()

def _normalize_label(s: str) -> str:
    s = (s or "").strip().lower()
    if "entail" in s: return "entailment"
    if "contra" in s: return "contradiction"
    if "neutral" in s: return "neutral"
    return s

_WS_RE = re.compile(r"\s+")
_NON_ALNUM_RE = re.compile(r"[^0-9a-z]+")


def _normalize_text_unicode(text: str) -> str:
    """Unicode-stable normalization used for gating/anchoring.

    This removes differences between unicode variants and lowercases while
    keeping spacing deterministic so token-set comparisons are stable.
    """
    norm = unicodedata.normalize("NFKC", text or "")
    norm = _WS_RE.sub(" ", norm).strip().lower()
    return norm


def _get_token_set(norm_text: str) -> set[str]:
    return {t for t in re.split(r"[^0-9a-z]+", norm_text) if t}


def _fuzzy_phrase_match(term: str, passage_norm: str, passage_tokens: set[str]) -> bool:
    """Deterministic token/phrase match that avoids brittle substrings."""
    norm = _normalize_text_unicode(term)
    if not norm:
        return False
    tokens = _get_token_set(norm)
    if not tokens:
        return False
    if len(tokens) == 1:
        return next(iter(tokens)) in passage_tokens
    return tokens.issubset(passage_tokens) or norm in passage_norm


# Deterministic, frozen relation keyword map used for predicate gating.
# Changing this map changes the Tested-Set definition and must be versioned.
_BASE_RELATION_KEYWORDS: Dict[str, set[str]] = {
    "DIRECTOR": {"director", "directed", "filmmaker", "helmed"},
    "BORN": {"born", "birth", "birthplace", "native"},
    "AWARD": {"award", "won", "prize", "winner", "nominated"},
    "CREATED": {"created", "founded", "wrote", "written", "author", "composed"},
    "LOCATION": {"located", "capital", "situated", "based", "headquarters"},
    "MARRIAGE": {"married", "spouse", "wife", "husband", "wedding"},
    "MOTHER": {"mother", "daughter", "son", "child"},
    "FATHER": {"father", "daughter", "son", "child"},
    "PARENT": {"parent", "daughter", "son", "child"},
    "CHILD": {"parent", "daughter", "son", "child"},
    "SPOUSE": {"spouse", "married", "wife", "husband", "partner"},
    "NATIONALITY": {"nationality", "citizen", "from"},
    "BIRTHPLACE": {"birthplace", "born", "raised", "grew"},
}

RELATION_REGISTRY = get_default_registry()
RELATION_KEYWORDS: Dict[str, set[str]] = {
    **_BASE_RELATION_KEYWORDS,
    **{spec.name: spec.keyword_set() for spec in RELATION_REGISTRY.specs()},
}

# Versioned hashes for Tested-Set definition (normalizer + gate + keyword map)
NORMALIZE_VERSION = "unicode_nfkc_nfkc_ws_v1"
GATE_VERSION = f"relation_gate_schema_v2_{RELATION_REGISTRY.version}"

def _hash_relation_keywords() -> str:
    items = sorted((k, sorted(v)) for k, v in RELATION_KEYWORDS.items())
    flat = "|".join([f"{k}:{','.join(vals)}" for k, vals in items])
    return _sha1(flat)

RELATION_KEYWORDS_HASH = _sha1(f"{RELATION_REGISTRY.keyword_hash()}|{_hash_relation_keywords()}")


def _check_relation_keywords(kind: str, passage_norm: str, passage_tokens: set[str], relation_pid: Optional[str] = None) -> bool:
    # Prefer schema-backed keywords when available
    keywords = set()
    if relation_pid:
        keywords = RELATION_REGISTRY.keywords_for(relation_pid)
    if not keywords and kind:
        keywords = RELATION_REGISTRY.keywords_for(kind)
    if not keywords:
        keywords = RELATION_KEYWORDS.get((kind or "").upper(), set())
    if not keywords:
        return False
    for kw in keywords:
        if " " in kw:
            if _fuzzy_phrase_match(kw, passage_norm, passage_tokens):
                return True
        elif kw in passage_tokens:
            return True
    return False

def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "")
    # Robust normalization: standard quotes
    s = (s.replace("'", "'").replace("'", "'")
          .replace(""", '"').replace(""", '"'))
    s = s.replace(",", "")
    s = _WS_RE.sub(" ", s).strip().lower()
    return s

def _normalize_for_matching(s: str) -> str:
    """
    Robust normalizer for entity/passage matching.
    Handles hyphens, Unicode, diacritics consistently.

    "Polish-Russian War" -> "polish russian war"
    "Żuławski" -> "zulawski"
    """
    if not s:
        return ""
    s = s.lower()
    s = unicodedata.normalize("NFKD", s)
    # Strip diacritics (combining characters)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    # Convert all non-alphanumerics (hyphens, punctuation) to spaces
    s = _NON_ALNUM_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s

def _token_set(s: str) -> set:
    """Get set of normalized tokens from string."""
    s = _normalize_for_matching(s)
    return set(s.split()) if s else set()

def _entity_in_passage(entity: str, passage: str, min_tokens: int = 2) -> bool:
    """
    Check if entity tokens are present in passage using robust matching.

    For multi-token entities: require all tokens present (containment).
    For single-token entities: require exact token match.

    Handles hyphen normalization: "Polish-Russian War" matches "polish russian war".
    """
    e_tokens = _token_set(entity)
    if not e_tokens:
        return False
    p_tokens = _token_set(passage)

    # Filter out very short tokens (likely noise) for matching
    e_tokens = {t for t in e_tokens if len(t) > 1}
    if not e_tokens:
        return False

    # For multi-token entities: require containment
    if len(e_tokens) >= min_tokens:
        return e_tokens.issubset(p_tokens)

    # For single-token entities, containment is too weak; require exact token hit
    t = next(iter(e_tokens))
    return t in p_tokens

def _contains_exact_phrase(passage: str, phrase: str) -> bool:
    return _norm(phrase) in _norm(passage) and _norm(phrase) != ""

def _token_in_text(tok: str, text: str) -> bool:
    return bool(re.search(rf"(?<!\w){re.escape(tok)}(?!\w)", text))

_STOP_WORDS = {"the", "and", "of", "in", "to", "a", "an", "for", "or", "is", "are", "was", "were"}

def _contains_tokens(passage: str, phrase: str) -> bool:
    """
    Looser check: do significant tokens from phrase appear in passage?
    Hyphen/punct robust: "Polish-Russian" -> ["polish", "russian"].
    """
    import math

    if not phrase:
        return False

    p_norm = _norm(passage)
    ph_norm = _norm(phrase)

    # Split phrase into tokens robustly (handles hyphen/underscore/punct)
    raw = re.split(r"[\s\-_]+", ph_norm)
    tokens = [t for t in raw if len(t) > 2 and t not in _STOP_WORDS]

    # If no significant tokens remain, fall back to exact-phrase match
    if not tokens:
        return _contains_exact_phrase(passage, phrase)

    hits = sum(1 for t in tokens if _token_in_text(t, p_norm))
    need = max(1, math.ceil(len(tokens) / 2))
    return hits >= need

def _contains_value(passage: str, value: str) -> bool:
    val_str = _norm(str(value))
    if not val_str: return False
    pas_str = _norm(passage)
    esc_val = re.escape(val_str)
    pattern = rf"(?<![\w.]){esc_val}(?![\w.])"
    return bool(re.search(pattern, pas_str))

# WH-words and generic terms that should not be treated as real entity endpoints
_WH_WORDS = {"who", "what", "which", "when", "where", "why", "how", "whom", "whose"}
_GENERIC_TERMS = {"the", "a", "an", "of", "to", "in", "on", "for", "and", "or", "is", "are",
                  "was", "were", "be", "been", "being", "have", "has", "had", "do", "does",
                  "did", "will", "would", "could", "should", "may", "might", "must", "shall",
                  "director", "film", "movie", "actor", "actress", "person", "thing", "place",
                  "name", "title", "author", "writer", "book", "song", "album", "band", "group"}

def _clean_relation_endpoint(x: str) -> str:
    """
    Clean a RELATION endpoint by removing WH-words and generic terms.
    Returns empty string if no meaningful content remains.
    """
    if not x:
        return ""
    # Extract alphanumeric tokens
    toks = re.findall(r"[A-Za-z0-9]+", x.lower())
    # Filter out WH-words and generic terms
    cleaned = [t for t in toks if t not in _WH_WORDS and t not in _GENERIC_TERMS and len(t) > 1]
    return " ".join(cleaned).strip()

def _check_lexical_gate(facet: Facet, passage_text: str) -> Optional[bool]:
    ft = facet.facet_type
    tpl = facet.template or {}
    meta = facet.metadata or {}

    def _record_failure(reason: str):
        # Mutable metadata dict even on frozen Facet
        meta["gate_failure_reason"] = reason

    if ft == FacetType.ENTITY:
        return _contains_exact_phrase(passage_text, str(tpl.get("mention", "")))
    if ft == FacetType.NUMERIC:
        return _contains_value(passage_text, str(tpl.get("value", "")))

    # Generic BRIDGE_HOP
    if "BRIDGE_HOP" in ft.value:
        e1 = str(tpl.get("entity1", "") or tpl.get("entity", "") or "")
        e2 = str(tpl.get("bridge_entity", "") or tpl.get("entity2", "") or "")

        if not e1 or not e2:
            return None

        # Use robust entity matching with hyphen/Unicode normalization
        return _entity_in_passage(e1, passage_text) and _entity_in_passage(e2, passage_text)

    if ft == FacetType.RELATION:
        passage_norm = _normalize_text_unicode(passage_text)
        passage_tokens = _get_token_set(passage_norm)

        s_raw = str(tpl.get("subject", ""))
        o_raw = str(tpl.get("object", ""))
        predicate = str(tpl.get("predicate", ""))

        # WH-subject handling (Who / What / Where)
        is_wh_subject = bool(meta.get("is_wh_subject")) or s_raw.strip().lower() in _WH_WORDS
        is_wh_object = bool(meta.get("is_wh_object"))

        relation_pid = str(tpl.get("relation_pid", "") or "") or None

        relation_kind = (
            tpl.get("scoring_relation_kind")
            or tpl.get("outer_relation_type")
            or tpl.get("relation_kind")
        )

        # Anchor constraint: require at least one grounded endpoint
        s_clean = _clean_relation_endpoint(s_raw)
        o_clean = _clean_relation_endpoint(o_raw)
        if is_wh_subject and is_wh_object:
            has_anchor = bool(s_clean or o_clean)
        elif is_wh_subject:
            has_anchor = bool(o_clean)
        else:
            has_anchor = bool(s_clean or o_clean)

        # Reject global probes or placeholder-only facets
        # WH-subject facets intentionally drop the subject endpoint from normalization,
        # but still require an object anchor.
        if not has_anchor:
            _record_failure("no_anchor")
            return False

        placeholder_re = re.compile(r"\[[A-Z0-9_]+_RESULT\]")
        if (s_clean and placeholder_re.search(s_raw)) or (o_clean and placeholder_re.search(o_raw)):
            _record_failure("placeholder_anchor")
            return False

        o_tokens = _get_token_set(_normalize_text_unicode(o_clean)) if o_clean else set()

        s_match = _fuzzy_phrase_match(s_clean, passage_norm, passage_tokens) if s_clean else False
        o_match = _fuzzy_phrase_match(o_clean, passage_norm, passage_tokens) if o_clean else False

        # If WH subject ("Who/What/..."), relax object anchoring:
        # allow partial token overlap instead of full token subset.
        if is_wh_subject and o_clean and not o_match:
            overlap = o_tokens.intersection(passage_tokens)
            # require at least 2 tokens overlap OR 1 token if object is short
            if len(overlap) >= 2 or (len(o_tokens) <= 2 and len(overlap) >= 1):
                o_match = True

        # CRITICAL FIX: For hop-2 outer relation facets (e.g., "mother of director of Film"),
        # allow single-endpoint anchor policy where only the bound entity needs to appear.
        # This makes hop-2 certification viable at scale for compositional questions.
        is_hop2_outer = bool(tpl.get("outer_relation_type"))
        if is_hop2_outer and is_wh_subject:
            # For hop-2 WH-subject facets (e.g., "Who is the mother of Xawery?"):
            # - Object = bound entity from hop-1 (e.g., "Xawery")
            # - Subject = unknown answer (e.g., "Who")
            # Only require object anchor (bound entity), not both endpoints
            anchor_policy = "SINGLE_ENDPOINT"
        else:
            anchor_policy = str(tpl.get("anchor_policy") or "ANY").upper()
            if anchor_policy not in {"ANY", "ALL"}:
                anchor_policy = "ANY"

        anchor_ok = False
        if anchor_policy == "SINGLE_ENDPOINT":
            # For hop-2: only require the bound entity (object) to match
            anchor_ok = bool(o_match)
        elif is_wh_subject and is_wh_object:
            anchor_ok = bool(s_match or o_match)
        elif is_wh_subject:
            anchor_ok = bool(o_match)
        elif anchor_policy == "ALL":
            anchor_ok = bool(s_match and o_match)
        else:
            anchor_ok = bool(s_match or o_match)

        if not anchor_ok:
            _record_failure("anchor_not_found")
            return False

        # Predicate constraint
        predicate_required = bool(relation_kind or relation_pid or predicate)
        predicate_ok = False
        if relation_kind or relation_pid:
            predicate_ok = _check_relation_keywords(relation_kind, passage_norm, passage_tokens, relation_pid=relation_pid)

        # CRITICAL FIX: For WH-subject relations, allow inverse surface forms
        # Many KB-style relations appear in inverse surface forms in text:
        # - mother/father ↔ son/daughter/child
        # - spouse ↔ married to / husband/wife
        # - birthplace ↔ born in
        # - founder ↔ founded by / established by
        # - author ↔ written by / composed by
        # - director ↔ directed by / filmmaker
        if not predicate_ok and is_wh_subject and relation_kind:
            kind_upper = relation_kind.upper()

            # Relation-specific inverse keyword mapping
            inverse_map = {
                "MOTHER": {
                    "son of", "daughter of", "child of",
                    "is the son", "is the daughter", "is the child",
                    "was the son", "was the daughter", "was the child"
                },
                "FATHER": {
                    "son of", "daughter of", "child of",
                    "is the son", "is the daughter", "is the child",
                    "was the son", "was the daughter", "was the child"
                },
                "PARENT": {
                    "son of", "daughter of", "child of",
                    "is the son", "is the daughter", "is the child",
                    "was the son", "was the daughter", "was the child"
                },
                "SPOUSE": {
                    "married to", "married", "husband of", "wife of",
                    "is married to", "was married to",
                    "spouse of", "partner of"
                },
                "BIRTHPLACE": {
                    "born in", "born at", "native of", "birth",
                    "was born in", "was born at"
                },
                "FOUNDER": {
                    "founded by", "established by", "created by",
                    "was founded by", "was established by"
                },
                "AUTHOR": {
                    "written by", "composed by", "authored by",
                    "was written by", "was composed by"
                },
                "DIRECTOR": {
                    "directed by", "film by", "directed",
                    "was directed by"
                },
                "NATIONALITY": {
                    "is a", "was a", "from", "national of",
                    "citizen of", "born in"
                },
            }

            inverse_keywords = inverse_map.get(kind_upper, set())
            if inverse_keywords:
                for kw in inverse_keywords:
                    if _fuzzy_phrase_match(kw, passage_norm, passage_tokens):
                        predicate_ok = True
                        break

        if not predicate_ok and predicate:
            predicate_ok = _fuzzy_phrase_match(predicate, passage_norm, passage_tokens)

        if not predicate_required:
            return True

        if not predicate_ok:
            _record_failure("predicate_mismatch")
            return False

        return predicate_ok

    if ft == FacetType.COMPARISON:
        e1 = str(tpl.get("entity1", ""))
        e2 = str(tpl.get("entity2", ""))
        return _contains_tokens(passage_text, e1) or _contains_tokens(passage_text, e2)

    if ft == FacetType.TEMPORAL:
        return _contains_exact_phrase(passage_text, str(tpl.get("time", "")))

    return None

class NLIScorer:
    def __init__(self, config: NLIConfig, device: str = "cuda:0"):
        self.config = config
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)  # nosec B615
        self.model = AutoModelForSequenceClassification.from_pretrained(config.model_name).to(device)  # nosec B615
        self.model.eval()
        self.cache_size = int(getattr(config, "cache_size", 10000))
        self.use_cache = bool(getattr(config, "use_cache", True))
        self.cache = OrderedDict()
        self.entail_idx, self.neutral_idx, self.contra_idx = self._infer_label_indices()

    def _infer_label_indices(self) -> Tuple[int, int, int]:
        num_labels = int(self.model.config.num_labels)
        if num_labels == 2: return 1, 0, 0

        id2label = getattr(self.model.config, "id2label", None) or {}
        label2id = getattr(self.model.config, "label2id", None) or {}
        idx_map = {}

        if isinstance(id2label, dict):
            for idx, lab in id2label.items(): idx_map[_normalize_label(str(lab))] = int(idx)
        if len(idx_map) < 3 and isinstance(label2id, dict):
            for lab, idx in label2id.items(): idx_map[_normalize_label(str(lab))] = int(idx)

        return idx_map.get("entailment", 2), idx_map.get("neutral", 1), idx_map.get("contradiction", 0)

    def _cache_key(self, passage_text: str, hypothesis: str) -> Tuple[str, str]:
        return (_sha1(passage_text), _sha1(hypothesis))

    def _normalize_mention(self, m: str) -> str:
        """
        Normalize entity mention for lexical matching.

        Fixes issues like ", Blind Shaft" where leading punctuation breaks matching.
        """
        import re
        m = m.strip()
        # Strip leading punctuation/quotes/commas
        m = re.sub(r"^[\s\W_]+", "", m)
        # Strip trailing punctuation
        m = re.sub(r"[\s\W_]+$", "", m)
        # Collapse whitespace
        m = re.sub(r"\s+", " ", m)
        # Optionally drop leading articles
        m = re.sub(r"^(the|a|an)\s+", "", m, flags=re.IGNORECASE)
        return m.lower()

    def _entity_lexical_score(self, passage_text: str, facet: Facet) -> float:
        """
        Score ENTITY facet using lexical containment (not NLI).

        NLI with "The passage mentions X" is too permissive - any passage
        containing the string gets high entailment even if irrelevant.

        Returns:
            0.0 if entity not found
            0.5-1.0 based on presence and prominence (position, frequency)
        """
        mention = facet.template.get("mention", "") if facet.template else ""
        if not mention:
            return 0.0

        # Normalize mention (fixes ", Blind Shaft" -> "blind shaft")
        mention_norm = self._normalize_mention(mention)
        if not mention_norm:
            return 0.0

        # Normalize passage text
        text_norm = re.sub(r"\s+", " ", passage_text.lower())

        # Check for normalized phrase match
        if mention_norm not in text_norm:
            # Try with hyphen/space normalization
            mention_hyphen = re.sub(r"[-_]+", " ", mention_norm)
            text_hyphen = re.sub(r"[-_]+", " ", text_norm)
            if mention_hyphen not in text_hyphen:
                return 0.0

        # Entity is present - compute score based on prominence
        # Base score: 0.7 for presence
        score = 0.7

        # Bonus for early mention (first 20% of passage) - up to +0.15
        first_pos = text_norm.find(mention_norm)
        if first_pos >= 0:
            rel_pos = first_pos / max(len(text_norm), 1)
            if rel_pos < 0.2:
                score += 0.15 * (1 - rel_pos / 0.2)

        # Bonus for multiple mentions - up to +0.15
        count = text_norm.count(mention_norm)
        if count > 1:
            score += min(0.15, 0.05 * (count - 1))

        return min(1.0, score)

    def score(self, passage: Passage, facet: Facet) -> float:
        premise = passage.text
        debug_rel = os.environ.get("TRIDENT_DEBUG_RELATION", "0") == "1"
        debug_entity = os.environ.get("TRIDENT_DEBUG_ENTITY", "0") == "1"

        # ENTITY: Use lexical containment, not NLI
        # NLI with "The passage mentions X" is too permissive
        if facet.facet_type == FacetType.ENTITY:
            score = self._entity_lexical_score(premise, facet)
            if debug_entity:
                mention = facet.template.get("mention", "") if facet.template else ""
                print(f"[ENTITY LEXICAL] mention={mention!r} score={score:.4f}")
            return score

        hypothesis = facet.to_hypothesis(passage.text)

        if debug_rel and facet.facet_type == FacetType.RELATION:
            print(f"[RELATION HYP] {hypothesis}")

        lexical_match = _check_lexical_gate(facet, premise)
        if lexical_match is False:
            if debug_rel and facet.facet_type == FacetType.RELATION:
                print(f"[RELATION SCORE] GATED OUT -> score=0.0")
            return 0.0

        # Log winning gated passage for debugging
        if debug_rel and facet.facet_type == FacetType.RELATION:
            # Show first 150 chars of passage that passed gate
            psg_preview = premise[:150].replace('\n', ' ')
            print(f"[RELATION PASSAGE] len={len(premise)} text={psg_preview!r}...")

        if self.use_cache:
            key = self._cache_key(premise, hypothesis)
            if key in self.cache:
                self.cache.move_to_end(key)
                return self.cache[key]

        inputs = self.tokenizer(premise, hypothesis, truncation="only_first", max_length=self.config.max_length, padding=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            probs = F.softmax(self.model(**inputs).logits[0], dim=-1)

        s = float(probs[self.entail_idx].item()) - 0.5 * float(probs[self.contra_idx].item() if probs.numel() > 2 else 0.0)

        if debug_rel and facet.facet_type == FacetType.RELATION:
            entail = float(probs[self.entail_idx].item())
            contra = float(probs[self.contra_idx].item()) if probs.numel() > 2 else 0.0
            print(f"[RELATION SCORE] NLI entail={entail:.4f} contra={contra:.4f} -> score={s:.4f}")

        if self.use_cache:
            if len(self.cache) >= self.cache_size: self.cache.popitem(last=False)
            self.cache[key] = s
        return s

    def batch_score(self, pairs: List[Tuple[Passage, Facet]]) -> List[float]:
        if not pairs: return []
        scores = []
        bs = int(getattr(self.config, "batch_size", 32))
        debug_rel = os.environ.get("TRIDENT_DEBUG_RELATION", "0") == "1"
        debug_entity = os.environ.get("TRIDENT_DEBUG_ENTITY", "0") == "1"

        for i in range(0, len(pairs), bs):
            batch = pairs[i:i + bs]
            to_run_indices = []
            to_run_inputs = []
            batch_data = []
            facet_types = []  # Track facet types for debug logging

            results_map = {}

            for idx, (p, f) in enumerate(batch):
                facet_types.append(f.facet_type)

                # ENTITY: Use lexical containment, not NLI
                if f.facet_type == FacetType.ENTITY:
                    s = self._entity_lexical_score(p.text, f)
                    if debug_entity:
                        mention = f.template.get("mention", "") if f.template else ""
                        print(f"[ENTITY BATCH] idx={idx} mention={mention!r} score={s:.4f}")
                    results_map[idx] = s
                    batch_data.append(None)  # Placeholder
                    continue

                hyp = f.to_hypothesis(p.text)
                key = self._cache_key(p.text, hyp)
                batch_data.append(key)

                lexical_match = _check_lexical_gate(f, p.text)
                if lexical_match is False:
                    if debug_rel and f.facet_type == FacetType.RELATION:
                        print(f"[RELATION BATCH] idx={idx} GATED OUT -> score=0.0")
                    results_map[idx] = 0.0
                    continue

                if self.use_cache and key in self.cache:
                    self.cache.move_to_end(key)
                    results_map[idx] = self.cache[key]
                else:
                    to_run_indices.append(idx)
                    to_run_inputs.append((p.text, hyp))

            if to_run_inputs:
                premises = [x[0] for x in to_run_inputs]
                hyps = [x[1] for x in to_run_inputs]

                inputs = self.tokenizer(premises, hyps, truncation="only_first", max_length=self.config.max_length, padding=True, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    probs = F.softmax(self.model(**inputs).logits, dim=-1)

                for k, idx in enumerate(to_run_indices):
                    entail = float(probs[k, self.entail_idx].item())
                    contra = float(probs[k, self.contra_idx].item()) if probs.size(-1) > 2 else 0.0
                    s = entail - 0.5 * contra
                    results_map[idx] = s

                    if debug_rel and facet_types[idx] == FacetType.RELATION:
                        print(f"[RELATION BATCH] idx={idx} NLI entail={entail:.4f} contra={contra:.4f} -> score={s:.4f}")

                    if self.use_cache:
                        key = batch_data[idx]
                        if len(self.cache) >= self.cache_size: self.cache.popitem(last=False)
                        self.cache[key] = s

            for idx in range(len(batch)):
                scores.append(results_map[idx])

        return scores

    def score_with_details(self, passage: Passage, facet: Facet) -> NLIScore:
        premise = passage.text

        # ENTITY: Use lexical containment, not NLI
        if facet.facet_type == FacetType.ENTITY:
            score = self._entity_lexical_score(premise, facet)
            # Return synthetic NLIScore with lexical score as both entail and final
            return NLIScore(score, 0.0, 1.0 - score, score)

        hypothesis = facet.to_hypothesis(passage.text)

        lexical_match = _check_lexical_gate(facet, premise)
        if lexical_match is False:
            return NLIScore(0.0, 0.0, 1.0, 0.0)

        inputs = self.tokenizer(premise, hypothesis, truncation="only_first", max_length=self.config.max_length, padding=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            probs = F.softmax(self.model(**inputs).logits[0], dim=-1)

        entail = float(probs[self.entail_idx].item())
        neutral = float(probs[self.neutral_idx].item()) if probs.numel() > 2 else float(probs[0].item())
        contra = float(probs[self.contra_idx].item()) if probs.numel() > 2 else 0.0
        final = entail - 0.5 * contra

        return NLIScore(entail, contra, neutral, final)

    def clear_cache(self) -> None:
        self.cache.clear()

    def get_cache_stats(self) -> Dict[str, Any]:
        return {"cache_size": len(self.cache), "max_cache_size": self.cache_size}
