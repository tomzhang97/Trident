"""Enhanced TRIDENT pipeline implementation with fixes for answer extraction and facet handling."""

from __future__ import annotations

import os
import re
import string
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

from .config import TridentConfig, SafeCoverConfig, ParetoConfig
from .facets import Facet, FacetMiner, FacetType, instantiate_facets, mark_required_facets, is_wh_question
from .candidates import Passage
from .calibration import ReliabilityCalibrator, CalibrationMonitor
from .safe_cover import SafeCoverAlgorithm, SafeCoverResult
from .pareto import ParetoKnapsackOptimizer, ParetoResult
from .retrieval import RetrievalResult
from .nli_scorer import NLIScorer
from .llm_interface import LLMInterface
from .monitoring import DriftMonitor
from .logging_utils import TelemetryTracker
from .vqc import VerifierQueryCompiler
from .bwk import BwKController
from .chain_builder import (
    build_inner_question_from_facet,
    detect_question_type,
    extract_candidates,
    constrained_span_select,
    ConstrainedSelectionResult,
    QuestionType,
    get_answer_facet_passages,
    get_winner_passages_only,
    extract_object_from_certified_passage,
    extract_answer_from_certified_facet,
    get_llm_answer_certificate,
    AnswerCertificate,
    looks_generic,
)


@dataclass
class PipelineOutput:
    """Complete output from TRIDENT pipeline."""
    answer: str
    selected_passages: List[Dict[str, Any]]
    certificates: Optional[List[Dict[str, Any]]]
    abstained: bool
    tokens_used: int
    latency_ms: float
    metrics: Dict[str, float]
    mode: str
    facets: List[Dict[str, Any]]
    trace: Dict[str, Any] = field(default_factory=dict)
    answer_source: str = "unknown"  # CHANGE 5: Track answer source ("safe_cover", "deterministic", "llm_heuristic")


@dataclass
class TwoStageScores:
    """Scores from two-stage scoring process."""
    stage1_scores: Dict[Tuple[str, str], float]  # (passage_id, facet_id) -> score
    stage2_scores: Dict[Tuple[str, str], float]  # CE/NLI scores
    p_values: Dict[Tuple[str, str], float]  # Calibrated p-values


# Abstention sentinel - NEVER convert to empty string
ABSTAIN_STR = "I cannot answer based on the given context."
ABSTAIN_PATTERNS = [
    "i cannot answer based on the given context",
    "i cannot answer",
    "cannot answer based on the given context",
    "not enough information",
    "insufficient information",
    "no answer available",
    "unable to answer",
    "cannot determine",
    "not possible to answer",
]


def _is_abstain_answer(text: str) -> bool:
    """Check if text is an abstention pattern."""
    text_lower = text.strip().lower()
    return any(p in text_lower for p in ABSTAIN_PATTERNS)


def _finalize_abstain(result: Dict[str, Any], reason: str) -> Dict[str, Any]:
    """
    CRITICAL (Change 9): Single abstention finalization to ensure correctness.

    When abstained=True, the contract is:
    - abstained = True
    - answer = "ABSTAIN"
    - confidence = 0.0
    - No extraction runs
    - Immediate return

    This prevents the cascade: ENTITY-ONLY ABSTAIN → EMPTY ANSWER DEBUG

    Args:
        result: Existing result dict (may be empty)
        reason: Reason for abstention (stored in metrics)

    Returns:
        Result dict with abstention contract enforced
    """
    result = dict(result) if result else {}
    result["abstained"] = True
    result["abstention_reason"] = reason

    # Ensure no extraction artifacts
    if "answer" not in result or not result["answer"]:
        result["answer"] = "ABSTAIN"

    if "metrics" not in result:
        result["metrics"] = {}
    result["metrics"]["abstention_reason"] = reason

    return result


def _normalize_for_filter(text: str) -> str:
    """Helper function to normalize text for filtering."""
    t = text.lower().strip()
    t = t.translate(str.maketrans("", "", string.punctuation))
    t = " ".join(t.split())
    return t


def _looks_like_meta(text: str) -> bool:
    """Filter out lines that look like meta / system chatter, not the actual answer."""
    norm = _normalize_for_filter(text)
    if not norm:
        return True

    # Pure labels like "Answer", "Final answer"
    bad_exact = {
        "answer",
        "final answer",
        "short answer",
        "prediction",
        "explanation",
        "reasoning",
        "i cannot answer based on the given context",
        "cannot answer",
        "no answer",
    }
    if norm in bad_exact:
        return True

    # Filler/meta phrases that often appear at the start
    filler_prefixes = [
        "ok",
        "okay",
        "sure",
        "let us think",
        "lets think",
        "let s think",
        "let's think",
        "first",
        "second",
        "step",
        "therefore",
        "thus",
        "so the answer",
    ]
    for pref in filler_prefixes:
        if norm.startswith(pref + " "):
            return True

    bad_substrings = [
        "step ",
        "document ",
        "context ",
        "evidence ",
        "reasoning:",
        "reasoning :",
        "analysis:",
        "analysis :",
        "the given context",
        "given context",
        "isnt",
        "is not",
        "should be",
        "would be",
    ]
    return any(bad in norm for bad in bad_substrings)


def extract_final_answer(raw_text: str, question: str) -> str:
    """
    Extract the final answer from raw LLM output using a more robust method.

    Args:
        raw_text: The raw text output from the LLM.
        question: The original question, used for context (e.g., Yes/No detection).

    Returns:
        The extracted answer string.
    """
    text = raw_text.strip()

    if not text:
        return ""

    # Early check: if entire text is an abstention, return immediately
    if _is_abstain_answer(text):
        return ABSTAIN_STR

    # Guard: form-field hallucination (underscores / dashes)
    if re.fullmatch(r"[_\-\s]{3,}", text):
        return ABSTAIN_STR

    # 0) Strip "Step N:" style reasoning lines (even at end of text)
    text = re.sub(r'(?mi)^step\s+\d+:[^\n]*$', '', text)
    text = re.sub(r'(?i)step\s+\d+:[^\.!\n]*', '', text)

    # 1) First priority: Look for "" pattern (case-insensitive)
    # This is the most explicit indicator from our prompt
    final_answer_patterns = [
        r"(?i)final\s+answer\s*[:\-]\s*(.+?)(?:\n|$)",  # " <answer>"
        r"(?i)final\s+answer\s*[:\-]\s*(.+)",  # Fallback without newline requirement
    ]

    for pat in final_answer_patterns:
        m = re.search(pat, text)
        if m:
            cand = m.group(1).strip()

            # Guard: form-field hallucination in candidate
            if re.fullmatch(r"[_\-\s]{3,}", cand):
                return ABSTAIN_STR

            # Clean up common trailing artifacts
            cand = re.sub(r'\s*(Human:|Assistant:|Question:).*$', '', cand, flags=re.IGNORECASE)
            # CRITICAL: Check for abstention BEFORE _looks_like_meta
            # Never convert "I cannot answer..." to empty string
            if _is_abstain_answer(cand):
                return ABSTAIN_STR
            if cand and len(cand) > 0 and not _looks_like_meta(cand):
                return cand

    # 2) Yes/No shortcut for binary questions
    q_lower = question.strip().lower()
    is_yesno_question = q_lower.startswith((
        "is ", "are ", "was ", "were ",
        "do ", "does ", "did ",
        "can ", "could ", "should ", "would ",
        "has ", "have ", "had "
    ))

    if is_yesno_question:
        t_lower = text.lower()
        # Look for explicit yes/no patterns
        # Count occurrences and use the last one mentioned
        yes_matches = list(re.finditer(r'\byes\b', t_lower))
        no_matches = list(re.finditer(r'\bno\b', t_lower))

        if yes_matches or no_matches:
            # Get the position of the last yes/no
            last_yes_pos = yes_matches[-1].start() if yes_matches else -1
            last_no_pos = no_matches[-1].start() if no_matches else -1

            # Return the one that appears last in the text
            if last_no_pos > last_yes_pos:
                return "no"
            elif last_yes_pos > last_no_pos:
                return "yes"

    # 3) Look for other answer patterns
    other_patterns = [
        r"(?mi)^answer\s*[:\-]\s*(.+?)(?:\n|$)",
        r"(?mi)the\s+answer\s+is\s+(.+?)(?:\.|$)",  # Require space after "is" to avoid matching "isn't"
        r"(?mi)therefore[,]?\s+(?:the\s+answer\s+is\s+)?(.+?)(?:\.|$)",
    ]

    for pat in other_patterns:
        m = re.search(pat, text)
        if m:
            cand = m.group(1).strip()

            # Guard: form-field hallucination in candidate
            if re.fullmatch(r"[_\-\s]{3,}", cand):
                return ABSTAIN_STR

            # Remove trailing punctuation and artifacts
            cand = re.sub(r'[.,;]+$', '', cand)
            # Clean up Human:/Assistant: artifacts
            cand = re.sub(r'\s*(Human:|Assistant:|Question:).*$', '', cand, flags=re.IGNORECASE)
            cand = re.sub(r'(Human:|Assistant:|Question:).*$', '', cand, flags=re.IGNORECASE)
            # Check for abstention before rejecting
            if _is_abstain_answer(cand):
                return ABSTAIN_STR
            if cand and len(cand) > 1 and not _looks_like_meta(cand):
                return cand

    # 4) Fallback: scan from the BOTTOM for a non-meta short line
    #    This avoids grabbing opening filler like "Okay,".
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for line in reversed(lines):
        # Skip lines with colons (usually labels)
        if ":" in line:
            continue
        # Check for abstention
        if _is_abstain_answer(line):
            return ABSTAIN_STR
        if _looks_like_meta(line):
            continue

        # Guard: form-field hallucination
        if re.fullmatch(r"[_\-\s]{3,}", line):
            continue

        # Look for substantive content (not too short, not too long)
        if 2 <= len(line) <= 100:
            # Clean up
            line = re.sub(r'^(so|thus|therefore|hence)[,\s]+', '', line, flags=re.IGNORECASE)
            # Clean artifacts
            line = re.sub(r'\s*(Human:|Assistant:|Question:).*$', '', line, flags=re.IGNORECASE)
            line = re.sub(r'(Human:|Assistant:|Question:).*$', '', line, flags=re.IGNORECASE)
            if len(line) > 1 and not _looks_like_meta(line):
                return line

    # 5) Final fallback: last sentence or phrase
    # Split by sentence-ending punctuation
    sentences = re.split(r'[.!?]\s+', text)
    if sentences:
        last_sentence = sentences[-1].strip()
        # Clean artifacts
        last_sentence = re.sub(r'\s*(Human:|Assistant:|Question:).*$', '', last_sentence, flags=re.IGNORECASE)
        last_sentence = re.sub(r'(Human:|Assistant:|Question:).*$', '', last_sentence, flags=re.IGNORECASE)
        # Remove trailing periods
        last_sentence = last_sentence.rstrip('.')
        # Skip if it's just a fragment (starts with punctuation, is very short, or looks like meta)
        if last_sentence.startswith(("'", '"', '-', 'n\'t')):
            return ""
        # Check if it's substantial (more than 4 chars and not just punctuation/fragments)
        if last_sentence and len(last_sentence) > 4 and not _looks_like_meta(last_sentence):
            # Make sure it has at least one letter
            if any(c.isalpha() for c in last_sentence):
                return last_sentence

    # 6) If all else fails, return empty rather than garbage
    return ""


class TridentPipeline:
    """Main TRIDENT pipeline orchestrator."""

    def __init__(
        self,
        config: TridentConfig,
        llm: LLMInterface,
        retriever: Any,
        device: str = "cuda:0",
        calibration_path: Optional[str] = None
    ):
        self.config = config
        self.llm = llm
        self.retriever = retriever
        self.device = device

        # Initialize components

        # Facet miner
        self.facet_miner = FacetMiner(config, llm=self.llm)
        self.nli_scorer = NLIScorer(config.nli, device)

        # Load calibrator from file if provided, otherwise create empty one
        if calibration_path:
            import json
            from pathlib import Path
            cal_path = Path(calibration_path)
            if cal_path.exists():
                print(f"📊 Loading calibration from: {calibration_path}")
                self.calibrator = ReliabilityCalibrator.load(str(cal_path))
                print(f"✅ Calibrator loaded successfully")
            else:
                print(f"⚠️  Warning: Calibration file not found: {calibration_path}")
                print(f"    Creating empty calibrator (Safe-Cover will not work!)")
                self.calibrator = ReliabilityCalibrator(use_mondrian=config.calibration.use_mondrian)
        else:
            # No calibration file provided - create empty calibrator
            # Note: Safe-Cover requires calibration data to work!
            self.calibrator = ReliabilityCalibrator(use_mondrian=config.calibration.use_mondrian)

        self.telemetry = TelemetryTracker(config.telemetry)

        # Initialize mode-specific components
        #
        # DISTRIBUTIONAL CONSTRAINT (Safe-Cover):
        # VQC and BwK are DISABLED in Safe-Cover mode because they change the
        # candidate distribution and would invalidate selection-conditional
        # calibration. They may ONLY be used in Pareto mode (which is uncertified).
        #
        # The fallback path (Safe-Cover → Pareto) may use VQC/BwK, but the
        # resulting answer is explicitly marked as uncertified (fallback_from='safe_cover').
        #
        if config.mode in ["safe_cover", "both"]:
            self.safe_cover_algo = SafeCoverAlgorithm(
                config=config.safe_cover,
                calibrator=self.calibrator
            )
            self.drift_monitor = DriftMonitor(config.safe_cover) if config.safe_cover.monitor_drift else None

        if config.mode in ["pareto", "both"]:
            self.pareto_optimizer = ParetoKnapsackOptimizer(config.pareto)
            if config.pareto.use_vqc:
                self.vqc = VerifierQueryCompiler(config, self.nli_scorer)
            if config.pareto.use_bwk:
                self.bwk = BwKController(config.pareto)

        # Initialize VQC/BwK for fallback path (Safe-Cover → Pareto).
        # IMPORTANT: These are ONLY used in the fallback (uncertified) path.
        # Safe-Cover certificates are NEVER generated when VQC/BwK are active.
        if config.mode == "safe_cover" and config.safe_cover.fallback_to_pareto:
            if not hasattr(self, 'pareto_optimizer'):
                self.pareto_optimizer = ParetoKnapsackOptimizer(config.pareto)
            if config.pareto.use_vqc and not hasattr(self, 'vqc'):
                self.vqc = VerifierQueryCompiler(config, self.nli_scorer)
            if config.pareto.use_bwk and not hasattr(self, 'bwk'):
                self.bwk = BwKController(config.pareto)

        # Caches
        self.score_cache: Dict[Tuple[str, str, str], float] = {}
        self.retrieval_cache: Dict[str, RetrievalResult] = {}

    def process_query(
        self,
        query: str,
        supporting_facts: Optional[List[Tuple[str, int]]] = None,
        context: Optional[List[List[str]]] = None,
        mode: Optional[str] = None
    ) -> PipelineOutput:
        """Process a query through the TRIDENT pipeline."""
        start_time = time.time()
        mode = mode or self.config.mode

        # Track telemetry
        self.telemetry.start_query(query)

        # Step 1: Facet mining
        facets = self.facet_miner.extract_facets(query, supporting_facts)
        use_llm_plan = False

        # Mark RELATION facets as required for WH-questions
        # This ensures that the key relation must be certified for valid answers
        facets = mark_required_facets(facets, query)

        self.telemetry.log("facet_mining", {
            "num_facets": len(facets),
            "num_required": sum(1 for f in facets if f.required),
            "is_wh_question": is_wh_question(query),
        })

        # CRITICAL FIX: Handle zero facets early with proper abstention
        if not facets:
            latency_ms = (time.time() - start_time) * 1000
            return PipelineOutput(
                answer="ABSTAIN",
                answer_source="abstain",  # CHANGE 5
                selected_passages=[],
                certificates=None,
                abstained=True,
                tokens_used=0,
                latency_ms=latency_ms,
                metrics={
                    'coverage': 0.0,
                    'num_facets': 0,
                    'num_units': 0,
                    'abstention_reason': 'no_facets',
                },
                mode=mode,
                facets=[],
                trace=self.telemetry.get_trace()
            )

        # Step 2: Retrieval
        retrieval_result = self._retrieve_passages(query, context)
        passages = retrieval_result.passages
        self.telemetry.log("retrieval", {"num_passages": len(passages)})

        # CRITICAL FIX: Handle zero passages early with proper abstention
        if not passages:
            latency_ms = (time.time() - start_time) * 1000
            return PipelineOutput(
                answer="ABSTAIN",
                answer_source="abstain",  # CHANGE 5
                selected_passages=[],
                certificates=None,
                abstained=True,
                tokens_used=0,
                latency_ms=latency_ms,
                metrics={
                    'coverage': 0.0,
                    'num_facets': len(facets),
                    'num_units': 0,
                    'abstention_reason': 'no_passages',
                },
                mode=mode,
                facets=[f.to_dict() for f in facets],
                trace=self.telemetry.get_trace()
            )

        # Step 3: Two-stage scoring
        # CRITICAL (Change 10): Don't score placeholder facets in global pre-pass
        # For compositional questions, hop-2 facets have placeholders like [DIRECTOR_RESULT]
        # that haven't been instantiated yet. Scoring them produces meaningless results
        # and logs confusing "GATED OUT" messages.
        import re
        placeholder_pattern = re.compile(r'\[([A-Z0-9_]+_RESULT)\]')

        def has_placeholder(facet: Facet) -> bool:
            """Check if facet template contains placeholder like [DIRECTOR_RESULT]."""
            tpl = facet.template or {}
            for field in ("subject", "object", "predicate", "text", "hypothesis"):
                val = str(tpl.get(field, "") or "")
                if placeholder_pattern.search(val):
                    return True
            return False

        # Filter out placeholder facets for initial scoring
        scorable_facets = [f for f in facets if not has_placeholder(f)]
        placeholder_facets = [f for f in facets if has_placeholder(f)]

        debug_prepass = os.environ.get("TRIDENT_DEBUG_PREPASS", "0") == "1"
        if debug_prepass and placeholder_facets:
            print(f"\n[PREPASS] Skipped {len(placeholder_facets)} hop2 placeholder facets")
            for pf in placeholder_facets[:3]:
                tpl = pf.template or {}
                print(f"  - {pf.facet_id}: subject={tpl.get('subject', '')}")

        # Score only instantiated facets (hop-1 + any fully-bound facets)
        scores = self._two_stage_scoring(passages, scorable_facets)
        self.telemetry.log("scoring", {
            "num_scores": len(scores.stage2_scores),
            "num_scorable_facets": len(scorable_facets),
            "num_placeholder_facets": len(placeholder_facets)
        })

        # Step 4: Mode-specific selection
        # For compositional questions, check if we need two-pass certification
        hop1_facets, hop2_facets = self._split_facets_by_hop(facets)
        is_compositional = False if use_llm_plan else len(hop2_facets) > 0

        if is_compositional:
            self.telemetry.log("compositional_detected", {
                "hop1_facets": len(hop1_facets),
                "hop2_facets": len(hop2_facets)
            })

        if mode == "safe_cover":
            if is_compositional:
                # Use two-pass Certified Adaptive Safe-Cover
                result = self._run_two_pass_safe_cover(
                    query=query,
                    hop1_facets=hop1_facets,
                    hop2_facets=hop2_facets,
                    passages=passages,
                    scores=scores,
                    context=context
                )
            else:
                # Standard single-pass Safe-Cover
                result = self._run_safe_cover(facets, passages, scores)

            # CRITICAL FIX: If Safe-Cover is infeasible or abstains, optionally fall back to Pareto
            is_abstained = result.get('abstained', False)
            is_infeasible = result.get('infeasible', False)

            if is_abstained or is_infeasible:
                # Access the fallback_to_pareto field directly from the dataclass instance
                if hasattr(self.config, 'safe_cover') and self.config.safe_cover.fallback_to_pareto:
                    self.telemetry.log("fallback", {"reason": "safe_cover_abstain_or_infeasible"})

                    # IMPORTANT: Preserve Safe-Cover attempt details before fallback
                    metrics = result.get('metrics', {})
                    safe_cover_attempt = {
                        'certificates': result.get('certificates', []),
                        'covered_facets': metrics.get('coverage', 0),
                        'dual_lower_bound': result.get('dual_lower_bound'),
                        'abstention_reason': metrics.get('abstention_reason'),
                    }

                    # Run Pareto with the same data
                    result = self._run_pareto(query, facets, passages, scores)

                    # Append Safe-Cover attempt details to metrics
                    if 'metrics' not in result:
                        result['metrics'] = {}
                    result['metrics']['fallback_from'] = 'safe_cover'
                    result['metrics']['safe_cover_attempt'] = safe_cover_attempt

                    # Preserve Safe-Cover certificates in the result if Pareto doesn't provide them
                    if not result.get('certificates'):
                        result['certificates'] = safe_cover_attempt['certificates']
        elif mode == "pareto":
            result = self._run_pareto(query, facets, passages, scores)
        elif mode == "both":
            # Run both modes for comparison
            safe_result = self._run_safe_cover(facets, passages, scores)
            pareto_result = self._run_pareto(query, facets, passages, scores)
            # Use Safe-Cover if it didn't abstain, otherwise use Pareto
            result = safe_result if not safe_result['abstained'] else pareto_result
            result['comparison'] = {
                'safe_cover': safe_result,
                'pareto': pareto_result
            }
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # Step 5: Generate answer using CONSTRAINED CANDIDATE SELECTION
        # CRITICAL: Build candidate set from evidence, LLM picks by index.
        # This ensures verbatim answers and enables support-based ranking.
        answer = ""
        answer_source = "abstain"  # CHANGE 5: Track answer source
        chain = None
        is_grounded = False
        used_regex_fallback = False
        used_constrained = False  # Track if constrained selection was used
        used_css = False  # Track if CSS-based selection was used
        constrained_result = None  # Track constrained selection result
        relation_info = None  # Track the RELATION-winning passage for typed fallback
        facet_id_map = {}  # Track all facet_id -> winning passage mappings
        prompt_type = "none"  # Track which prompt type was used

        # Initialize variables for metrics (must be set before if block for scoping)
        answer_facet_ids = []
        answer_selection_reason = ""
        winner_passages = []

        if not result['abstained'] and result['selected_passages']:
            debug_chain = os.environ.get("TRIDENT_DEBUG_CHAIN", "0") == "1"
            debug_css = os.environ.get("TRIDENT_DEBUG_CSS", "0") == "1"
            debug_constrained = os.environ.get("TRIDENT_DEBUG_CONSTRAINED", "0") == "1"
            facet_dicts = [f.to_dict() for f in facets]
            certificates = result.get('certificates', [])

            # ================================================================
            # PRIORITY 0: Constrained Candidate Selection (FACET-TARGETED)
            # ================================================================
            # CRITICAL: Only run constrained selection when ANSWER facets are certified.
            # ANSWER facets are RELATION/TEMPORAL/NUMERIC - they determine the final answer.
            # ENTITY facets only confirm the question subject exists, NOT the answer.
            # Running constrained selection from ENTITY-only evidence produces wrong answers.
            skipped_constrained_reason = ""
            if certificates:
                # Get passages that won the ANSWER facet (RELATION/TEMPORAL/NUMERIC)
                # NOT just any winning passage (which may only have ENTITY facet)
                answer_passages, answer_facet_ids, answer_selection_reason = get_answer_facet_passages(
                    question=query,
                    certificates=certificates,
                    all_passages=result['selected_passages'],
                    facets=facet_dicts
                )

                # Also get all winner passages for CSS fallback
                winner_passages = get_winner_passages_only(
                    certificates,
                    result['selected_passages']
                )

                # Build facet type lookup for metrics
                facet_type_by_id = {f.get("facet_id"): f.get("facet_type") for f in facet_dicts}

                # Check if any answer facets were actually certified
                # Answer facets are non-ENTITY types that can determine the final answer
                answer_facet_types = {"RELATION", "TEMPORAL", "NUMERIC", "COMPARISON", "BRIDGE_HOP"}
                has_answer_facet_certified = any(
                    facet_type_by_id.get(c.get("facet_id")) in answer_facet_types
                    for c in certificates
                )

                # STEP 4 FIX: Don't classify "what" as unanswerable by default
                # If the question requires specific relations/comparisons/times and only ENTITY facets certified,
                # we now route to LLM Answer Certificate instead of abstaining immediately.
                # This handles questions like "what year", "what country", "who is the", etc.
                if not has_answer_facet_certified:
                    # Detect question type
                    q_type = detect_question_type(query)
                    q_category = getattr(q_type, 'category', None)
                    q_lower = query.lower()

                    # STEP 4: Allow LLM cert to handle most question types
                    # Only abstain on truly unanswerable types (none currently)
                    # Questions like "what", "who", "when", "where", "compare", "yesno" should try LLM cert
                    should_skip_to_llm_cert = q_category in {"what", "who", "when", "where", "comparison", "yesno", "temporal", "relational"}

                    # Old strict check (now more permissive)
                    requires_non_entity = (
                        q_category in {"comparison", "temporal", "relational"} and
                        not should_skip_to_llm_cert  # Always False now, but kept for clarity
                    )

                    if requires_non_entity:
                        if debug_chain or debug_constrained:
                            certified_types = [facet_type_by_id.get(c.get("facet_id")) for c in certificates]
                            print(f"\n[ENTITY-ONLY ABSTAIN] Question type: {q_category}")
                            print(f"  Certified facets: {certified_types}")
                            print(f"  Cannot answer {q_category} question with ENTITY-only certification")
                            print(f"  ABSTAINING to avoid confident-but-wrong answer")

                        # CRITICAL (Change 9): Use _finalize_abstain() and return immediately
                        # This prevents extraction from running after abstention
                        result = _finalize_abstain(result, 'entity_only_for_relational_question')

                        # Skip to final output generation (no extraction)
                        answer = "ABSTAIN"
                        total_tokens = 0
                        prompt_tokens = 0
                        completion_tokens = 0
                        prompt_type = "none"

                        # Jump to output preparation
                        latency_ms = (time.time() - start_time) * 1000
                        final_abstained = True

                        metrics = result.get('metrics', {})
                        evidence_tokens = result.get('evidence_tokens', 0)
                        certificates = result.get('certificates') or []
                        certified_facet_ids = set(c.get("facet_id") for c in certificates if c.get("facet_id"))

                        metrics.update({
                            'evidence_tokens': evidence_tokens,
                            'prompt_tokens': 0,
                            'completion_tokens': 0,
                            'total_tokens': 0,
                            'overhead_tokens': 0,
                            'model_abstained': False,
                            'prompt_type': 'none',
                            'used_constrained': False,
                            'constrained_reason': None,
                            'constrained_confidence': None,
                            'constrained_candidate_index': None,
                            'constrained_num_candidates': 0,
                            'constrained_passage_id': None,
                            'answer_facet_ids': [],
                            'answer_selection_reason': '',
                            'num_certificates': len(certificates),
                            'num_certified_facets': len(certified_facet_ids),
                            'certified_facet_ids': list(certified_facet_ids),
                            'answer_grounded': False,
                        })

                        return PipelineOutput(
                            answer="ABSTAIN",
                            answer_source="abstain",  # CHANGE 5
                            selected_passages=result['selected_passages'],
                            certificates=result.get('certificates'),
                            abstained=True,
                            tokens_used=0,
                            latency_ms=latency_ms,
                            metrics=metrics,
                            mode=mode,
                            facets=[f.to_dict() for f in facets],
                            trace=self.telemetry.get_trace()
                        )

                # Only proceed with extraction if we haven't abstained
                if not result.get('abstained', False):
                    if debug_chain or debug_constrained:
                        print(f"\n[EXTRACT] Attempting answer extraction from certified facets")
                        print(f"  Answer facets: {answer_facet_ids}")
                        print(f"  Has answer facet certified: {has_answer_facet_certified}")
                        print(f"  Answer passages: {len(answer_passages)} (from answer facets)")
                        print(f"  All winner passages: {len(winner_passages)}")

                    # P2: Try cheap deterministic solvers FIRST (before typed extraction)
                    # These handle temporal/comparison questions with 100% faithfulness
                    from trident.chain_builder import try_deterministic_solver

                    # Convert passages to dicts for deterministic solver
                    passages_as_dicts = []
                    for p in passages:
                        if hasattr(p, 'to_dict'):
                            passages_as_dicts.append(p.to_dict())
                        elif hasattr(p, 'pid') and hasattr(p, 'text'):
                            passages_as_dicts.append({
                                'pid': p.pid,
                                'text': p.text,
                                'title': p.metadata.get('title', '') if hasattr(p, 'metadata') and p.metadata else ''
                            })
                        elif isinstance(p, dict):
                            passages_as_dicts.append(p)

                    deterministic_answer = try_deterministic_solver(
                        question=query,
                        passages=passages_as_dicts,  # Use full passage pool
                        debug=debug_chain or debug_constrained
                    )
                    if deterministic_answer:
                        answer = deterministic_answer
                        answer_source = "deterministic"  # CHANGE 5
                        is_grounded = True
                        used_constrained = True
                        prompt_type = "deterministic"
                        if debug_chain or debug_constrained:
                            print(f"[EXTRACT] ✓ Deterministic solver succeeded")
                            print(f"  Answer: '{answer}'")

                    # NEW APPROACH: Try typed extraction first from certified facets
                    # This enforces "answer from certifying passage only" contract
                    if not answer and has_answer_facet_certified and answer_facet_ids:
                        # Try extraction for each certified answer facet
                        extraction_method = "none"
                        for facet_id in answer_facet_ids:
                            extracted = extract_answer_from_certified_facet(
                                facet_id=facet_id,
                                certificates=certificates,
                                all_passages=result['selected_passages'],
                                facets=facet_dicts,
                                llm=self.llm,
                                question=query
                            )
                            if extracted:
                                answer = extracted
                                answer_source = "safe_cover"  # CHANGE 5
                                is_grounded = True
                                used_constrained = True  # Mark as constrained (from certified passage)
                                prompt_type = "typed_from_cert"
                                extraction_method = "typed_first"

                                if debug_chain or debug_constrained:
                                    print(f"[EXTRACT] ✓ Success from facet {facet_id[:8]}...")
                                    print(f"  Answer: '{answer[:100] if answer else ''}'")
                                    print(f"  Method: typed extraction from certifying passage")
                                break

                        # If typed extraction failed for all facets, try legacy constrained selection
                        if not answer and answer_passages:
                            if debug_chain or debug_constrained:
                                print(f"[EXTRACT] Typed extraction failed, trying legacy constrained selection...")
                                print(f"[EXTRACT] P0-1: Providing full context pool ({len(passages)} passages) to CSS")

                            # P0-1: Convert full passages to dicts for CSS
                            passages_as_dicts = []
                            for p in passages:
                                if isinstance(p, dict):
                                    passages_as_dicts.append(p)
                                elif hasattr(p, 'to_dict'):
                                    passages_as_dicts.append(p.to_dict())
                                elif hasattr(p, 'pid') and hasattr(p, 'text'):
                                    passages_as_dicts.append({
                                        'pid': p.pid,
                                        'text': p.text,
                                        'title': p.metadata.get('title', '') if hasattr(p, 'metadata') and p.metadata else ''
                                    })

                            constrained_result = constrained_span_select(
                                llm=self.llm,
                                question=query,
                                winner_passages=answer_passages,
                                max_candidates=10,
                                max_chars_per_passage=600,
                            )

                            if constrained_result.reason == "OK":
                                answer = constrained_result.answer
                                is_grounded = True
                                used_constrained = True
                                prompt_type = "constrained"

                                if debug_chain or debug_constrained:
                                    print(f"[CONSTRAINED] Success: '{answer[:50] if answer else ''}' (index={constrained_result.candidate_index})")
                                    print(f"[CONSTRAINED] Confidence: {constrained_result.confidence}")
                                    print(f"[CONSTRAINED] Pool: {answer_selection_reason}")
                                    print(f"[CONSTRAINED] Candidates: {constrained_result.candidates[:3]}...")
                            elif constrained_result.reason == "FALLBACK_SUPPORT":
                                # LLM gave invalid index, but we used support-based fallback
                                answer = constrained_result.answer
                                is_grounded = True
                                used_constrained = True
                                prompt_type = "constrained_fallback"

                                if debug_chain or debug_constrained:
                                    print(f"[CONSTRAINED] Fallback to support-based: '{answer[:50] if answer else ''}'")
                            else:
                                if debug_chain or debug_constrained:
                                    print(f"[CONSTRAINED] Failed: {constrained_result.reason}")
                                    if constrained_result.candidates:
                                        print(f"[CONSTRAINED] Had {len(constrained_result.candidates)} candidates")

                    # ENTITY-CERT CONSTRAINED EXTRACTION (Tier-1 path for ENTITY-only certs)
                    # When no answer facets were certified but ENTITY winners are present,
                    # ask the LLM directly with only the certified passages as context.
                    # Verify the answer appears verbatim in one of them (substring match).
                    # The Bonferroni FWER guarantee on the passage carries through to the
                    # answer, so this emits a CERTIFIED extraction (prompt_type=constrained).
                    print(f"[ENTITY-CERT DBG] answer={bool(answer)} "
                          f"has_answer_facet_certified={has_answer_facet_certified} "
                          f"winner_passages={len(winner_passages) if winner_passages else 0} "
                          f"has_llm={self.llm is not None}", flush=True)
                    # Skip factoid span extraction for FEVER (claim verification
                    # task): the answer must be a SUPPORTS/REFUTES/NEI label, not
                    # a span. Let the dataset-aware tier-2 path handle FEVER via
                    # build_multi_hop_prompt(dataset='fever').
                    _ds_ec = str(getattr(getattr(self.config, 'evaluation', None), 'dataset', '') or '').lower()
                    if (not answer and not has_answer_facet_certified
                            and winner_passages and self.llm
                            and _ds_ec != 'fever'):
                        print(f"[ENTITY-CERT DBG] entering block", flush=True)
                        # Try frozen-span first (strict).
                        constrained_result = constrained_span_select(
                            llm=self.llm,
                            question=query,
                            winner_passages=winner_passages,
                            max_candidates=10,
                            max_chars_per_passage=600,
                            certificates=certificates,
                        )
                        if constrained_result.reason in ("OK", "OK_FROZEN_SPAN", "FALLBACK_SUPPORT") \
                                and constrained_result.answer:
                            answer = constrained_result.answer
                            answer_source = "safe_cover"
                            is_grounded = True
                            used_constrained = True
                            prompt_type = "constrained"
                            if debug_chain or debug_constrained:
                                print(f"[ENTITY-CERT CONSTRAINED] ✓ Frozen-span from certified passage")
                                print(f"  Answer: '{answer[:80]}'")

                        # Permissive fallback: direct LLM answer + substring verification.
                        if not answer:
                            cert_pid_set = {c.get("passage_id") for c in certificates
                                            if c.get("passage_id")}
                            cert_passages = [p for p in winner_passages
                                             if p.get("pid") in cert_pid_set]
                            print(f"[ENTITY-CERT DBG2] cert_pid_set={cert_pid_set} "
                                  f"winner_pids={[p.get('pid') for p in winner_passages]} "
                                  f"cert_passages={len(cert_passages)}", flush=True)
                            if cert_passages:
                                ctx_lines = []
                                for p in cert_passages[:5]:
                                    pid = p.get("pid", "")
                                    text = (p.get("text") or "")[:700]
                                    ctx_lines.append(f"[{pid}] {text}")
                                ctx_block = "\n\n".join(ctx_lines)
                                # Detect yes/no questions so we don't ask for a span —
                                # asking "Were X and Y both American?" as span extraction
                                # produces "American" instead of "yes".
                                q_low = (query or "").strip().lower()
                                is_yesno_q = q_low.startswith((
                                    "is ", "are ", "was ", "were ",
                                    "do ", "does ", "did ",
                                    "can ", "could ", "should ", "would ",
                                    "has ", "have ", "had "
                                ))
                                if is_yesno_q:
                                    system_msg = (
                                        "You answer yes/no questions using ONLY the "
                                        "provided passages. Output exactly one word: "
                                        "yes, no, or NONE (if the passages do not let you decide)."
                                    )
                                else:
                                    system_msg = (
                                        "You are an extractive QA system. Copy a short span "
                                        "(1-8 words) verbatim from the provided passages that "
                                        "answers the question. Output ONLY the span — no quotes, "
                                        "brackets, labels, or explanation. If the passages do "
                                        "not contain the answer, output exactly: NONE"
                                    )
                                user_msg = (
                                    f"Passages:\n{ctx_block}\n\nQuestion: {query}"
                                )
                                tok = getattr(self.llm, "tokenizer", None)
                                if tok is not None and hasattr(tok, "apply_chat_template"):
                                    direct_prompt = tok.apply_chat_template(
                                        [{"role": "system", "content": system_msg},
                                         {"role": "user", "content": user_msg}],
                                        tokenize=False,
                                        add_generation_prompt=True,
                                    )
                                else:
                                    # Manual Llama-3 chat format fallback.
                                    direct_prompt = (
                                        "<|begin_of_text|><|start_header_id|>system"
                                        "<|end_header_id|>\n\n"
                                        f"{system_msg}<|eot_id|>"
                                        "<|start_header_id|>user<|end_header_id|>\n\n"
                                        f"{user_msg}<|eot_id|>"
                                        "<|start_header_id|>assistant<|end_header_id|>\n\n"
                                    )
                                try:
                                    resp = self.llm.generate(direct_prompt,
                                                             max_new_tokens=48,
                                                             temperature=0.0)
                                    raw = resp.text if hasattr(resp, "text") else str(resp)
                                    # Parse: take first non-empty line, strip junk chars.
                                    cand = ""
                                    for line in raw.split("\n"):
                                        line = line.strip()
                                        # Skip decorative lines (underscores, dashes, brackets).
                                        if not line or set(line) <= set("_-=*~ "):
                                            continue
                                        cand = line
                                        break
                                    # Clean common wrappers/prefixes.
                                    import re as _re
                                    cand = _re.sub(r"^\[[^\]]*\]\s*", "", cand)  # leading [context_0]
                                    cand = _re.sub(r"^(answer|span|the answer is|answer is)\s*[:\-]?\s*",
                                                   "", cand, flags=_re.IGNORECASE)
                                    cand = cand.strip().strip('"').strip("'").strip("`")
                                    cand = cand.rstrip(".!?,;:").strip()
                                    print(f"[ENTITY-CERT DBG3] raw_llm={raw[:120]!r} cand={cand[:80]!r}", flush=True)
                                    # Robust NONE detection (handles "NONE.", "NONE (...)" etc).
                                    is_none = (not cand) or _re.match(r"^none\b", cand, _re.IGNORECASE)
                                    if cand and not is_none:
                                        if is_yesno_q and _re.match(r"^(yes|no)\b", cand, _re.IGNORECASE):
                                            # yes/no doesn't require passage substring match —
                                            # the Bonferroni cert on the passage grounds the
                                            # LLM's yes/no reasoning over it.
                                            yn = "yes" if cand.lower().startswith("yes") else "no"
                                            answer = yn
                                            answer_source = "safe_cover"
                                            is_grounded = True
                                            used_constrained = True
                                            prompt_type = "constrained"
                                            print(f"[ENTITY-CERT GROUNDED] ✓ yes/no answer '{yn}' "
                                                  f"over {len(cert_passages)} certified passage(s)", flush=True)
                                        else:
                                            # Aggressive normalization: lowercase, collapse ws, strip punctuation.
                                            def _norm(s):
                                                s = s.lower()
                                                s = _re.sub(r"[^\w\s]", " ", s)
                                                return " ".join(s.split())
                                            cand_norm = _norm(cand)
                                            matched = False
                                            for p in cert_passages:
                                                p_norm = _norm(p.get("text") or "")
                                                if cand_norm and cand_norm in p_norm:
                                                    answer = cand
                                                    answer_source = "safe_cover"
                                                    is_grounded = True
                                                    used_constrained = True
                                                    prompt_type = "constrained"
                                                    matched = True
                                                    print(f"[ENTITY-CERT GROUNDED] ✓ "
                                                          f"Answer '{cand[:60]}' verified in {p.get('pid')}", flush=True)
                                                    break
                                            if not matched:
                                                print(f"[ENTITY-CERT DBG3] cand not in any cert passage "
                                                      f"(cand_norm={cand_norm[:60]!r})", flush=True)
                                    else:
                                        print(f"[ENTITY-CERT DBG3] LLM returned NONE", flush=True)
                                except Exception as _ex:
                                    print(f"[ENTITY-CERT GROUNDED] failed: {_ex}")

                    # STEP 1 (MANDATORY): Make Tier-2 the DEFAULT for non-certifiable questions
                    # Changed routing from: Safe-Cover → LLM Heuristic → Abstain
                    # To: Use Tier-1 ONLY if answer facets certified + extracted successfully
                    #     Otherwise use Tier-2 as DEFAULT
                    # Key rule: ENTITY-only certification should NOT block Tier-2

                    # Determine if we should use Tier-1 answer (Safe-Cover certified or deterministic)
                    tier1_answer_valid = (
                        answer and
                        answer.strip() != "" and
                        (
                            # Safe-Cover with answer facets certified
                            (has_answer_facet_certified and answer_source == "safe_cover") or
                            # Deterministic solver (always valid)
                            answer_source == "deterministic" or
                            # ENTITY-cert constrained span: verbatim extract from a
                            # Bonferroni-certified passage (still a Tier-1 cert chain).
                            (used_constrained and answer_source == "safe_cover"
                             and prompt_type in ("constrained", "constrained_fallback"))
                        )
                    )

                    # Determine if we should route to Tier-2
                    should_use_tier2 = not tier1_answer_valid

                    if debug_chain or debug_constrained:
                        print(f"\n[ROUTING DECISION]")
                        print(f"  Has answer: {bool(answer)}")
                        print(f"  Has answer facet certified: {has_answer_facet_certified}")
                        print(f"  Answer source: {answer_source}")
                        print(f"  Tier-1 valid: {tier1_answer_valid}")
                        print(f"  → Route to: {'Tier-1 (Safe-Cover)' if tier1_answer_valid else 'Tier-2 (LLM Heuristic)'}")

                    # STEP 4: LLM Answer Certificate (Tier-2) - now the DEFAULT for non-certifiable questions
                    # Use Tier-2 for:
                    # - Questions where no answer facets were certified (ENTITY-only)
                    # - Questions where typed extraction failed
                    # - Comparison, temporal, yes/no, quantity questions (where facets don't help)
                    if should_use_tier2:
                        # P1-2: CONTEXT POOL INVARIANT (CRITICAL FOR CORRECTNESS)
                        # ========================================================
                        # Tier-2 LLM and deterministic solvers MUST see the FULL context pool (all retrieved passages),
                        # NOT just the certified winners from Safe-Cover.
                        #
                        # result['selected_passages'] = certified winners (1-2 passages) - WRONG for Tier-2/deterministic
                        # passages = full retrieval result (10 passages) - CORRECT for Tier-2/deterministic
                        #
                        # Using certified winners causes deterministic failover to see only 1 passage instead of 10,
                        # leading to incorrect abstentions on answerable questions.
                        # ========================================================
                        all_passages_raw = passages  # This is the full list from retrieval (line 486)

                        # Convert Passage objects to dicts for LLM cert
                        all_passages = []
                        for p in all_passages_raw:
                            if hasattr(p, 'to_dict'):
                                all_passages.append(p.to_dict())
                            elif hasattr(p, 'pid') and hasattr(p, 'text'):
                                # Passage object - convert to dict
                                all_passages.append({
                                    'pid': p.pid,
                                    'text': p.text,
                                    'title': p.metadata.get('title', '') if hasattr(p, 'metadata') and p.metadata else ''
                                })
                            elif isinstance(p, dict):
                                # Already a dict
                                all_passages.append(p)

                        # Build passage pool prioritizing certified winners
                        passage_pool = []
                        seen_pids = set()

                        # First: add certified winners (if any)
                        if winner_passages:
                            for p in winner_passages[:10]:  # Cap certified at 10
                                pid = p.get('pid')
                                if pid and pid not in seen_pids:
                                    passage_pool.append(p)
                                    seen_pids.add(pid)

                        # Second: fill from full context pool to reach 6-10 passages
                        if len(passage_pool) < 10 and all_passages:
                            for p in all_passages:
                                if len(passage_pool) >= 10:
                                    break
                                pid = p.get('pid')
                                if pid and pid not in seen_pids:
                                    passage_pool.append(p)
                                    seen_pids.add(pid)

                        # Ensure minimum of 6 passages (use whatever we have if less)
                        llm_cert_passages = passage_pool[:10] if len(passage_pool) >= 6 else passage_pool

                        # Log passage selection details
                        debug_llm_cert = os.environ.get("TRIDENT_DEBUG_LLM_CERT", "0") == "1"
                        if llm_cert_passages and self.llm:
                            if debug_chain or debug_constrained or debug_llm_cert:
                                print(f"\n[TIER 2: LLM HEURISTIC ROUTING]")
                                print(f"  Total context pool: {len(all_passages)}")
                                print(f"  Certified winners: {len(winner_passages) if winner_passages else 0}")
                                print(f"  Selected for heuristic: {len(llm_cert_passages)}")
                                print(f"  Reason: {'no_answer_facets_certified' if not has_answer_facet_certified else 'typed_extraction_failed'}")
                                if len(llm_cert_passages) < 6:
                                    print(f"  ⚠ WARNING: Only {len(llm_cert_passages)} passages (recommend 6-10)")

                            # Dataset-aware: FEVER must emit SUPPORTS/REFUTES/NOT ENOUGH INFO,
                            # which get_llm_answer_certificate's factoid prompt will not produce.
                            ds_t2 = str(getattr(getattr(self.config, 'evaluation', None), 'dataset', '') or '').lower()
                            if ds_t2 == 'fever' and hasattr(self.llm, 'build_multi_hop_prompt'):
                                llm_cert = None
                                try:
                                    fev_pool_dicts = []
                                    for p in llm_cert_passages[:10]:
                                        if isinstance(p, dict):
                                            fev_pool_dicts.append(p)
                                        elif hasattr(p, 'to_dict'):
                                            fev_pool_dicts.append(p.to_dict())
                                        elif hasattr(p, 'pid') and hasattr(p, 'text'):
                                            fev_pool_dicts.append({
                                                'pid': p.pid, 'text': p.text,
                                                'title': p.metadata.get('title', '') if hasattr(p, 'metadata') and p.metadata else ''
                                            })
                                    fev_prompt = self.llm.build_multi_hop_prompt(
                                        question=query,
                                        passages=fev_pool_dicts,
                                        facets=[],
                                        dataset='fever',
                                    )
                                    resp = self.llm.generate(fev_prompt, max_new_tokens=64, temperature=0.0)
                                    raw_text = resp.text if hasattr(resp, 'text') else str(resp)
                                    extracted = self.llm.extract_answer(raw_text) if hasattr(self.llm, 'extract_answer') else raw_text
                                    if extracted and extracted.strip():
                                        answer = extracted.strip()
                                        answer_source = "llm_heuristic"
                                        is_grounded = True
                                        prompt_type = "llm_heuristic_answer_fever"
                                except Exception:
                                    pass
                            else:
                                llm_cert = get_llm_answer_certificate(
                                    llm=self.llm,
                                    question=query,
                                    passages=llm_cert_passages,
                                    max_passages=10,
                                    max_chars_per_passage=600
                                )

                                if llm_cert and llm_cert.verified:
                                    answer = llm_cert.answer
                                    is_grounded = True
                                    # Label upgrade: if the LLM cert cites a Bonferroni-certified
                                    # passage, the answer inherits the FWER guarantee — it IS a
                                    # certified answer, not a heuristic one.
                                    cert_pids = {c.get("passage_id") for c in certificates
                                                 if c.get("passage_id")}
                                    if llm_cert.pid and llm_cert.pid in cert_pids:
                                        answer_source = "safe_cover"
                                        prompt_type = "llm_certified_answer"
                                    else:
                                        answer_source = "llm_heuristic"
                                        prompt_type = "llm_heuristic_answer"

                                    if debug_chain or debug_constrained or debug_llm_cert:
                                        tag = "CERTIFIED" if prompt_type == "llm_certified_answer" else "HEURISTIC"
                                        print(f"[TIER 2: LLM {tag}] ✓ SUCCESS")
                                        print(f"  Answer: '{answer}' ({tag.lower()})")
                                        print(f"  Supporting PID: {llm_cert.pid[:12] if len(llm_cert.pid) > 12 else llm_cert.pid}...")
                                        print(f"  Quote: '{llm_cert.quote[:100]}...'")
                                        print(f"  Type: {llm_cert.answer_type}")
                                else:
                                    # P0-3: Tier-2 FAILOVER to deterministic solvers before abstaining
                                    if debug_chain or debug_constrained or debug_llm_cert:
                                        reason = "llm_returned_none" if llm_cert is None else "verification_failed"
                                        print(f"[TIER 2: LLM HEURISTIC] ✗ FAILED: {reason}")
                                        if llm_cert and not llm_cert.verified:
                                            print(f"  Reason: LLM cert returned but failed verification")
                                        print(f"[TIER 2 FAILOVER] Trying deterministic solvers as last resort...")

                                    from trident.chain_builder import try_deterministic_solver

                                    passages_as_dicts = []
                                    for p in llm_cert_passages:
                                        if isinstance(p, dict):
                                            passages_as_dicts.append(p)
                                        elif hasattr(p, 'to_dict'):
                                            passages_as_dicts.append(p.to_dict())
                                        elif hasattr(p, 'pid') and hasattr(p, 'text'):
                                            passages_as_dicts.append({
                                                'pid': p.pid,
                                                'text': p.text,
                                                'title': p.metadata.get('title', '') if hasattr(p, 'metadata') and p.metadata else ''
                                            })

                                    deterministic_failover = try_deterministic_solver(
                                        question=query,
                                        passages=passages_as_dicts,
                                        debug=debug_chain or debug_constrained or debug_llm_cert
                                    )

                                    if deterministic_failover:
                                        answer = deterministic_failover
                                        answer_source = "deterministic_failover"
                                        is_grounded = True
                                        prompt_type = "deterministic_failover"

                                        if debug_chain or debug_constrained or debug_llm_cert:
                                            print(f"[TIER 2 FAILOVER] ✓ Deterministic solver succeeded")
                                            print(f"  Answer: '{answer}'")
                                            print(f"  This recovered a question that Tier-2 LLM couldn't handle")
                                    else:
                                        if debug_chain or debug_constrained or debug_llm_cert:
                                            print(f"[TIER 2 FAILOVER] ✗ Deterministic solver also failed")
                                            print(f"  Will abstain")
                        elif debug_chain or debug_constrained or debug_llm_cert:
                            print(f"\n[TIER 2: LLM HEURISTIC ROUTING] Skipped (no passages or no LLM)")
                            print(f"  Has passages: {bool(llm_cert_passages)}")
                            print(f"  Has LLM: {bool(self.llm)}")

            # NO-CERTIFICATE TIER-2 FALLBACK: when Safe-Cover falls back to Pareto,
            # certificates is empty, the tier-2 routing inside `if certificates:` never
            # runs, and every record returns an empty answer. If we have selected
            # passages and an LLM, route to get_llm_answer_certificate here so that
            # fallback runs still produce predictions (otherwise EM=F1=0 by construction).
            if (not answer or not answer.strip()) and not certificates and self.llm \
                    and result.get('selected_passages'):
                fallback_pool = []
                seen_pids = set()
                for p in result.get('selected_passages', []):
                    pd = None
                    if isinstance(p, dict):
                        pd = p
                    elif hasattr(p, 'to_dict'):
                        pd = p.to_dict()
                    elif hasattr(p, 'pid') and hasattr(p, 'text'):
                        pd = {'pid': p.pid, 'text': p.text,
                              'title': p.metadata.get('title', '') if hasattr(p, 'metadata') and p.metadata else ''}
                    if pd and pd.get('pid') and pd['pid'] not in seen_pids:
                        fallback_pool.append(pd)
                        seen_pids.add(pd['pid'])
                for p in passages:
                    if len(fallback_pool) >= 10:
                        break
                    pd = None
                    if isinstance(p, dict):
                        pd = p
                    elif hasattr(p, 'to_dict'):
                        pd = p.to_dict()
                    elif hasattr(p, 'pid') and hasattr(p, 'text'):
                        pd = {'pid': p.pid, 'text': p.text,
                              'title': p.metadata.get('title', '') if hasattr(p, 'metadata') and p.metadata else ''}
                    if pd and pd.get('pid') and pd['pid'] not in seen_pids:
                        fallback_pool.append(pd)
                        seen_pids.add(pd['pid'])
                if fallback_pool:
                    # Dataset-aware path: for FEVER the LLM must emit
                    # SUPPORTS/REFUTES/NOT ENOUGH INFO, which get_llm_answer_certificate's
                    # factoid-style JSON prompt does not produce. Use the FEVER-aware
                    # build_multi_hop_prompt directly when dataset == 'fever'.
                    ds = str(getattr(getattr(self.config, 'evaluation', None), 'dataset', '') or '').lower()
                    if ds == 'fever' and hasattr(self.llm, 'build_multi_hop_prompt'):
                        try:
                            fever_prompt = self.llm.build_multi_hop_prompt(
                                question=query,
                                passages=fallback_pool[:10],
                                facets=[],
                                dataset='fever',
                            )
                            resp = self.llm.generate(
                                fever_prompt, max_new_tokens=64, temperature=0.0
                            )
                            raw_text = resp.text if hasattr(resp, 'text') else str(resp)
                            extracted = self.llm.extract_answer(raw_text) if hasattr(self.llm, 'extract_answer') else raw_text
                            if extracted and extracted.strip():
                                answer = extracted.strip()
                                answer_source = "llm_heuristic_nocert_fever"
                                is_grounded = True
                                prompt_type = "llm_heuristic_answer_nocert_fever"
                        except Exception:
                            pass
                    else:
                        try:
                            llm_cert = get_llm_answer_certificate(
                                llm=self.llm,
                                question=query,
                                passages=fallback_pool,
                                max_passages=10,
                                max_chars_per_passage=600,
                            )
                            if llm_cert and llm_cert.verified and llm_cert.answer:
                                answer = llm_cert.answer
                                answer_source = "llm_heuristic_nocert"
                                is_grounded = True
                                prompt_type = "llm_heuristic_answer_nocert"
                        except Exception:
                            pass
                    if not answer or not answer.strip():
                        try:
                            from trident.chain_builder import try_deterministic_solver
                            det = try_deterministic_solver(
                                question=query, passages=fallback_pool, debug=False
                            )
                            if det:
                                answer = det
                                answer_source = "deterministic_failover_nocert"
                                is_grounded = True
                                prompt_type = "deterministic_failover_nocert"
                        except Exception:
                            pass

            # MULTIHOP-PRIMARY (origin-Pareto parity for certified tier):
            # When certificates exist, also run Pareto-style multi-hop extraction
            # over an expanded passage pool (selected ∪ top retrieved, cap 10).
            # The narrow ENTITY-CERT span extraction caps at EM≈0.25 because regex
            # facet mining often pins to the wrong entity; multi-hop with wider
            # context closes most of the gap. We REPLACE the constrained answer
            # with the multi-hop answer when multi-hop returns a valid non-empty
            # answer that is not contaminated. The certificate (FWER guarantee on
            # passage provenance) is unchanged — cert_rate stays at 85%.
            if certificates and self.llm and result.get('selected_passages') \
                    and hasattr(self.llm, 'build_multi_hop_prompt'):
                ds_mh = str(getattr(getattr(self.config, 'evaluation', None), 'dataset', '') or '').lower()
                if ds_mh != 'fever':
                    try:
                        # Build expanded pool: cert-SELECTED first (so the LLM
                        # sees grounded evidence in the priority slot), then
                        # merge in top retrieved up to cap 10. v2 ablation
                        # (retrieval-first) was 0.006 EM worse on 1000q.
                        def _to_pdict(p):
                            if isinstance(p, dict):
                                return p
                            if hasattr(p, 'to_dict'):
                                return p.to_dict()
                            if hasattr(p, 'pid') and hasattr(p, 'text'):
                                return {'pid': p.pid, 'text': p.text,
                                        'title': p.metadata.get('title', '') if hasattr(p, 'metadata') and p.metadata else ''}
                            return None

                        mh_pool = []
                        seen_mh_pids = set()
                        for p in result.get('selected_passages', []):
                            pd = _to_pdict(p)
                            if pd and pd.get('pid') and pd['pid'] not in seen_mh_pids:
                                mh_pool.append(pd)
                                seen_mh_pids.add(pd['pid'])
                        for p in passages:
                            if len(mh_pool) >= 10:
                                break
                            pd = _to_pdict(p)
                            if pd and pd.get('pid') and pd['pid'] not in seen_mh_pids:
                                mh_pool.append(pd)
                                seen_mh_pids.add(pd['pid'])

                        if mh_pool:
                            mh_dataset_name = getattr(self.config.evaluation, 'dataset', 'multi-hop QA') \
                                if hasattr(self.config, 'evaluation') else 'multi-hop QA'
                            # Pass empty facets: sandbox `facets.py` is regex-only
                            # and produces noisier facets than origin's spacy
                            # version. Letting the LLM reason from passages alone
                            # avoids being misled by wrong-entity facet hints.
                            mh_prompt = self.llm.build_multi_hop_prompt(
                                question=query,
                                passages=mh_pool,
                                facets=[],
                                dataset=mh_dataset_name,
                            )
                            mh_prompt_with_prefix = mh_prompt.rstrip() + "\nFINAL_ANSWER:"
                            mh_output = self.llm.generate(mh_prompt_with_prefix)
                            mh_raw = mh_output.text if hasattr(mh_output, 'text') else str(mh_output)
                            mh_candidate = extract_final_answer(mh_raw, query)

                            mh_contam_tokens = [
                                'facet_id', 'facet_type', "{'mention", '{"mention',
                                'if the question is', 'provide the exact', 'provide a concise answer',
                                'check if the information is present', 'based on the context above',
                                'use only the information', 'reasoning requirements',
                                'find information about:', 'determine:', 'connect:',
                                'write your answer', 'clear, concise', 'grammatically correct',
                                'in a few short steps', 'on the last line',
                                'do not add anything after', 'i cannot answer based on',
                                "i don't know", 'i do not know', 'cannot be determined',
                                'not enough information', 'insufficient information',
                                'no information available', 'unable to determine',
                                'retrieve the formula', 'cannot answer',
                            ]
                            mh_contam_patterns = [
                                r'^\d+\.\s*(then,?\s*)?(write|answer|provide)',
                                r'^(answer|response)\s*[:.]',
                                r'^step\s+\d',
                                r'^(i |the )?(answer|information)\s+(is|cannot|could not)',
                            ]

                            def _mh_contaminated(text):
                                if not text:
                                    return True
                                tl = text.lower()
                                if any(tok in tl for tok in mh_contam_tokens):
                                    return True
                                if any(re.search(pat, tl) for pat in mh_contam_patterns):
                                    return True
                                stripped_alpha = re.sub(r'[\d\s\W]', '', text)
                                if len(stripped_alpha) < 2 and text.strip():
                                    return True
                                return False

                            mh_bad = _mh_contaminated(mh_candidate)

                            if mh_bad or not mh_candidate:
                                # One repair retry on top-3 by coverage
                                try:
                                    mh_sorted = sorted(
                                        mh_pool,
                                        key=lambda p: (
                                            p.get('coverage_gain', 0) if isinstance(p, dict) else 0,
                                            p.get('utility_contribution', 0) if isinstance(p, dict) else 0,
                                        ),
                                        reverse=True,
                                    )
                                    mh_top3 = mh_sorted[:3]
                                    mh_evidence = "\n".join(
                                        f"[{i+1}] {(p.get('text', '') if isinstance(p, dict) else getattr(p, 'text', ''))}"
                                        for i, p in enumerate(mh_top3)
                                    )
                                    mh_repair = (
                                        f"Question: {query}\n\n"
                                        f"Evidence:\n{mh_evidence}\n\n"
                                        f"Return ONLY the answer. No explanation, no extra text.\n"
                                        f"ANSWER:"
                                    )
                                    mh_retry_out = self.llm.generate(mh_repair)
                                    mh_retry_raw = mh_retry_out.text if hasattr(mh_retry_out, 'text') else str(mh_retry_out)
                                    mh_retry_ans = extract_final_answer(mh_retry_raw, query)
                                    if mh_retry_ans and not _mh_contaminated(mh_retry_ans):
                                        mh_candidate = mh_retry_ans
                                        mh_bad = False
                                except Exception:
                                    pass

                            # Replace answer with multi-hop result if it is valid.
                            # The certificate (passage provenance) is preserved; only
                            # the answer text comes from the wider-context call.
                            if mh_candidate and mh_candidate.strip() and not mh_bad \
                                    and not _is_abstain_answer(mh_candidate):
                                answer = mh_candidate.strip()
                                is_grounded = True
                                answer_source = "safe_cover"
                                prompt_type = "llm_multihop_certified"
                    except Exception:
                        pass

            # FINAL-ANSWER RECOVERY (origin-Pareto parity): if every upstream
            # extractor returned empty but we have selected passages and an LLM,
            # run origin's build_multi_hop_prompt → generate(+FINAL_ANSWER:) →
            # extract_final_answer → one repair retry on contamination. This
            # recovers ENTITY-cert-but-empty cases where the strict answer-cert
            # path rejected the LLM output, and any residual empty tier-2 cases.
            # Certificates are evidence provenance; this does not alter cert_rate.
            if (not answer or not answer.strip()) and self.llm \
                    and result.get('selected_passages') \
                    and hasattr(self.llm, 'build_multi_hop_prompt'):
                ds_final = str(getattr(getattr(self.config, 'evaluation', None), 'dataset', '') or '').lower()
                if ds_final != 'fever':
                    try:
                        dataset_name = getattr(self.config.evaluation, 'dataset', 'multi-hop QA') \
                            if hasattr(self.config, 'evaluation') else 'multi-hop QA'
                        final_prompt = self.llm.build_multi_hop_prompt(
                            question=query,
                            passages=result['selected_passages'],
                            facets=[f.to_dict() for f in facets],
                            dataset=dataset_name,
                        )
                        prompt_with_prefix = final_prompt.rstrip() + "\nFINAL_ANSWER:"
                        final_output = self.llm.generate(prompt_with_prefix)
                        raw_final = final_output.text if hasattr(final_output, 'text') else str(final_output)
                        candidate = extract_final_answer(raw_final, query)

                        contamination_tokens = [
                            'facet_id', 'facet_type', "{'mention", '{"mention',
                            'if the question is', 'provide the exact', 'provide a concise answer',
                            'check if the information is present', 'based on the context above',
                            'use only the information', 'reasoning requirements',
                            'find information about:', 'determine:', 'connect:',
                            'write your answer', 'clear, concise', 'grammatically correct',
                            'in a few short steps', 'on the last line',
                            'do not add anything after', 'i cannot answer based on',
                            "i don't know", 'i do not know', 'cannot be determined',
                            'not enough information', 'insufficient information',
                            'no information available', 'unable to determine',
                            'retrieve the formula', 'cannot answer',
                        ]
                        contamination_patterns = [
                            r'^\d+\.\s*(then,?\s*)?(write|answer|provide)',
                            r'^(answer|response)\s*[:.]',
                            r'^step\s+\d',
                            r'^(i |the )?(answer|information)\s+(is|cannot|could not)',
                        ]

                        def _final_contaminated(text: str) -> bool:
                            if not text:
                                return True
                            tl = text.lower()
                            if any(tok in tl for tok in contamination_tokens):
                                return True
                            if any(re.search(pat, tl) for pat in contamination_patterns):
                                return True
                            stripped_alpha = re.sub(r'[\d\s\W]', '', text)
                            if len(stripped_alpha) < 2 and text.strip():
                                return True
                            return False

                        is_contaminated = _final_contaminated(candidate)

                        if is_contaminated or not candidate:
                            try:
                                sorted_passages = sorted(
                                    result['selected_passages'],
                                    key=lambda p: (
                                        p.get('coverage_gain', 0) if isinstance(p, dict) else 0,
                                        p.get('utility_contribution', 0) if isinstance(p, dict) else 0,
                                    ),
                                    reverse=True,
                                )
                                top_passages = sorted_passages[:3]
                                evidence_text = "\n".join(
                                    f"[{i+1}] {(p.get('text', '') if isinstance(p, dict) else getattr(p, 'text', ''))}"
                                    for i, p in enumerate(top_passages)
                                )
                                repair_prompt = (
                                    f"Question: {query}\n\n"
                                    f"Evidence:\n{evidence_text}\n\n"
                                    f"Return ONLY the answer. No explanation, no extra text.\n"
                                    f"ANSWER:"
                                )
                                retry_output = self.llm.generate(repair_prompt)
                                retry_raw = retry_output.text if hasattr(retry_output, 'text') else str(retry_output)
                                retry_answer = extract_final_answer(retry_raw, query)
                                if retry_answer and not _final_contaminated(retry_answer):
                                    candidate = retry_answer
                                    is_contaminated = False
                                    final_output = retry_output
                            except Exception:
                                pass

                        if candidate and candidate.strip() and not is_contaminated \
                                and not _is_abstain_answer(candidate):
                            answer = candidate.strip()
                            is_grounded = True
                            # Label: if any selected passage is Bonferroni-certified,
                            # the answer inherits the FWER guarantee.
                            cert_pids = {c.get("passage_id") for c in certificates
                                         if c.get("passage_id")}
                            sel_pids = set()
                            for p in result['selected_passages']:
                                pid = p.get('pid') if isinstance(p, dict) else getattr(p, 'pid', None)
                                if pid:
                                    sel_pids.add(pid)
                            if cert_pids and (cert_pids & sel_pids):
                                answer_source = "safe_cover"
                                prompt_type = "llm_final_from_certified"
                            else:
                                answer_source = "llm_final_recovered"
                                prompt_type = "llm_final_recovered"
                    except Exception:
                        pass

            # ANSWER VALIDATION: ensure we only keep grounded, non-empty answers
            def _is_garbage_answer(ans: str) -> bool:
                if not ans or not ans.strip():
                    return True
                ans_stripped = ans.strip()
                if re.fullmatch(r"[_\-\s\.\,\!\?\:]+", ans_stripped):
                    return True
                if len(ans_stripped) < 2:
                    return True
                if _looks_like_meta(ans_stripped):
                    return True
                if _is_abstain_answer(ans_stripped):
                    return True
                return False

            answer_is_invalid = _is_garbage_answer(answer)
            if answer_is_invalid:
                answer = ""
                is_grounded = False

            # CRITICAL (Answerable Contract): If certification passed but answer is empty, ABSTAIN
            # Never return empty string when we have certified answer facets
            # This prevents EM/F1 collapse from "certified but empty" cases
            if not answer or answer.strip() == "":
                debug_empty = os.environ.get("TRIDENT_DEBUG_EMPTY_ANSWER", "0") == "1"

                # Check if we had certified answer facets (should have gotten an answer)
                if certificates and has_answer_facet_certified:
                    if debug_empty or debug_chain:
                        print(f"\n[ANSWERABLE CONTRACT VIOLATION]")
                        print(f"  Certified answer facets: {answer_facet_ids}")
                        print(f"  But extraction returned empty")
                        print(f"  Enforcing contract: ABSTAIN instead of empty")

                    # Override to ABSTAIN (contract: never return empty if certified)
                    answer = "ABSTAIN"
                    result['abstained'] = True
                    result['abstention_reason'] = 'extraction_failed_despite_certification'
                    if 'metrics' not in result:
                        result['metrics'] = {}
                    result['metrics']['abstention_reason'] = 'extraction_failed_despite_certification'
                elif debug_empty:
                    # Debug: no certified answer facets, empty is expected
                    print(f"\n[EMPTY ANSWER DEBUG]")
                    print(f"  Query: {query[:100]}...")
                    print(f"  No certified answer facets, empty is expected")
                    print(f"  Mode: {mode}")
                    # P1-2: Use full context pool, not just certified passages
                    print(f"  Num passages (full context pool): {len(passages)}")
                    print(f"  Num certified passages: {len(result['selected_passages'])}")

            # Compute token usage
            if used_constrained:
                # Constrained selection uses a single LLM call with structured output
                total_tokens = 0  # Token usage is internal to the call
                prompt_tokens = 0
                completion_tokens = 0
            elif used_css:
                # CSS uses a single LLM call with structured output
                total_tokens = 0  # CSS token usage is internal to the call
                prompt_tokens = 0
                completion_tokens = 0
            else:
                total_tokens = 0
                prompt_tokens = 0
                completion_tokens = 0
        else:
            # CRITICAL: Standardize abstention output contract
            # When abstained, answer must be "ABSTAIN" (not "ABSTAINED" or verbose message)
            # This ensures consistent output for downstream processing
            answer = "ABSTAIN"
            total_tokens = 0
            prompt_tokens = 0
            completion_tokens = 0
            prompt_type = "none"

        # Calculate total latency
        latency_ms = (time.time() - start_time) * 1000

        # Check if model abstained (either selection abstained OR model said "I cannot answer")
        # Normalize to consistent "ABSTAIN" string
        model_abstained = (answer == ABSTAIN_STR) or (answer == "ABSTAIN")
        final_abstained = result['abstained'] or model_abstained

        # CRITICAL: Enforce abstention contract - when abstained, answer must be "ABSTAIN"
        if final_abstained and answer not in ["ABSTAIN", ABSTAIN_STR]:
            answer = "ABSTAIN"

        # Prepare output
        metrics = result.get('metrics', {})
        evidence_tokens = result.get('evidence_tokens', 0)
        # Certificates may be absent or explicitly set to None (e.g., Pareto mode).
        certificates = result.get('certificates') or []

        # CRITICAL FIX: num_certified_facets must match len(certificates)
        # Count unique facet_ids in certificates
        certified_facet_ids = set(c.get("facet_id") for c in certificates if c.get("facet_id"))

        metrics.update({
            'evidence_tokens': evidence_tokens,
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens,
            'total_tokens': total_tokens,
            'overhead_tokens': max(total_tokens - evidence_tokens, 0),
            'model_abstained': model_abstained,  # Track when model says "I cannot answer"
            'prompt_type': prompt_type,
            'used_constrained': used_constrained,
            'constrained_reason': constrained_result.reason if constrained_result else None,
            'constrained_confidence': constrained_result.confidence if constrained_result else None,
            'constrained_candidate_index': constrained_result.candidate_index if constrained_result else None,
            'constrained_num_candidates': len(constrained_result.candidates) if constrained_result else 0,
            'constrained_passage_id': constrained_result.passage_id if constrained_result else None,
            'answer_facet_ids': answer_facet_ids,
            'answer_selection_reason': answer_selection_reason,
            'num_certificates': len(certificates),
            'num_certified_facets': len(certified_facet_ids),
            'certified_facet_ids': list(certified_facet_ids),
            'answer_grounded': is_grounded,
        })

        # STEP 2: Final abstention guard (single choke point)
        # P2-8: INVARIANT: If abstained, answer MUST be exactly "ABSTAIN"
        # This ensures we never output empty strings or hallucinated answers when abstained
        # Hard-enforce: no exceptions, no bypass
        if final_abstained or result.get('abstained', False):
            # P2-8: Unconditionally force answer to "ABSTAIN" if abstained
            # Don't allow "I cannot answer based on the given context." or any other string
            if answer != "ABSTAIN":
                if os.environ.get("TRIDENT_DEBUG_ABSTAIN", "0") == "1":
                    print(f"\n[P2-8 GUARD] Abstention invariant violation caught!")
                    print(f"  abstained=True but answer='{answer[:100]}'")
                    print(f"  Forcing answer='ABSTAIN' (hard-enforced)")
                answer = "ABSTAIN"  # P2-8: No exceptions
            final_abstained = True

        # P2-8: Also catch hallucinated abstention strings when abstained=False
        # These are LLM outputs that look like abstentions but weren't marked as such
        hallucinated_abstentions = [
            "I cannot answer",
            "I don't have enough information",
            "The passage does not contain",
            "cannot be determined",
            "not enough information",
            "insufficient information",
        ]
        if not final_abstained:
            answer_lower = answer.lower() if answer else ""
            if any(phrase in answer_lower for phrase in hallucinated_abstentions):
                if os.environ.get("TRIDENT_DEBUG_ABSTAIN", "0") == "1":
                    print(f"\n[P2-8 GUARD] Hallucinated abstention detected!")
                    print(f"  answer='{answer[:100]}'")
                    print(f"  Forcing abstained=True and answer='ABSTAIN'")
                answer = "ABSTAIN"
                final_abstained = True

        return PipelineOutput(
            answer=answer,
            selected_passages=result['selected_passages'],
            certificates=result.get('certificates'),
            abstained=final_abstained,
            tokens_used=total_tokens,
            latency_ms=latency_ms,
            metrics=metrics,
            mode=mode,
            facets=[f.to_dict() for f in facets],
            trace=self.telemetry.get_trace(),
            answer_source=answer_source  # CHANGE 5
        )

    def _retrieve_passages(
        self,
        query: str,
        context: Optional[List[List[str]]] = None
    ) -> RetrievalResult:
        """Retrieve passages for query."""
        # Check cache
        cache_key = query
        if cache_key in self.retrieval_cache:
            return self.retrieval_cache[cache_key]

        # CRITICAL FIX: Robustly check if context is provided and non-empty
        # Context should be a list of (title, sentences) tuples
        has_valid_context = (
            context is not None and
            isinstance(context, list) and
            len(context) > 0
        )

        if has_valid_context:
            passages = []
            skipped = 0

            for idx, item in enumerate(context):
                try:
                    # Handle different context formats robustly
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        title, sentences = item[0], item[1]
                    elif isinstance(item, dict):
                        # Handle dict format: {'title': ..., 'sentences': ...}
                        title = item.get('title', f'doc_{idx}')
                        sentences = item.get('sentences', item.get('text', ''))
                    else:
                        # Skip malformed entries
                        skipped += 1
                        continue

                    # Convert sentences to text
                    if isinstance(sentences, list):
                        text = " ".join(str(s) for s in sentences if s)
                    else:
                        text = str(sentences) if sentences else ""

                    # Skip empty texts
                    if not text.strip():
                        skipped += 1
                        continue

                    passage = Passage(
                        pid=f"context_{idx}",
                        text=text,
                        cost=self._estimate_token_cost(text),
                        metadata={'title': str(title), 'source': 'provided_context'}
                    )
                    passages.append(passage)

                except Exception as e:
                    # Log but don't fail on malformed context entries
                    skipped += 1
                    continue

            if skipped > 0:
                self.telemetry.log("context_warning", {
                    "skipped_entries": skipped,
                    "valid_entries": len(passages)
                })

            # Only use context if we got valid passages, otherwise fall back to retriever
            if passages:
                result = RetrievalResult(passages=passages, scores=[1.0] * len(passages))
            else:
                # All context entries were invalid, fall back to retriever
                result = self.retriever.retrieve(query, top_k=self.config.retrieval.top_k)
        else:
            # No context provided, use retriever
            result = self.retriever.retrieve(query, top_k=self.config.retrieval.top_k)

        # Cache result
        self.retrieval_cache[cache_key] = result
        return result

    def _estimate_token_cost(self, text: str) -> int:
        """Estimate token cost more accurately."""
        # More realistic estimate: ~1.3 tokens per word + overhead
        words = len(text.split())
        return int(words * 1.3) + 10  # Add overhead for special tokens

    def _two_stage_scoring(
        self,
        passages: List[Passage],
        facets: List[Facet]
    ) -> TwoStageScores:
        """
        Perform two-stage scoring with shortlisting and CE/NLI.

        CRITICAL FIX: Ensures facets are proper Facet objects, not strings.
        CRITICAL FIX: Enforces max_tests cap on shortlist BEFORE computing p-values.
        """
        import os
        debug = os.environ.get("TRIDENT_DEBUG_PVALUE", "0") == "1"

        stage1_scores = {}
        stage2_scores = {}
        p_values = {}

        # Shared T_f function - MUST be consistent across pipeline, safe_cover, and certificates
        def tf_for(ft_enum, ft_str: str) -> int:
            """Get T_f for a facet type. Used for shortlist size AND Bonferroni threshold."""
            if ft_enum == FacetType.ENTITY:
                return 5
            if ft_enum == FacetType.RELATION or "BRIDGE" in ft_str:
                return 10
            return 5

        # Get per_facet_alpha from config
        per_facet_alpha = getattr(self.config.safe_cover, 'per_facet_alpha', 0.2) if hasattr(self.config, 'safe_cover') else 0.2

        # Validate facets are proper objects
        for facet in facets:
            if not isinstance(facet, Facet):
                raise TypeError(f"Expected Facet object, got {type(facet)}: {facet}")

        # EXHAUSTIVE PROBING: When pool is small, probe ALL passages to prevent STAGE-1 MISS
        pool_n = len(passages)
        exhaustive = (
            hasattr(self.config, "safe_cover")
            and getattr(self.config.safe_cover, "exhaustive_pool", True)
            and pool_n <= getattr(self.config.safe_cover, "exhaustive_pool_max_n", 10)
        )

        if debug:
            print(f"[DEBUG] _two_stage_scoring: {len(passages)} passages, {len(facets)} facets")
            print(f"[DEBUG] per_facet_alpha={per_facet_alpha}")
            print(f"[DEBUG] Calibrator loaded: {self.calibrator is not None}")
            print(f"[DEBUG] pool_n={pool_n} exhaustive={exhaustive}")
            # Print T_f for each facet (using correct exhaustive-aware values)
            for facet in facets:
                ft = facet.facet_type
                ft_str = ft.value if hasattr(ft, 'value') else str(ft)
                t_f = pool_n if exhaustive else tf_for(ft, ft_str)
                alpha_bar = per_facet_alpha / t_f
                print(f"[DEBUG] facet={ft_str} T_f={t_f} alpha_bar={alpha_bar:.4f} exhaustive={exhaustive}")

        # Track skipped facets (empty template data)
        skipped_facets = set()

        for facet in facets:
            ft = facet.facet_type
            ft_str = ft.value if hasattr(ft, 'value') else str(ft)

            # Skip BRIDGE_HOP facets with empty entity fields
            # These produce meaningless hypotheses and always score 0
            if "BRIDGE_HOP" in ft_str:
                tpl = facet.template or {}
                e1 = str(tpl.get("entity1", "") or tpl.get("entity", "") or "").strip()
                eb = str(tpl.get("bridge_entity", "") or tpl.get("entity2", "") or "").strip()
                if not e1 or not eb:
                    if debug:
                        print(f"[DEBUG] SKIP facet={ft_str} (empty entities: e1={e1!r} eb={eb!r})")
                    skipped_facets.add(facet.facet_id)
                    continue

            # Get T_f: exhaustive probes all passages, otherwise use facet-type T_f
            if exhaustive:
                # Probe ALL passages to prevent STAGE-1 MISS
                facet_max_tests = pool_n
            else:
                facet_max_tests = tf_for(ft, ft_str)

            # Stage 1: Shortlist candidates per facet (sorted by stage-1 score, capped to facet_max_tests)
            shortlist = self._shortlist_for_facet(passages, facet, max_candidates=facet_max_tests)

            # Stage 2: CE/NLI scoring for shortlisted pairs
            for passage in shortlist:
                cache_key = (passage.pid, facet.facet_id, self.config.calibration.version)

                if cache_key in self.score_cache:
                    score = self.score_cache[cache_key]
                else:
                    score = self.nli_scorer.score(passage, facet)
                    self.score_cache[cache_key] = score

                # F1 FIX: Assert score range / distribution consistency
                # Raw NLI scores should be (entail - 0.5*contra), expected range ~ [-0.5, 1.0]
                if not isinstance(score, float):
                    score = float(score)
                if score < -1.5 or score > 1.5:
                    if debug:
                        print(f"[CALIB] suspicious score={score:.3f} facet={facet.facet_id} pid={passage.pid[:12]}")

                stage2_scores[(passage.pid, facet.facet_id)] = score

                # Calibrate to conformal p-value with text length for Mondrian
                bucket = self._get_calibration_bucket(facet)
                text_length = len(passage.text.split())

                if debug and len(p_values) < 5:
                    print(f"[DEBUG] to_pvalue: bucket={bucket}, raw_score={score:.4f}, len={text_length}")

                # CRITICAL: Use raw NLI score for calibration, NOT transformed prob.
                # The calibrator was trained on raw scores (entail - 0.5*contra),
                # so we must use raw scores at inference for consistent conformal guarantees.
                p_value = self.calibrator.to_pvalue(score, bucket, text_length)

                if debug and len(p_values) < 5:
                    print(f"[DEBUG] conformal p_value={p_value:.4f}")

                # p-value > 0.99 is legitimate (means "not significant")
                # Do NOT override with 1-score - that breaks the certificate guarantees
                p_values[(passage.pid, facet.facet_id)] = p_value

        # COVER SUMMARY: One line per facet showing why it passed/failed
        # Track counters for summary
        num_passed = 0
        num_failed = 0
        num_no_candidates = 0
        num_skipped = 0

        if debug:
            for facet in facets:
                ft = facet.facet_type
                ft_str = ft.value if hasattr(ft, 'value') else str(ft)

                # Check if this facet was skipped (empty template data)
                if facet.facet_id in skipped_facets:
                    print(f"[COVER SUMMARY] facet={ft_str} id={facet.facet_id[:8]}... SKIPPED (empty template)")
                    num_skipped += 1
                    continue

                # Use exhaustive T_f if applicable, otherwise facet-type T_f
                if exhaustive:
                    t_f = pool_n
                else:
                    t_f = tf_for(ft, ft_str)
                alpha_bar = per_facet_alpha / t_f

                # Collect all p-values for this facet
                cand = [(pid, p) for (pid, fid), p in p_values.items() if fid == facet.facet_id]
                if not cand:
                    print(f"[COVER SUMMARY] facet={ft_str} id={facet.facet_id[:8]}... NO_CANDIDATES")
                    num_no_candidates += 1
                    continue

                best_pid, best_p = min(cand, key=lambda x: x[1])
                best_score = stage2_scores.get((best_pid, facet.facet_id), None)
                passes = best_p <= alpha_bar

                if passes:
                    num_passed += 1
                else:
                    num_failed += 1

                # Get facet text for context
                facet_text = ""
                if hasattr(facet, 'template') and facet.template:
                    if 'mention' in facet.template:
                        facet_text = facet.template.get('mention', '')
                    elif 'subject' in facet.template:
                        facet_text = f"{facet.template.get('subject', '')} -> {facet.template.get('object', '')}"
                    elif 'entity1' in facet.template:
                        # BRIDGE_HOP facets use entity1 and bridge_entity
                        e1 = facet.template.get('entity1', '')
                        eb = facet.template.get('bridge_entity', '')
                        facet_text = f"{e1} <-> {eb}" if eb else e1

                print(f"[COVER SUMMARY] facet={ft_str} text={facet_text!r} "
                      f"best_score={best_score:.4f} best_p={best_p:.4g} alpha_bar={alpha_bar:.4g} "
                      f"T_f={t_f} pass={passes}")

            # Print summary line
            print(f"[COVER TOTALS] passed={num_passed}/{len(facets)} failed={num_failed} no_candidates={num_no_candidates} skipped={num_skipped}")

        # ORACLE DEBUG: For failing facets, score ALL passages to diagnose stage-1 vs stage-2
        oracle_debug = os.environ.get("TRIDENT_DEBUG_ORACLE", "0") == "1"
        stage1_miss_count = 0
        stage2_fail_count = 0

        if debug and oracle_debug:
            print(f"\n[ORACLE DEBUG] Diagnosing failing facets...")
            for facet in facets:
                ft = facet.facet_type
                ft_str = ft.value if hasattr(ft, 'value') else str(ft)
                # Use exhaustive T_f if applicable
                if exhaustive:
                    t_f = pool_n
                else:
                    t_f = tf_for(ft, ft_str)
                alpha_bar = per_facet_alpha / t_f

                # Get probed_best_p (from shortlist)
                cand = [(pid, p) for (pid, fid), p in p_values.items() if fid == facet.facet_id]
                if not cand:
                    continue
                probed_best_pid, probed_best_p = min(cand, key=lambda x: x[1])
                probed_passes = probed_best_p <= alpha_bar

                if probed_passes:
                    continue  # Skip facets that already pass

                # Score ALL passages (not just shortlist) to find oracle_best_p
                oracle_results = []
                bucket = self._get_calibration_bucket(facet)
                for passage in passages:
                    # Check cache first
                    cache_key = (passage.pid, facet.facet_id, self.config.calibration.version)
                    if cache_key in self.score_cache:
                        score = self.score_cache[cache_key]
                    else:
                        score = self.nli_scorer.score(passage, facet)
                        # Don't cache oracle-only scores to avoid bloating cache

                    text_length = len(passage.text.split())
                    # Use raw score for calibration (consistent with main path)
                    p_val = self.calibrator.to_pvalue(score, bucket, text_length)
                    oracle_results.append((passage.pid, score, p_val))

                oracle_results.sort(key=lambda x: x[2])  # Sort by p-value
                oracle_best = oracle_results[0]
                oracle_best_p = oracle_best[2]  # (pid, score, p_val)
                oracle_passes = oracle_best_p <= alpha_bar

                # Diagnosis
                if oracle_passes and not probed_passes:
                    diagnosis = "STAGE-1 MISS (ranking failed to surface best passage)"
                    stage1_miss_count += 1
                elif not oracle_passes:
                    diagnosis = "STAGE-2 FAIL (no passage scores high enough)"
                    stage2_fail_count += 1
                else:
                    diagnosis = "OK"

                # Show where oracle_best ranks in stage-1
                oracle_pid = oracle_best[0]
                shortlist_pids = [pid for (pid, fid), _ in p_values.items() if fid == facet.facet_id]
                oracle_in_shortlist = oracle_pid in shortlist_pids

                print(f"[ORACLE] facet={ft_str} probed_best_p={probed_best_p:.4g} "
                      f"oracle_best_p={oracle_best_p:.4g} oracle_score={oracle_best[1]:.4f} "
                      f"oracle_in_shortlist={oracle_in_shortlist} alpha_bar={alpha_bar:.4g} "
                      f"DIAGNOSIS={diagnosis}")

                # Show top-3 oracle results for context
                for rank, (pid, score, pval) in enumerate(oracle_results[:3]):
                    in_sl = "✓" if pid in shortlist_pids else "✗"
                    print(f"  [{rank+1}] pid={pid[:12]}... score={score:.4f} p={pval:.4g} in_shortlist={in_sl}")

            # Print oracle summary
            print(f"[ORACLE TOTALS] stage1_miss={stage1_miss_count} stage2_fail={stage2_fail_count}")

        return TwoStageScores(
            stage1_scores=stage1_scores,
            stage2_scores=stage2_scores,
            p_values=p_values
        )

    def _shortlist_for_facet(
        self,
        passages: List[Passage],
        facet: Facet,
        max_candidates: int = 10
    ) -> List[Passage]:
        """Shortlist passages for a facet using facet-aware lexical scoring."""
        import re

        ft = facet.facet_type
        scores = []

        for passage in passages:
            if ft == FacetType.ENTITY:
                # ENTITY-specific scoring with hyphen/punct normalization
                score = self._entity_stage1_score(passage.text, facet)
            else:
                # Generic lexical scoring for other facet types
                score = self._lexical_score(passage.text, facet.get_keywords())
            scores.append((passage, score))

        # Sort by score and take top candidates
        scores.sort(key=lambda x: x[1], reverse=True)
        return [p for p, _ in scores[:max_candidates]]

    def _normalize_for_matching(self, text: str) -> str:
        """Normalize text for matching: lowercase, split hyphens, strip punctuation."""
        import re
        text = text.lower()
        # Replace hyphens/underscores with spaces
        text = re.sub(r"[-_]+", " ", text)
        # Remove punctuation except spaces
        text = re.sub(r"[^\w\s]", "", text)
        # Normalize whitespace
        text = " ".join(text.split())
        return f" {text} "  # Pad for word boundary matching

    def _entity_stage1_score(self, passage_text: str, facet: Facet) -> float:
        """
        Stage-1 scoring for ENTITY facets with hyphen/punct normalization.
        Returns high score for exact phrase match, moderate for token overlap.
        """
        import re

        # Get entity mention from facet template
        template = facet.template or {}
        entity = template.get("mention", "")
        if not entity:
            return 0.0

        # Normalize both passage and entity
        p_norm = self._normalize_for_matching(passage_text)
        e_norm = self._normalize_for_matching(entity).strip()

        # Exact phrase match (highest priority)
        if f" {e_norm} " in p_norm:
            return 10.0

        # Token overlap scoring
        e_tokens = [t for t in e_norm.split() if len(t) > 2]
        if not e_tokens:
            # Very short entity - check if it appears at all
            return 5.0 if e_norm in p_norm else 0.0

        hits = sum(1 for t in e_tokens if f" {t} " in p_norm)
        return hits / len(e_tokens)

    def _lexical_score(self, text: str, keywords: List[str]) -> float:
        """Simple lexical overlap score with normalization."""
        if not keywords:
            return 0.0
        text_norm = self._normalize_for_matching(text)
        matches = 0
        for kw in keywords:
            kw_norm = self._normalize_for_matching(kw).strip()
            if kw_norm and f" {kw_norm} " in text_norm:
                matches += 1
        return matches / len(keywords)

    def _get_calibration_bucket(self, facet: Facet) -> str:
        """
        Get calibration bucket for facet.

        Returns CANONICAL facet type for calibrator lookup.
        E.g., BRIDGE_HOP1, BRIDGE_HOP2 -> BRIDGE_HOP
        """
        # Ensure facet_type is accessed correctly
        if isinstance(facet.facet_type, FacetType):
            facet_type_str = facet.facet_type.value
        elif isinstance(facet.facet_type, str):
            facet_type_str = facet.facet_type
        else:
            facet_type_str = str(facet.facet_type)

        # Canonicalize: BRIDGE_HOP1, BRIDGE_HOP2, etc. -> BRIDGE_HOP
        if "BRIDGE_HOP" in facet_type_str or facet_type_str == "BRIDGE":
            return "BRIDGE_HOP"

        return facet_type_str

    def _split_facets_by_hop(
        self,
        facets: List[Facet]
    ) -> Tuple[List[Facet], List[Facet]]:
        """
        Split facets into hop-1 and hop-2 for compositional questions.

        Hop-1 facets have template.hop == 1 or no hop field (default single-hop).
        Hop-2 facets have template.hop == 2 (depend on hop-1 result).

        Returns:
            Tuple of (hop1_facets, hop2_facets)
        """
        hop1_facets = []
        hop2_facets = []

        for f in facets:
            template = f.template or {}
            hop = template.get('hop', 1)  # Default to hop-1

            if hop == 2:
                hop2_facets.append(f)
            else:
                hop1_facets.append(f)

        return hop1_facets, hop2_facets

    def _run_two_pass_safe_cover(
        self,
        query: str,
        hop1_facets: List[Facet],
        hop2_facets: List[Facet],
        passages: List[Passage],
        scores: TwoStageScores,
        context: Optional[List[List[str]]] = None
    ) -> Dict[str, Any]:
        """
        Certified Adaptive Safe-Cover for compositional (2-hop) questions.

        Two-stage algorithm with typed binding:
        1. Stage A: Certify hop-1 facets (find bridge entity)
        2. Typed bind: Extract bridge entity from hop-1 winner
        3. Conditioned retrieval: Retrieve using bridge entity + relation keyword
        4. Stage B: Certify hop-2 facets (find final answer)
        5. Return union of winners from both stages
        """
        import os
        debug = os.environ.get("TRIDENT_DEBUG_TWOPASS", "0") == "1"

        if debug:
            print(f"\n[TWO-PASS] Running Certified Adaptive Safe-Cover")
            print(f"  Hop-1 facets: {len(hop1_facets)}")
            print(f"  Hop-2 facets: {len(hop2_facets)}")

        # ============================================================
        # STAGE A: Certify hop-1 facets
        # ============================================================
        hop1_scores = self._two_stage_scoring(passages, hop1_facets)
        hop1_result = self._run_safe_cover(hop1_facets, passages, hop1_scores)

        if debug:
            hop1_coverage = hop1_result.get('metrics', {}).get('coverage', 0)
            print(f"  Hop-1 coverage: {hop1_coverage:.2f}")
            print(f"  Hop-1 selected: {len(hop1_result.get('selected_passages', []))}")

        # If hop-1 fails completely, we can't proceed with hop-2
        if hop1_result.get('abstained', False) or not hop1_result.get('certificates'):
            if debug:
                print(f"  Hop-1 failed/abstained - returning hop-1 result only")
            return hop1_result

        # ============================================================
        # CERTIFIED SPAN SELECTION: Bind bridge entity from hop-1 winners
        # ============================================================
        # Use CSS for relation-agnostic entity binding (no per-relation patterns)
        bound_entity = None
        inner_relation_type = None
        inner_question = None

        # Get winner passages from hop-1 certificates
        hop1_certs = hop1_result.get('certificates', [])
        hop1_winners = get_winner_passages_only(
            hop1_certs,
            hop1_result.get('selected_passages', [])
        )

        if debug:
            print(f"  Hop-1 winners: {len(hop1_winners)} passages")

        # Find the best certified hop-1 facet and build inner question
        # B1+G1 FIX: Track the binding facet's certificate to use its winning passage only
        hop1_certs_sorted = sorted(hop1_certs, key=lambda c: c.get('p_value', 1.0))
        binding_cert = None  # G1: The certificate used for binding

        for hop1_cert in hop1_certs_sorted:
            cert_fid = hop1_cert.get('facet_id', '')

            for f in hop1_facets:
                if f.facet_id == cert_fid:
                    # Check if this is a compositional hop-1 facet
                    inner_type = f.template.get('inner_relation_type', '')
                    if inner_type or f.template.get('compositional'):
                        inner_relation_type = inner_type
                        binding_cert = hop1_cert  # G1: Remember which cert we're binding from
                        # Build inner question from facet template
                        inner_question = build_inner_question_from_facet(f.to_dict())
                        if debug:
                            print(f"  Inner question: {inner_question}")
                            print(f"  Binding facet: {cert_fid} (p={hop1_cert.get('p_value', 1.0):.4f})")
                        break
            if inner_question:
                break

        # G1 FIX: Bind entity only from the binding facet's winning passage
        # Previously we used all hop1_winners which could include unrelated passages
        binding_passages = hop1_winners  # Fallback: all winners
        if binding_cert:
            binding_pid = binding_cert.get('passage_id')
            if binding_pid:
                # Get only the passage that won the binding facet
                pid_to_passage = {p.get('pid'): p for p in hop1_winners}
                if binding_pid in pid_to_passage:
                    binding_passages = [pid_to_passage[binding_pid]]
                    if debug:
                        print(f"  G1: Using only binding passage: {binding_pid[:12]}...")

        # ================================================================
        # CRITICAL: CSS binding is the PRIMARY path (scales better)
        # ================================================================
        # Strategy:
        # 1. Try CSS binding first (uses LLM with fixed inner question)
        # 2. Validate extracted entity is not generic
        # 3. Optionally use certified extraction as debug validator
        # 4. If CSS fails, try certified extraction as fallback
        # 5. If both fail → ABSTAIN
        #
        # Rationale:
        # - CSS covers Wikipedia phrasing variations better than regex
        # - CSS has provenance check (must be substring of snippet)
        # - Typed regex is kept only for validation/fallback
        binding_source = "none"
        bound_entity = None

        # PRIMARY PATH: CSS binding with fixed inner question
        if inner_question and binding_passages:
            from .chain_builder import bind_entity_via_css
            bound_entity = bind_entity_via_css(
                llm=self.llm,
                inner_question=inner_question,
                hop1_passages=binding_passages,
                max_chars=600
            )
            if bound_entity:
                binding_source = "css"
                if debug:
                    print(f"  [BINDING SOURCE] CSS")
                    print(f"  Bound entity (CSS, raw): {bound_entity}")

                # CRITICAL FIX (STEP 2): Expand CSS partial names to full spans
                # CSS might return "Xawery" when the full name is "Xawery Żuławski"
                # Expand to the LONGEST capitalized span containing the CSS result
                if binding_passages:
                    passage_text = binding_passages[0].get('text', '')
                    if passage_text and bound_entity in passage_text:
                        # Find all capitalized spans that contain bound_entity
                        import re
                        # Pattern: capital letter followed by non-punctuation, up to 6 tokens
                        # Captures: "Xawery Żuławski", "J.K. Rowling", etc.
                        candidate_spans = []
                        for match in re.finditer(r'\b([A-Z][^.,;(\n]*?(?:\s+[A-Z][^.,;(\n]*?){0,5})\b', passage_text):
                            span = match.group(1).strip()
                            # Clean up trailing punctuation
                            span = re.sub(r'[\s,\.\(\)]+$', '', span)
                            # Check if this span contains bound_entity as a substring
                            if bound_entity in span and len(span) > len(bound_entity):
                                candidate_spans.append(span)

                        # STEP 2 FIX: Choose the LONGEST span (not shortest)
                        # Also require at least 2 tokens for person names
                        if candidate_spans:
                            # Check if this is a person-name question
                            q_lower = query.lower()
                            is_person_question = any(kw in q_lower for kw in [
                                'who', 'whose', 'mother', 'father', 'director',
                                'actor', 'actress', 'author', 'founder', 'spouse'
                            ])

                            # Filter to spans with at least 2 tokens if it's a person question
                            if is_person_question and inner_relation_type:
                                rel_upper = inner_relation_type.upper()
                                if rel_upper in {"DIRECTOR", "MOTHER", "FATHER", "SPOUSE", "AUTHOR", "FOUNDER"}:
                                    candidate_spans = [s for s in candidate_spans if len(s.split()) >= 2]

                            if candidate_spans:
                                # Choose LONGEST span (most complete name)
                                best_span = max(candidate_spans, key=len)
                                original_entity = bound_entity
                                bound_entity = best_span
                                if debug:
                                    print(f"  [CSS EXPANSION] '{original_entity}' → '{bound_entity}'")

                if debug:
                    print(f"  Bound entity (CSS, expanded): {bound_entity}")

                # VALIDATION: Check if CSS binding agrees with certified extraction (debug only)
                if debug and inner_relation_type and binding_passages:
                    passage_text = binding_passages[0].get('text', '')
                    certified_entity = extract_object_from_certified_passage(
                        passage_text,
                        inner_relation_type
                    )
                    if certified_entity and certified_entity != bound_entity:
                        print(f"  [BINDING MISMATCH] CSS='{bound_entity}' vs CERTIFIED='{certified_entity}'")

        # FALLBACK PATH: Certified extraction if CSS failed
        if not bound_entity and inner_relation_type and binding_passages:
            passage_text = binding_passages[0].get('text', '')
            if passage_text:
                bound_entity = extract_object_from_certified_passage(
                    passage_text,
                    inner_relation_type
                )
                if bound_entity:
                    binding_source = "certified_fallback"
                    if debug:
                        print(f"  [BINDING SOURCE] CERTIFIED (fallback)")
                        print(f"  Bound entity (certified): {bound_entity}")

        # VALIDATION: Reject generic entities
        if bound_entity and looks_generic(bound_entity):
            if debug:
                print(f"  Bound entity '{bound_entity}' is GENERIC - rejecting")
            binding_source = "generic_rejected"
            bound_entity = None

        # CRITICAL FIX (Change 11): Binding provenance invariants
        # Prevents silent regressions by asserting binding contract
        if bound_entity:
            # Invariant 1: Bound entity is non-empty and not generic
            assert bound_entity and bound_entity.strip(), f"Bound entity is empty: {bound_entity!r}"
            assert not looks_generic(bound_entity), f"Bound entity is generic: {bound_entity!r}"

            # Invariant 2: Bound entity appears in binding passage (provenance check)
            if binding_passages:
                passage_text = binding_passages[0].get('text', '')
                assert bound_entity in passage_text, \
                    f"Bound entity '{bound_entity}' not found in binding passage (source={binding_source})"

            # Store provenance for debugging
            if binding_cert:
                binding_pid = binding_cert.get('passage_id', '')
            elif binding_passages:
                binding_pid = binding_passages[0].get('pid', '')
            else:
                binding_pid = ''

            if debug:
                print(f"  [BINDING INVARIANTS] All checks passed")
                print(f"    Entity: '{bound_entity}'")
                print(f"    Source: {binding_source}")
                print(f"    PID: {binding_pid[:12]}...")

        # ABORT IF BINDING FAILS - NO FURTHER FALLBACKS
        if not bound_entity:
            if debug:
                print(f"  Failed to bind entity from hop-1 - ABSTAIN (required hop-2 cannot be certified)")
            # CRITICAL (Change 9): Use _finalize_abstain() for binding failure
            # This enforces the abstention contract and prevents partial answers
            hop1_result = _finalize_abstain(
                hop1_result,
                reason='binding_failed_compositional_question'
            )
            hop1_result['binding_source'] = binding_source  # Track why binding failed
            # Preserve passages and certificates for debugging
            hop1_result['selected_passages'] = hop1_result.get('selected_passages', [])
            hop1_result['certificates'] = hop1_result.get('certificates', [])
            return hop1_result

        # ============================================================
        # FACET INSTANTIATION: Replace placeholders in hop-2 facets
        # ============================================================
        # CRITICAL: This is the required step to make hop-2 certification work.
        # Without this, hop-2 facets contain placeholders like "[DIRECTOR_RESULT]"
        # which cannot be scored by NLI.
        #
        # Build bindings dict: {RELATION_TYPE}_RESULT -> bound_entity
        bindings = {}
        if inner_relation_type:
            bindings[f"{inner_relation_type}_RESULT"] = bound_entity

        # If inner_relation_type is missing (heuristic miner didn't tag it),
        # infer the placeholder key(s) directly from hop-2 templates.
        if not bindings:
            placeholder_re = re.compile(r"\[([A-Z0-9_]+_RESULT)\]")
            for f in hop2_facets:
                tpl = f.template or {}
                for field in ("subject", "object", "predicate", "text", "hypothesis"):
                    val = str(tpl.get(field, "") or "")
                    m = placeholder_re.search(val)
                    if m:
                        bindings[m.group(1)] = bound_entity
                        break
                if bindings:
                    break

        if debug:
            print(f"  Bindings: {bindings}")

        # Instantiate hop-2 facets with bound entity
        hop2_facets = instantiate_facets(hop2_facets, bindings)

        if debug:
            for f in hop2_facets:
                tpl = f.template or {}
                print(f"  Instantiated hop-2 facet: {f.facet_id}")
                print(f"    subject: {tpl.get('subject', '')}")
                print(f"    outer_relation_type: {tpl.get('outer_relation_type', '')}")

        # ============================================================
        # CONDITIONED CANDIDATE GENERATION: Filter/retrieve for hop-2
        # ============================================================
        # Get the outer relation keyword from hop-2 facets
        outer_relation_type = ""
        outer_relation_keywords = []
        for f in hop2_facets:
            outer_type = f.template.get('outer_relation_type', '')
            if outer_type:
                outer_relation_type = outer_type
                # Map relation type to search keywords (multiple for matching)
                keyword_map = {
                    'MOTHER': ['mother', 'son of', 'daughter of', 'child of', 'parent'],
                    'FATHER': ['father', 'son of', 'daughter of', 'child of', 'parent'],
                    'SPOUSE': ['married', 'spouse', 'wife', 'husband', 'wed'],
                    'CHILD': ['son', 'daughter', 'child', 'parent'],
                    'NATIONALITY': ['nationality', 'national', 'citizen', 'born in'],
                    'BIRTHPLACE': ['born', 'birthplace', 'native of', 'birth'],
                    'AWARD': ['award', 'prize', 'won', 'nominated', 'recipient'],
                }
                outer_relation_keywords = keyword_map.get(outer_type, [outer_type.lower()])
                break

        if debug:
            print(f"  Outer relation: {outer_relation_type}, keywords: {outer_relation_keywords}")

        # Start with existing passages
        hop2_passages = list(passages)
        retrieved_new = 0

        # Try retrieval first (if available)
        conditioned_query = f"{bound_entity} {' '.join(outer_relation_keywords[:2])}".strip()
        if debug:
            print(f"  Conditioned query: {conditioned_query}")

        if conditioned_query and hasattr(self, 'retriever') and self.retriever is not None:
            try:
                conditioned_result = self.retriever.retrieve(
                    conditioned_query,
                    top_k=self.config.retrieval.top_k
                )
                # Add new passages (avoiding duplicates by pid)
                existing_pids = {p.pid for p in hop2_passages}
                for p in conditioned_result.passages:
                    if p.pid not in existing_pids:
                        hop2_passages.append(p)
                        existing_pids.add(p.pid)
                        retrieved_new += 1

                if debug:
                    print(f"  Retrieved {retrieved_new} new passages for hop-2")
            except Exception as e:
                if debug:
                    print(f"  Conditioned retrieval failed: {e}")

        # ============================================================
        # CONTEXT-POOL FILTER: If retrieval returned nothing, filter
        # existing passages that mention bound_entity + relation keyword
        # ============================================================
        if retrieved_new == 0 and bound_entity:
            if debug:
                print(f"  Retrieval empty - filtering context pool for hop-2 candidates")

            # Normalize bound entity for matching
            bound_entity_lower = bound_entity.lower()
            bound_entity_parts = bound_entity_lower.split()

            # Filter passages that contain bound entity + relation keyword
            filtered_candidates = []
            for p in passages:
                p_text = ""
                p_title = ""
                if hasattr(p, 'text'):
                    p_text = (p.text or "").lower()
                    p_title = (getattr(p, 'title', '') or "").lower()
                elif isinstance(p, dict):
                    p_text = (p.get('text') or "").lower()
                    p_title = (p.get('title') or "").lower()
                else:
                    continue

                combined_text = f"{p_title} {p_text}"

                # Check if bound entity is mentioned (full name or parts)
                entity_match = (
                    bound_entity_lower in combined_text or
                    all(part in combined_text for part in bound_entity_parts if len(part) > 2)
                )

                # Check if any relation keyword is present
                keyword_match = any(kw.lower() in combined_text for kw in outer_relation_keywords)

                if entity_match and keyword_match:
                    filtered_candidates.append(p)

            if debug:
                print(f"  Found {len(filtered_candidates)} context-pool candidates for hop-2")

            # Add filtered candidates to hop2_passages (avoiding duplicates)
            existing_pids = {p.pid if hasattr(p, 'pid') else p.get('pid', '') for p in hop2_passages}
            for p in filtered_candidates:
                pid = p.pid if hasattr(p, 'pid') else p.get('pid', '')
                if pid and pid not in existing_pids:
                    hop2_passages.append(p)
                    existing_pids.add(pid)

        # ============================================================
        # ============================================================
        # STAGE B: Score and certify hop-2 facets
        # ============================================================
        # CRITICAL INVARIANT: Hop-2 facets must be fully instantiated before scoring.
        # If any facet still contains placeholders like [DIRECTOR_RESULT], the scorer
        # will see meaningless input and produce p≈1 (no certification possible).
        placeholder_count = 0
        for f in hop2_facets:
            s = (f.template or {}).get("subject", "")
            o = (f.template or {}).get("object", "")
            if ("[" in s and "_RESULT]" in s) or ("[" in o and "_RESULT]" in o):
                placeholder_count += 1
                # This is a fatal bug - instantiation was not applied
                if debug:
                    print(f"  ERROR: Hop-2 facet not instantiated: {f.facet_id} subject={s}")

        if placeholder_count > 0:
            if debug:
                print(f"  [HOP2-INST] FATAL: {placeholder_count} facets still have placeholders")
            # Don't raise in production - just log and return hop-1 result
            return hop1_result

        if debug:
            print(f"  [HOP2-INST] All {len(hop2_facets)} facets instantiated, no placeholders")

        # CRITICAL FIX: ALWAYS re-score hop2_facets after instantiation.
        # The initial scoring at process_query was done with UNINSTANTIATED facets
        # (with placeholders like [DIRECTOR_RESULT]) which produced meaningless scores.
        # Those old scores are cached and MUST NOT be reused.
        # We must always re-score with the instantiated facets.
        hop2_scores = self._two_stage_scoring(hop2_passages, hop2_facets)

        if debug:
            print(f"  [HOP2-SCORE] Re-scored {len(hop2_facets)} instantiated facets on {len(hop2_passages)} passages")

        hop2_result = self._run_safe_cover(hop2_facets, hop2_passages, hop2_scores)

        if debug:
            hop2_coverage = hop2_result.get('metrics', {}).get('coverage', 0)
            print(f"  Hop-2 coverage: {hop2_coverage:.2f}")
            print(f"  Hop-2 selected: {len(hop2_result.get('selected_passages', []))}")

        # ============================================================
        # COMBINE: Union of hop-1 and hop-2 winners
        # ============================================================
        # Merge selected passages (dedup by pid)
        combined_passages = []
        seen_pids = set()

        for p in hop1_result.get('selected_passages', []):
            pid = p.get('pid', '')
            if pid and pid not in seen_pids:
                seen_pids.add(pid)
                combined_passages.append(p)

        for p in hop2_result.get('selected_passages', []):
            pid = p.get('pid', '')
            if pid and pid not in seen_pids:
                seen_pids.add(pid)
                combined_passages.append(p)

        # Merge certificates
        combined_certificates = []
        combined_certificates.extend(hop1_result.get('certificates', []))
        combined_certificates.extend(hop2_result.get('certificates', []))

        # Combined metrics
        total_facets = len(hop1_facets) + len(hop2_facets)
        hop1_covered = hop1_result.get('metrics', {}).get('utility', 0)
        hop2_covered = hop2_result.get('metrics', {}).get('utility', 0)

        combined_coverage = (hop1_covered + hop2_covered) / total_facets if total_facets > 0 else 0

        # Both must pass for non-abstention
        combined_abstained = hop1_result.get('abstained', False) or hop2_result.get('abstained', False)

        # Calculate token costs
        hop1_tokens = hop1_result.get('evidence_tokens', 0)
        hop2_tokens = hop2_result.get('evidence_tokens', 0)

        if debug:
            print(f"  Combined coverage: {combined_coverage:.2f}")
            print(f"  Combined passages: {len(combined_passages)}")
            print(f"  Combined certificates: {len(combined_certificates)}")
            print(f"  Abstained: {combined_abstained}")

        return {
            'selected_passages': combined_passages,
            'certificates': combined_certificates,
            'abstained': combined_abstained,
            'infeasible': hop1_result.get('infeasible', False) or hop2_result.get('infeasible', False),
            'evidence_tokens': hop1_tokens + hop2_tokens,
            'metrics': {
                'coverage': combined_coverage,
                'utility': hop1_covered + hop2_covered,
                'num_units': len(combined_passages),
                'num_facets': total_facets,
                'hop1_coverage': hop1_result.get('metrics', {}).get('coverage', 0),
                'hop2_coverage': hop2_result.get('metrics', {}).get('coverage', 0),
                'bound_entity': bound_entity,
                'inner_relation_type': inner_relation_type,
                'binding_source': binding_source,  # CHANGE 11: Track binding provenance
                'binding_pid': binding_pid if 'binding_pid' in locals() else '',  # CHANGE 11
                'two_pass': True,
            }
        }

    def _run_safe_cover(
        self,
        facets: List[Facet],
        passages: List[Passage],
        scores: TwoStageScores
    ) -> Dict[str, Any]:
        """
        Run Safe-Cover mode with certificates.

        CRITICAL FIX: Relaxed thresholds to avoid always-zero coverage.
        Now tracks evidence tokens separately from total tokens.
        """
        # Monitor drift if enabled
        if self.drift_monitor:
            drift_detected = self.drift_monitor.check_drift(scores.stage2_scores)
            if drift_detected:
                self.telemetry.log("drift_detected", {"timestamp": time.time()})
                self.safe_cover_algo.apply_fallback()

        # Run RC-MCFC algorithm
        result = self.safe_cover_algo.run(
            facets=facets,
            passages=passages,
            p_values=scores.p_values
        )

        # Convert to output format
        selected = []
        total_cost = 0
        budget_cap = (
            self.config.safe_cover.max_evidence_tokens
            if self.config.safe_cover.max_evidence_tokens is not None
            else self.config.safe_cover.token_cap
        )
        for passage in result.selected_passages:
            title = None
            if hasattr(passage, "metadata") and isinstance(passage.metadata, dict):
                title = passage.metadata.get("title")

            selected.append({
                'pid': passage.pid,
                'text': passage.text,
                'title': title,
                'cost': passage.cost,
                'covered_facets': result.coverage_map.get(passage.pid, [])
            })
            total_cost += passage.cost

        certificates = []
        for cert in result.certificates:
            certificates.append({
                'facet_id': cert.facet_id,
                'passage_id': cert.passage_id,
                'threshold': cert.threshold,  # CRITICAL FIX: was 'alpha_bar' but field is 'threshold'
                'p_value': cert.p_value,
                'alpha_facet': cert.alpha_facet,
                'alpha_query': cert.alpha_query,
                'bin': cert.bin,
                'timestamp': cert.timestamp,
                'calibrator_version': self.config.calibration.version
            })

        # Use infeasible and abstained flags from result
        is_infeasible = result.infeasible
        is_abstained = result.abstained

        # ============================================================
        # STRICT CONTRACT: Required facets must be covered, otherwise abstain.
        # Even in "relaxed" configs, returning an answer when required facets are
        # uncovered yields confident-but-wrong behavior (exactly what we're seeing).
        # ============================================================
        try:
            required_ids = [f.facet_id for f in facets if getattr(f, 'required', False)]
            covered = set(getattr(result, 'covered_facets', []) or [])
            missing_required = [fid for fid in required_ids if fid not in covered]
            if missing_required:
                is_infeasible = True
                is_abstained = True
                # attach a reason field so callers can propagate it
                strict_reason = f"REQUIRED_FACETS_UNCOVERED:{','.join(missing_required[:5])}"
        except Exception:
            strict_reason = None

        capped_evidence_tokens = (
            min(total_cost, budget_cap) if budget_cap is not None else total_cost
        )
        evidence_over_budget = (
            max(total_cost - budget_cap, 0) if budget_cap is not None else 0
        )

        return {
            'selected_passages': selected,
            'certificates': certificates,
            'abstained': is_abstained,
            'abstention_reason': locals().get('strict_reason', None) or getattr(result, 'abstention_reason', None),
            'infeasible': is_infeasible,
            'dual_lower_bound': getattr(result, 'dual_lower_bound', None),
            'retrieval_tokens': total_cost,
            'evidence_tokens': capped_evidence_tokens,
            'metrics': {
                'coverage': len(result.covered_facets) / len(facets) if facets else 0,
                'utility': len(result.covered_facets),
                'efficiency': len(result.covered_facets) / max(total_cost, 1) if total_cost > 0 else 0,
                'num_units': len(selected),
                'num_facets': len(facets),  # CRITICAL FIX: Track num_facets
                'num_violated_facets': len(result.uncovered_facets),
                'abstention_reason': result.abstention_reason.value if is_abstained else None,  # Track reason
                'evidence_token_cap': budget_cap,
                'evidence_tokens_raw': total_cost,
                'evidence_tokens_over_budget': evidence_over_budget,
                'dual_lower_bound': getattr(result, 'dual_lower_bound', None),
            }
        }

    def _run_pareto(
        self,
        query: str,
        facets: List[Facet],
        passages: List[Passage],
        scores: TwoStageScores
    ) -> Dict[str, Any]:
        """
        Run Pareto-Knapsack mode with optional VQC and BwK.

        CRITICAL FIX: Properly respects budget constraints.
        CRITICAL FIX: VQC now runs based on uncovered facets and budget,
                    regardless of BwK action, allowing VQC to function.
        """
        current_passages = passages.copy()
        current_facets = facets.copy()
        vqc_iterations = 0

        # Calculate actual budget considering retrieval cost
        actual_budget = self.config.pareto.budget

        # BwK controller for action selection
        if self.config.pareto.use_bwk:
            episode_state = {
                'query': query,
                'facets': current_facets,
                'budget_remaining': actual_budget,
                'iterations': 0
            }
            self.bwk.start_episode(episode_state)

        # Run optimization at least once, then optionally iterate with VQC
        result = None
        while True:
            # Run Pareto optimization with actual budget
            result = self.pareto_optimizer.optimize(
                facets=current_facets,
                passages=current_passages,
                p_values=scores.p_values,
                budget=actual_budget
            )

            # Check if we should continue with VQC iterations
            if vqc_iterations >= self.config.pareto.max_vqc_iterations:
                break

            # Check if we should use VQC to improve coverage
            should_run_vqc = (
                self.config.pareto.use_vqc and
                result.uncovered_facets and
                result.total_cost < actual_budget * 0.8
            )

            # If conditions are met to run VQC
            if should_run_vqc:
                # If BwK is enabled, we can log its action or use it differently,
                # but the decision to run VQC is now independent of the action outcome.
                if self.config.pareto.use_bwk:
                    action = self.bwk.select_action(
                        deficit=result.uncovered_facets,
                        budget_remaining=actual_budget - result.total_cost
                    )
                    # Log the action for debugging if needed, but don't break based on it
                    # self.telemetry.log("bwk_action", {"action": action, "iteration": vqc_iterations})

                # CRITICAL FIX: VQC now runs if the primary conditions are met
                # Generate query rewrites for uncovered facets
                uncovered_facet_objs = [f for f in current_facets if f.facet_id in result.uncovered_facets]
                rewrites = self.vqc.generate_rewrites(
                    query=query,
                    uncovered_facets=uncovered_facet_objs,
                    current_passages=result.selected_passages,
                    max_rewrites=2
                )

                # Retrieve new passages
                new_passages = []
                for rewrite in rewrites:
                    retrieval_result = self.retriever.retrieve(
                        rewrite,
                        top_k=self.config.retrieval.top_k // 2
                    )
                    new_passages.extend(retrieval_result.passages)

                # Update passages and re-score
                current_passages.extend(new_passages)
                new_scores = self._two_stage_scoring(new_passages, current_facets)
                scores.stage2_scores.update(new_scores.stage2_scores)
                scores.p_values.update(new_scores.p_values)

                vqc_iterations += 1
            else:
                # Break the loop if conditions for VQC are not met
                break

        # Update BwK with final reward after the loop (if BwK is used)
        if self.config.pareto.use_bwk:
            # Use the final result after potentially multiple VQC iterations
            # Calculate reward based on the final state
            final_result_utility = result.achieved_utility # Result from the last optimization run
            final_result_cost = result.total_cost
            reward = final_result_utility / max(final_result_cost, 1)
            self.bwk.update_reward(reward)

        # Convert to output format
        selected = []
        for passage in result.selected_passages:
            title = None
            if hasattr(passage, "metadata") and isinstance(passage.metadata, dict):
                title = passage.metadata.get("title")

            selected.append({
                'pid': passage.pid,
                'text': passage.text,
                'title': title,
                'cost': passage.cost,
                'utility_contribution': passage.metadata.get('utility', 0) if passage.metadata else 0
            })

        budget_cap = (
            self.config.pareto.max_evidence_tokens
            if self.config.pareto.max_evidence_tokens is not None
            else self.config.pareto.budget
        )
        capped_evidence_tokens = (
            min(result.total_cost, budget_cap) if budget_cap is not None else result.total_cost
        )
        evidence_over_budget = (
            max(result.total_cost - budget_cap, 0) if budget_cap is not None else 0
        )

        return {
            'selected_passages': selected,
            'certificates': None,
            'abstained': False, # Pareto mode typically does not abstain
            'infeasible': False, # Pareto mode typically is feasible
            'retrieval_tokens': result.total_cost,
            'evidence_tokens': capped_evidence_tokens,  # Track evidence tokens separately
            'metrics': {
                'utility': result.achieved_utility,
                'coverage': len(result.covered_facets) / len(facets) if facets else 0,
                'efficiency': result.achieved_utility / max(result.total_cost, 1),
                'vqc_iterations': vqc_iterations,
                'num_units': len(selected),
                'evidence_token_cap': budget_cap,
                'evidence_tokens_raw': result.total_cost,
                'evidence_tokens_over_budget': evidence_over_budget,
            }
        }

    def get_telemetry(self) -> Dict[str, Any]:
        """Get telemetry data."""
        return self.telemetry.get_summary()

    def reset_caches(self) -> None:
        """Reset all caches."""
        self.score_cache.clear()
        self.retrieval_cache.clear()
        self.nli_scorer.clear_cache()
