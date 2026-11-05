"""Enhanced configuration module for TRIDENT with full specification support."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path


@dataclass
class LLMConfig:
    """Configuration for Large Language Model."""
    model_name: str = "meta-llama/Llama-2-7b-hf"
    temperature: float = 0.0
    max_new_tokens: int = 512
    device: str = "cuda:0"
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    batch_size: int = 1
    use_flash_attention: bool = False
    dtype: str = "float16"  # float16, bfloat16, float32


@dataclass
class RetrievalConfig:
    """Configuration for retrieval system."""
    method: str = "dense"  # dense, sparse, hybrid
    encoder_model: str = "facebook/contriever"
    corpus_path: Optional[str] = None
    index_path: Optional[str] = None
    top_k: int = 100
    rerank_top_k: int = 20
    use_reranker: bool = True
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@dataclass
class FacetMinerConfig:
    """Configuration for facet mining behavior."""

    # Toggle LLM-driven facet planning (experimental)
    use_llm_facet_plan: bool = False
    # Public normalized relation schema JSON (optional)
    relation_schema_json_path: str | None = None
    # Option B: single-call LLM to choose relation from schema
    use_llm_relation_plan: bool = False


@dataclass
class FacetConfig:
    """
    Per-facet statistical controls.

    Per TRIDENT Framework Section 4.2:
    - alpha: Per-facet error budget (α_f)
    - max_tests: T_f, max passages tested per facet for Bonferroni
    - alpha_bar is computed as: α_f / T_f

    NOTE: max_tests=3 is the new default (was 10). Lower T_f means less
    Bonferroni penalty, making certification more feasible. With T_f=10
    and |F|=3, threshold was ~0.001; with T_f=3, it's ~0.02.
    """
    alpha: float = 0.01  # α_f: Per-facet Type I error level
    max_tests: int = 3  # T_f: Max passages tested per facet (reduced from 10)
    prefilter_tests: int = 3  # Tests for spurious facet filtering
    fallback_scale: float = 0.5  # ρ: Conservative threshold multiplier on drift (0.5-0.9)
    weight: float = 1.0  # w_f: Facet weight for utility computation


@dataclass
class SafeCoverConfig:
    """
    Settings for RC-MCFC Safe-Cover algorithm.

    Per TRIDENT Framework Section 4:
    - Query-level FWER ≤ α_query for facet-coverage claims
    - Context budget respected; sound abstention when infeasible
    - Full audit trail (versions, thresholds, p-values, bins)

    Alpha Semantics (Authoritative Mapping):
    - alpha_query := per_facet_alpha (legacy name kept for backward compatibility)
    - alpha_f := alpha_query / |F(q)|  (per-facet allocation)
    - alpha_bar_f := alpha_f / T_f     (per-test threshold)

    These three values are logged in every certificate. The legacy name
    'per_facet_alpha' is preserved for backward compatibility but should be
    understood as the QUERY-LEVEL target, not per-facet.

    IMPORTANT: VQC and BwK are disabled in Safe-Cover mode (use_vqc=False,
    use_bwk=False) because they change candidate distribution and invalidate
    selection-conditional calibration.
    """
    # FWER Control (Section 4.2)
    # NOTE: per_facet_alpha is the QUERY-LEVEL FWER target (α_query).
    # The name is legacy; it does NOT mean each facet gets this budget.
    # Actual per-facet budget is: α_f = per_facet_alpha / |F(q)|
    per_facet_alpha: float = 0.05  # α_query: Query-level FWER target (LEGACY NAME)
    per_facet_configs: Dict[str, FacetConfig] = field(default_factory=dict)

    # Budget Control (Section 4.3)
    token_cap: Optional[int] = 2000  # B_ctx: Hard context budget cap
    max_evidence_tokens: Optional[int] = None  # Alternative: Maximum evidence tokens
    max_units: Optional[int] = None  # Maximum number of passages/units to select

    # Algorithm Parameters
    dual_tolerance: float = 1e-6  # Tolerance for dual LP convergence
    # NOTE: coverage_threshold is for auxiliary pre-screen only, not p-value decisions
    coverage_threshold: float = 0.15  # Auxiliary pre-screen threshold (not for p-value decisions)

    # Abstention Control (Section 4.6)
    early_abstain: bool = True  # Abstain early if LB_dual > B_ctx
    abstain_on_infeasible: bool = False  # Abstain if constraints cannot be satisfied
    stop_on_budget: bool = True  # Stop selection when budget is exhausted

    # Certificate Generation (Section 4.7)
    use_certificates: bool = True  # Emit certificates for covered facets

    # Shift Monitoring (Section 5)
    monitor_drift: bool = False  # Enable PSI/KL monitoring
    psi_threshold: float = 0.25  # PSI alarm threshold (was 0.5, now per spec)
    kl_threshold: float = 0.5  # KL divergence alarm threshold
    violation_multiplier: float = 2.0  # Violation rate > 2·α_query triggers alarm
    threshold_shrink_factor: float = 0.7  # ρ ∈ (0.5, 0.9) for threshold shrinking
    recalibration_buffer_size: int = 1000  # N_recal for scheduling recalibration

    # Fallback Behavior
    fallback_to_pareto: bool = True  # Fall back to Pareto mode on abstention

    # P-Value Configuration
    pvalue_mode: str = "deterministic"  # "deterministic" or "randomized"
    label_noise_epsilon: float = 0.0  # ε for label-noise robust p-values

    # Feasibility Policy for Deterministic P-Values (Section 3.4)
    # When alpha_bar_f < 1/(n_b+1), apply in order:
    # 1) Switch to randomized for that facet
    # 2) Merge bins per policy (retriever_score → length → facet_type)
    # 3) Abstain with reason="pvalue_infeasible_small_bin"
    feasibility_auto_switch_to_randomized: bool = True  # Auto-switch to randomized when infeasible
    feasibility_auto_merge_bins: bool = True  # Auto-merge bins when infeasible

    # Facet Pre-Pruning (OFF by default - Section 11.3)
    # NOTE: Pre-pruning drops hypotheses after seeing verifier evidence, which
    # changes the multiple-testing family. It is OFF by default and out-of-scope
    # for FWER guarantees. If enabled, treat it as part of the miner (pre-verification).
    enable_spurious_facet_prepass: bool = False  # OFF by default
    spurious_alpha: float = 1e-4  # Threshold for spurious facet filtering
    spurious_max_tests: int = 3  # Max tests per facet in pre-pass

    @property
    def alpha_query(self) -> float:
        """Alias for per_facet_alpha with correct semantic name."""
        return self.per_facet_alpha

    @alpha_query.setter
    def alpha_query(self, value: float) -> None:
        """Set the query-level FWER target."""
        self.per_facet_alpha = value


@dataclass
class ParetoConfig:
    """Settings for Pareto-Knapsack mode."""
    budget: int = 2000
    relaxed_alpha: float = 0.3
    weight_default: float = 1.0
    use_vqc: bool = True  # Verifier-driven Query Compiler
    use_bwk: bool = True  # Bandits with Knapsacks
    max_vqc_iterations: int = 3
    bwk_exploration_bonus: float = 0.1
    # Budget control fields for config families
    max_evidence_tokens: Optional[int] = None  # Maximum evidence tokens to use
    max_units: Optional[int] = None  # Maximum number of passages/units to select
    stop_on_budget: bool = True  # Stop selection when budget is exhausted


@dataclass
class CalibrationConfig:
    """
    Settings for score calibration.

    Per TRIDENT Framework Section 3.3:
    - Selection-conditional Mondrian calibration
    - Bins: b(p,f) = (facet_type(f), length_bucket(p), retriever_score_bucket(p))
    - Typical grid: 6 × 3 × 3 = 54 bins
    """
    method: str = "conformal"  # "conformal", "isotonic", "platt", "beta"
    version: str = "v2.2"

    # Mondrian Bin Configuration (Section 3.3)
    use_mondrian: bool = True
    n_min: int = 25  # Minimum per-bin negatives before merging

    # Bin Specification - use canonical types only
    facet_types: List[str] = field(default_factory=lambda: [
        "ENTITY", "RELATION", "TEMPORAL", "NUMERIC", "BRIDGE_HOP"
    ])
    # BOTH DISABLED by default - bucket conditioning can cause lookup misses
    # if calibrator was trained without bucketing or with "all" bucket.
    # Re-enable after confirming certificates work with unconditioned lookup.
    length_buckets: Optional[List[Tuple[int, int]]] = None
    retriever_score_buckets: Optional[List[Tuple[float, float]]] = None

    # Legacy fields
    facet_bins: Dict[str, List[float]] = field(default_factory=dict)
    reliability_table_size: int = 20
    recalibration_buffer_size: int = 1000

    # P-Value Mode (Section 3.4)
    pvalue_mode: str = "deterministic"  # "deterministic" or "randomized"

    # Label Noise Robustness (Section 3.5)
    label_noise_epsilon: float = 0.0  # ε for denominator-inflated p-values


@dataclass
class NLIConfig:
    """
    Configuration for NLI/Cross-encoder scoring.

    NOTE on score_threshold: This is used ONLY for optional computational
    pre-screening (pruning candidates before full p-value computation).
    All coverage decisions in Safe-Cover are made by CALIBRATED P-VALUES,
    not raw NLI scores. Do not confuse this with coverage thresholds.
    """
    model_name: str = "microsoft/deberta-v2-xlarge-mnli"
    batch_size: int = 32
    max_length: int = 512
    use_cache: bool = True
    cache_size: int = 10000
    # NOTE: score_threshold is for PRE-SCREEN only (computational pruning).
    # Coverage decisions use calibrated p-values, NOT this threshold.
    score_threshold: float = 0.9  # Pre-screen threshold (NOT for coverage decisions)


@dataclass
class EvaluationConfig:
    """Configuration for evaluation metrics."""
    metrics: List[str] = field(default_factory=lambda: ["em", "f1", "support_em", "faithfulness"])
    dataset: str = "hotpotqa"
    save_predictions: bool = True
    save_certificates: bool = True
    compute_pareto_curves: bool = True


@dataclass
class TelemetryConfig:
    """Configuration for telemetry and monitoring."""
    enable: bool = True
    log_level: str = "INFO"
    track_latency: bool = True
    track_gpu_util: bool = True
    track_cache_stats: bool = True
    profile_memory: bool = False
    save_traces: bool = True


@dataclass
class TridentConfig:
    """Complete TRIDENT system configuration."""
    mode: str = "safe_cover"  # safe_cover, pareto, both
    use_llm_question_planner: bool = False
    llm_planner_max_new_tokens: int = 160
    safe_cover: SafeCoverConfig = field(default_factory=SafeCoverConfig)
    pareto: ParetoConfig = field(default_factory=ParetoConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    facet_miner: FacetMinerConfig = field(default_factory=FacetMinerConfig)
    nli: NLIConfig = field(default_factory=NLIConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    telemetry: TelemetryConfig = field(default_factory=TelemetryConfig)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "TridentConfig":
        """Create config from dictionary."""
        safe_cover = SafeCoverConfig(**config_dict.get("safe_cover", {}))
        pareto = ParetoConfig(**config_dict.get("pareto", {}))
        calibration = CalibrationConfig(**config_dict.get("calibration", {}))
        llm = LLMConfig(**config_dict.get("llm", {}))
        retrieval = RetrievalConfig(**config_dict.get("retrieval", {}))
        facet_miner = FacetMinerConfig(**config_dict.get("facet_miner", {}))
        nli = NLIConfig(**config_dict.get("nli", {}))
        evaluation = EvaluationConfig(**config_dict.get("evaluation", {}))
        telemetry = TelemetryConfig(**config_dict.get("telemetry", {}))

        return cls(
            mode=config_dict.get("mode", "safe_cover"),
            use_llm_question_planner=config_dict.get("use_llm_question_planner", False),
            llm_planner_max_new_tokens=config_dict.get("llm_planner_max_new_tokens", 160),
            safe_cover=safe_cover,
            pareto=pareto,
            calibration=calibration,
            llm=llm,
            retrieval=retrieval,
            facet_miner=facet_miner,
            nli=nli,
            evaluation=evaluation,
            telemetry=telemetry
        )

    @classmethod
    def from_file(cls, path: str) -> "TridentConfig":
        """Load configuration from JSON file."""
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def save(self, path: str) -> None:
        """Save configuration to JSON file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)
