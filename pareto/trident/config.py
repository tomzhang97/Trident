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
class FacetConfig:
    """
    Per-facet statistical controls.

    Per TRIDENT Framework Section 4.2:
    - alpha: Per-facet error budget (α_f)
    - max_tests: T_f, max passages tested per facet for Bonferroni
    - alpha_bar is computed as: α_f / T_f
    """
    alpha: float = 0.01  # α_f: Per-facet Type I error level
    max_tests: int = 10  # T_f: Max passages tested per facet (Bonferroni budget)
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
    """
    # FWER Control (Section 4.2)
    # CRITICAL: per_facet_alpha should be reasonably high to avoid always-abstain
    # With deterministic p-values, min achievable is 1/(n_b+1) ~ 0.02 for 50 negatives
    per_facet_alpha: float = 0.1  # α_query: Query-level FWER target
    per_facet_configs: Dict[str, FacetConfig] = field(default_factory=dict)

    # Budget Control (Section 4.3)
    token_cap: Optional[int] = 2000  # B_ctx: Hard context budget cap
    max_evidence_tokens: Optional[int] = None  # Alternative: Maximum evidence tokens
    max_units: Optional[int] = None  # Maximum number of passages/units to select

    # Algorithm Parameters
    dual_tolerance: float = 1e-6  # Tolerance for dual LP convergence
    coverage_threshold: float = 0.15  # Minimum coverage threshold

    # Abstention Control (Section 4.6)
    # CRITICAL: Set to False by default to allow partial coverage
    early_abstain: bool = False  # Abstain early if LB_dual > B_ctx
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
    # CRITICAL: Use "randomized" to avoid deterministic p-value discretization trap
    pvalue_mode: str = "randomized"  # "deterministic" or "randomized"
    label_noise_epsilon: float = 0.0  # ε for label-noise robust p-values


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
    version: str = "v1.0"

    # Mondrian Bin Configuration (Section 3.3)
    use_mondrian: bool = True
    n_min: int = 50  # Minimum per-bin negatives before merging

    # Bin Specification (6 × 3 × 3 = 54 bins)
    facet_types: List[str] = field(default_factory=lambda: [
        "ENTITY", "RELATION", "TEMPORAL", "NUMERIC", "BRIDGE_HOP1", "BRIDGE_HOP2"
    ])
    length_buckets: List[Tuple[int, int]] = field(default_factory=lambda: [
        (0, 50), (50, 150), (150, 10000)  # short, medium, long
    ])
    retriever_score_buckets: List[Tuple[float, float]] = field(default_factory=lambda: [
        (0.0, 0.33), (0.33, 0.67), (0.67, 1.0)  # low, medium, high
    ])

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
    """Configuration for NLI/Cross-encoder scoring."""
    model_name: str = "microsoft/deberta-v2-xlarge-mnli"
    batch_size: int = 32
    max_length: int = 512
    use_cache: bool = True
    cache_size: int = 10000
    score_threshold: float = 0.9


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
    safe_cover: SafeCoverConfig = field(default_factory=SafeCoverConfig)
    pareto: ParetoConfig = field(default_factory=ParetoConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
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
        nli = NLIConfig(**config_dict.get("nli", {}))
        evaluation = EvaluationConfig(**config_dict.get("evaluation", {}))
        telemetry = TelemetryConfig(**config_dict.get("telemetry", {}))

        return cls(
            mode=config_dict.get("mode", "safe_cover"),
            safe_cover=safe_cover,
            pareto=pareto,
            calibration=calibration,
            llm=llm,
            retrieval=retrieval,
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
