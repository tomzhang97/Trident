"""Tests for Safe-Cover algorithm."""

import importlib.util

try:
    import pytest

    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False

HAS_NUMPY = importlib.util.find_spec("numpy") is not None
if HAS_PYTEST:
    pytestmark = pytest.mark.skipif(
        not HAS_NUMPY, reason="numpy is required for Safe-Cover"
    )

if HAS_NUMPY:
    from trident.calibration import ReliabilityCalibrator
    from trident.candidates import Passage
    from trident.config import FacetConfig, SafeCoverConfig
    from trident.facets import Facet, FacetType
    from trident.safe_cover import SafeCoverAlgorithm, AbstentionReason
else:  # pragma: no cover - placeholders to satisfy type checkers when skipped
    ReliabilityCalibrator = Passage = FacetConfig = SafeCoverConfig = None  # type: ignore
    Facet = FacetType = SafeCoverAlgorithm = AbstentionReason = None  # type: ignore


def test_safe_cover_selects_passage_when_threshold_met():
    """Test that passages are selected when p-values are below threshold."""
    facets = [
        Facet(
            facet_id="f1", facet_type=FacetType.ENTITY, template={"mention": "Einstein"}
        )
    ]
    passages = [Passage(pid="p1", text="Albert Einstein wrote many papers.", cost=50)]

    calibrator = ReliabilityCalibrator()
    calibrator.tables["ENTITY"] = [(0.0, 1.0), (0.9, 0.01), (1.0, 0.0)]

    config = SafeCoverConfig(per_facet_alpha=0.1)
    algo = SafeCoverAlgorithm(config=config, calibrator=calibrator)

    # P-values indicating the passage supports the facet
    p_values = {("p1", "f1"): 0.05}  # Below threshold

    result = algo.run(facets, passages, p_values)
    assert len(result.selected_passages) == 1
    assert result.selected_passages[0].pid == "p1"
    assert not result.abstained


def test_safe_cover_carries_calibrator_version_into_certificates():
    """Certificates should record the calibrator version used for the run."""
    facets = [
        Facet(
            facet_id="f1", facet_type=FacetType.ENTITY, template={"mention": "Einstein"}
        )
    ]
    passages = [Passage(pid="p1", text="Albert Einstein wrote many papers.", cost=50)]

    calibrator = ReliabilityCalibrator(version="cal_v123")
    calibrator.tables["ENTITY"] = [(0.0, 1.0), (0.9, 0.01), (1.0, 0.0)]

    config = SafeCoverConfig(per_facet_alpha=0.1)
    algo = SafeCoverAlgorithm(config=config, calibrator=calibrator)

    p_values = {("p1", "f1"): 0.05}

    result = algo.run(facets, passages, p_values)

    assert result.certificates, "Expected a certificate for the covered facet"
    assert result.certificates[0].calibrator_version == "cal_v123"


def test_safe_cover_records_bin_size_from_legacy_bins():
    """Certificates should capture bin size even when bins are simple lists."""
    facets = [
        Facet(
            facet_id="f1", facet_type=FacetType.ENTITY, template={"mention": "Einstein"}
        )
    ]
    passages = [Passage(pid="p1", text="Albert Einstein wrote many papers.", cost=50)]

    calibrator = ReliabilityCalibrator(version="cal_v123")
    calibrator.tables["ENTITY"] = [(0.0, 1.0), (0.9, 0.01), (1.0, 0.0)]
    calibrator.bins["default"] = [0.2, 0.3, 0.4]

    config = SafeCoverConfig(per_facet_alpha=0.1)
    algo = SafeCoverAlgorithm(config=config, calibrator=calibrator)

    p_values = {("p1", "f1"): 0.05}

    result = algo.run(facets, passages, p_values)

    assert result.certificates, "Expected a certificate for the covered facet"
    assert result.certificates[0].bin_size == 3


def test_safe_cover_records_bin_size_from_dict_bins():
    """Certificates should capture bin size when bin metadata is a dict."""
    facets = [
        Facet(
            facet_id="f1", facet_type=FacetType.ENTITY, template={"mention": "Einstein"}
        )
    ]
    passages = [Passage(pid="p1", text="Albert Einstein wrote many papers.", cost=50)]

    calibrator = ReliabilityCalibrator(version="cal_v123")
    calibrator.tables["ENTITY"] = [(0.0, 1.0), (0.9, 0.01), (1.0, 0.0)]
    calibrator.bins["default"] = {"n_negatives": 7}

    config = SafeCoverConfig(per_facet_alpha=0.1)
    algo = SafeCoverAlgorithm(config=config, calibrator=calibrator)

    p_values = {("p1", "f1"): 0.05}

    result = algo.run(facets, passages, p_values)

    assert result.certificates, "Expected a certificate for the covered facet"
    assert result.certificates[0].bin_size == 7


def test_safe_cover_abstains_on_no_facets():
    """Test that Safe-Cover abstains when no facets are provided."""
    facets = []  # No facets
    passages = [Passage(pid="p1", text="Some text.", cost=50)]

    calibrator = ReliabilityCalibrator()
    config = SafeCoverConfig(per_facet_alpha=0.1)
    algo = SafeCoverAlgorithm(config=config, calibrator=calibrator)

    p_values = {}
    result = algo.run(facets, passages, p_values)

    assert result.abstained
    assert result.abstention_reason == AbstentionReason.NO_FACETS
    assert len(result.selected_passages) == 0


def test_safe_cover_abstains_on_no_passages():
    """Test that Safe-Cover abstains when no passages are provided."""
    facets = [
        Facet(
            facet_id="f1", facet_type=FacetType.ENTITY, template={"mention": "Einstein"}
        )
    ]
    passages = []  # No passages

    calibrator = ReliabilityCalibrator()
    config = SafeCoverConfig(per_facet_alpha=0.1)
    algo = SafeCoverAlgorithm(config=config, calibrator=calibrator)

    p_values = {}
    result = algo.run(facets, passages, p_values)

    assert result.abstained
    assert result.abstention_reason == AbstentionReason.NO_PASSAGES
    assert len(result.selected_passages) == 0
    assert result.uncovered_facets == ["f1"]


def test_safe_cover_handles_uncoverable_structural_facets():
    """Uncoverable STRUCTURAL facet (RELATION) forces abstention under
    abstain_on_infeasible=True. Optional facets do not."""
    facets = [
        Facet(facet_id="f1", facet_type=FacetType.RELATION, template={"relation": "A"}),
        Facet(facet_id="f2", facet_type=FacetType.RELATION, template={"relation": "B"}),
    ]
    passages = [Passage(pid="p1", text="About A", cost=50)]

    calibrator = ReliabilityCalibrator()
    config = SafeCoverConfig(per_facet_alpha=0.1, abstain_on_infeasible=True)
    algo = SafeCoverAlgorithm(config=config, calibrator=calibrator)

    p_values = {("p1", "f1"): 0.005, ("p1", "f2"): 1.0}

    result = algo.run(facets, passages, p_values)
    assert result.abstained
    assert result.abstention_reason in (
        AbstentionReason.NO_COVERING_PASSAGES,
        AbstentionReason.INFEASIBILITY_PROVEN,
        AbstentionReason.INFEASIBLE_MISSING_FACETS,
        AbstentionReason.BUDGET_EXHAUSTED,
    )


def test_safe_cover_optional_facets_do_not_abstain():
    """Uncoverable ENTITY facets (non-structural) should never block a query."""
    facets = [
        Facet(facet_id="f1", facet_type=FacetType.ENTITY, template={"mention": "A"}),
        Facet(facet_id="f2", facet_type=FacetType.ENTITY, template={"mention": "B"}),
    ]
    passages = [Passage(pid="p1", text="About A", cost=50)]

    calibrator = ReliabilityCalibrator()
    config = SafeCoverConfig(per_facet_alpha=0.1, abstain_on_infeasible=False)
    algo = SafeCoverAlgorithm(config=config, calibrator=calibrator)

    p_values = {("p1", "f1"): 0.05, ("p1", "f2"): 1.0}
    result = algo.run(facets, passages, p_values)

    assert not result.abstained
    assert len(result.selected_passages) == 1
    assert "f1" in result.covered_facets


def test_safe_cover_fallback_does_not_abstain_on_initial_infeasibility():
    """With abstain_on_infeasible=False and a structurally infeasible query,
    safe_cover should return what it has (empty if nothing covers) rather
    than short-circuiting so the pipeline can fall back to Pareto."""
    facets = [
        Facet(facet_id="f1", facet_type=FacetType.RELATION, template={}),
    ]
    passages = [Passage(pid="p1", text="x", cost=50)]

    calibrator = ReliabilityCalibrator()
    config = SafeCoverConfig(
        per_facet_alpha=0.1, abstain_on_infeasible=False, early_abstain=True
    )
    algo = SafeCoverAlgorithm(config=config, calibrator=calibrator)

    p_values = {("p1", "f1"): 1.0}  # structurally uncoverable
    result = algo.run(facets, passages, p_values)
    # with abstain_on_infeasible=False, greedy didn't pick p1 (no coverage)
    # but we still set abstained=False — pipeline is expected to Pareto-fallback
    assert not result.abstained
    assert result.abstention_reason == AbstentionReason.NONE


def test_safe_cover_bonferroni_uses_structural_count():
    """alpha_bar for a structural facet = alpha_query / n_structural / T_f.
    When there are optional facets mixed in, they do NOT inflate the
    Bonferroni denominator."""
    facets = [
        Facet(facet_id="r1", facet_type=FacetType.RELATION, template={}),
        Facet(facet_id="e1", facet_type=FacetType.ENTITY, template={}),
        Facet(facet_id="e2", facet_type=FacetType.ENTITY, template={}),
    ]
    passages = [Passage(pid="p1", text="x", cost=50)]
    calibrator = ReliabilityCalibrator()
    config = SafeCoverConfig(per_facet_alpha=0.2)
    algo = SafeCoverAlgorithm(config=config, calibrator=calibrator)

    knobs = algo._freeze_episode_knobs(facets, passages)
    # n_structural = 1 → alpha_f[r1] = 0.2
    assert abs(knobs.alpha_f["r1"] - 0.2) < 1e-9
    # optional facets get alpha_f=1.0
    assert knobs.alpha_f["e1"] == 1.0
    assert knobs.alpha_f["e2"] == 1.0


def test_safe_cover_heuristic_mode_applies_floor():
    """is_heuristic_mode raises alpha_bar to at least HEURISTIC_ALPHA_FLOOR
    for structural facets."""
    from trident.safe_cover import HEURISTIC_ALPHA_FLOOR

    facets = [
        Facet(facet_id="r1", facet_type=FacetType.RELATION, template={}),
    ]
    passages = [
        Passage(pid=f"p{i}", text="x", cost=50) for i in range(12)
    ]  # > exhaustive_pool_max_n
    calibrator = ReliabilityCalibrator()
    config = SafeCoverConfig(per_facet_alpha=0.01)  # very tight
    algo = SafeCoverAlgorithm(config=config, calibrator=calibrator)

    knobs = algo._freeze_episode_knobs(facets, passages, is_heuristic_mode=True)
    assert knobs.alpha_bar_f["r1"] >= HEURISTIC_ALPHA_FLOOR


def test_safe_cover_certificates_match_greedy_decisions():
    """Every selected passage should have at least one certificate for a
    facet in coverage_map[pid]."""
    facets = [
        Facet(facet_id="f1", facet_type=FacetType.RELATION, template={}),
        Facet(facet_id="f2", facet_type=FacetType.RELATION, template={}),
    ]
    passages = [
        Passage(pid="pA", text="", cost=50),
        Passage(pid="pB", text="", cost=50),
    ]
    calibrator = ReliabilityCalibrator()
    config = SafeCoverConfig(per_facet_alpha=0.2, abstain_on_infeasible=False)
    algo = SafeCoverAlgorithm(config=config, calibrator=calibrator)

    p_values = {
        ("pA", "f1"): 0.01,
        ("pA", "f2"): 1.0,
        ("pB", "f1"): 1.0,
        ("pB", "f2"): 0.01,
    }
    result = algo.run(facets, passages, p_values)
    assert len(result.selected_passages) == 2
    # coverage_map must be populated for every selection
    for p in result.selected_passages:
        assert p.pid in result.coverage_map
        assert len(result.coverage_map[p.pid]) >= 1
    # certificates cover both facets
    cert_fids = {c.facet_id for c in result.certificates}
    assert cert_fids == {"f1", "f2"}


def test_safe_cover_coverage_calculation():
    """Test coverage metrics calculation."""
    facets = [
        Facet(facet_id="f1", facet_type=FacetType.ENTITY, template={"mention": "A"}),
        Facet(facet_id="f2", facet_type=FacetType.ENTITY, template={"mention": "B"}),
    ]
    passages = [
        Passage(pid="p1", text="About A", cost=50),
        Passage(pid="p2", text="About B", cost=50),
    ]

    calibrator = ReliabilityCalibrator()
    config = SafeCoverConfig(per_facet_alpha=0.1, abstain_on_infeasible=False)
    algo = SafeCoverAlgorithm(config=config, calibrator=calibrator)

    # Both facets can be covered
    p_values = {
        ("p1", "f1"): 0.05,
        ("p2", "f2"): 0.05,
    }

    result = algo.run(facets, passages, p_values)

    assert not result.abstained
    assert len(result.covered_facets) == 2
    assert len(result.uncovered_facets) == 0
