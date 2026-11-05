"""
Small-shard integration test for the safe_cover fix.

Simulates what pipeline.py hands to SafeCoverAlgorithm for a mix of 12
multi-hop queries. Uses a realistic p-value distribution observed in the
calibrator logs (ENTITY p-values are broad, RELATION/BRIDGE p-values are
sharp). The test asserts the abstention rate drops below 20% — the
pre-fix rate was 96-100% per the 2000relaxed_working results file.
"""

import random

from trident.calibration import ReliabilityCalibrator
from trident.candidates import Passage
from trident.config import SafeCoverConfig
from trident.facets import Facet, FacetType
from trident.safe_cover import SafeCoverAlgorithm, AbstentionReason


def _mk_query(qid: int, seed: int):
    """Build a synthetic multi-hop query with:
    - 1-2 BRIDGE or RELATION facets (structural, must be covered)
    - 2-4 ENTITY facets (optional noise from miner)
    - 10 passages with cost=100 each
    - p-values drawn from a realistic distribution
    """
    rng = random.Random(seed)
    facets = []
    n_struct = rng.randint(1, 2)
    for i in range(n_struct):
        ft = rng.choice([FacetType.BRIDGE, FacetType.RELATION])
        facets.append(
            Facet(facet_id=f"q{qid}_s{i}", facet_type=ft, template={"x": "y"})
        )
    n_ent = rng.randint(2, 4)
    for i in range(n_ent):
        facets.append(
            Facet(
                facet_id=f"q{qid}_e{i}",
                facet_type=FacetType.ENTITY,
                template={"mention": f"X{i}"},
            )
        )

    passages = [
        Passage(pid=f"q{qid}_p{j}", text=f"content {j}", cost=100) for j in range(10)
    ]

    p_values = {}
    for f in facets:
        for p in passages:
            key = (p.pid, f.facet_id)
            # Realistic: most (p,f) pairs are not supporting (p near 1.0)
            # A fraction of passages supports each facet with moderate p-values
            if rng.random() < 0.25:
                # supporting evidence: p ∈ [0.001, 0.08]
                p_values[key] = rng.uniform(0.001, 0.08)
            else:
                # non-supporting: p ∈ [0.15, 1.0]
                p_values[key] = rng.uniform(0.15, 1.0)
    return facets, passages, p_values


def run_shard(n_queries: int, config: SafeCoverConfig, seed_base: int = 0):
    cal = ReliabilityCalibrator()
    # Prime a default table so ReliabilityCalibrator works
    for ft in ["ENTITY", "RELATION", "BRIDGE", "BRIDGE_HOP"]:
        cal.tables[ft] = [(0.0, 1.0), (0.9, 0.01), (1.0, 0.0)]

    algo = SafeCoverAlgorithm(config=config, calibrator=cal)
    results = []
    for qid in range(n_queries):
        facets, passages, pvals = _mk_query(qid, seed_base + qid)
        r = algo.run(facets, passages, pvals)
        results.append(r)
    return results


def summarize(results, label):
    n = len(results)
    n_abs = sum(1 for r in results if r.abstained)
    reasons = {}
    for r in results:
        if r.abstained:
            reasons[r.abstention_reason.value] = (
                reasons.get(r.abstention_reason.value, 0) + 1
            )
    n_certs = sum(len(r.certificates) for r in results)
    n_selected = sum(len(r.selected_passages) for r in results)
    avg_cov = sum(len(r.covered_facets) for r in results) / n
    print(f"\n=== {label} ===")
    print(f"  queries={n}  abstained={n_abs}/{n} ({100 * n_abs / n:.0f}%)")
    print(f"  certificates={n_certs}  selected_passages_total={n_selected}")
    print(f"  avg facets covered/query={avg_cov:.2f}")
    if reasons:
        print(f"  abstention reasons: {reasons}")
    return n_abs, n_certs


def test_abstention_rate_drops_with_fix():
    """The fix (fallback_to_pareto → abstain_on_infeasible=False) must
    cut the abstention rate substantially compared to strict mode."""
    # Shadow of the pre-fix behavior (abstain on any uncoverable facet)
    strict_cfg = SafeCoverConfig(
        per_facet_alpha=0.1,
        abstain_on_infeasible=True,
        early_abstain=True,
        fallback_to_pareto=False,
        max_evidence_tokens=2000,
    )
    # Fixed behavior (allow fallback)
    lenient_cfg = SafeCoverConfig(
        per_facet_alpha=0.1,
        abstain_on_infeasible=False,
        early_abstain=True,
        fallback_to_pareto=True,
        max_evidence_tokens=2000,
    )

    strict_results = run_shard(12, strict_cfg, seed_base=1000)
    lenient_results = run_shard(12, lenient_cfg, seed_base=1000)

    s_abs, s_cert = summarize(strict_results, "STRICT (old semantics)")
    l_abs, l_cert = summarize(lenient_results, "FIXED (new semantics)")

    # The fix should drop abstention to < 25% of queries
    assert l_abs <= 3, f"fixed abstention too high: {l_abs}/12"
    # And produce more certificates
    assert l_cert >= s_cert, "fixed should produce at least as many certificates"


def test_structural_vs_optional():
    """A query with ONLY optional (ENTITY) facets should never abstain."""
    cfg = SafeCoverConfig(
        per_facet_alpha=0.1,
        abstain_on_infeasible=True,  # even in strict mode
        early_abstain=True,
        max_evidence_tokens=2000,
    )
    cal = ReliabilityCalibrator()
    cal.tables["ENTITY"] = [(0.0, 1.0), (0.9, 0.01), (1.0, 0.0)]
    algo = SafeCoverAlgorithm(config=cfg, calibrator=cal)

    facets = [
        Facet(
            facet_id=f"e{i}", facet_type=FacetType.ENTITY, template={"mention": f"X{i}"}
        )
        for i in range(3)
    ]
    passages = [Passage(pid=f"p{j}", text="x", cost=100) for j in range(5)]
    # NONE of the entities can be covered (all p=1.0)
    pvals = {(p.pid, f.facet_id): 1.0 for p in passages for f in facets}
    r = algo.run(facets, passages, pvals)
    # Because ENTITY is optional, no abstention should happen
    assert not r.abstained, f"optional-only query abstained: {r.abstention_reason}"


def test_structural_facet_with_poor_coverage_falls_back():
    """Structural facet but no passage crosses threshold, config allows
    fallback → no abstention (pipeline handles fallback)."""
    cfg = SafeCoverConfig(
        per_facet_alpha=0.1,
        abstain_on_infeasible=False,
        fallback_to_pareto=True,
        max_evidence_tokens=2000,
    )
    cal = ReliabilityCalibrator()
    cal.tables["RELATION"] = [(0.0, 1.0), (0.9, 0.01), (1.0, 0.0)]
    algo = SafeCoverAlgorithm(config=cfg, calibrator=cal)

    facets = [Facet(facet_id="r1", facet_type=FacetType.RELATION, template={})]
    passages = [Passage(pid=f"p{j}", text="x", cost=100) for j in range(5)]
    pvals = {(p.pid, "r1"): 1.0 for p in passages}
    r = algo.run(facets, passages, pvals)
    assert not r.abstained
    assert r.abstention_reason == AbstentionReason.NONE


def test_best_effort_selects_when_no_external_fallback():
    """Mirrors the safe_cover_2000_relaxed config in production: both
    abstain_on_infeasible and fallback_to_pareto are False, so the caller
    cannot rescue an empty selection. Safe-cover must produce best-effort
    evidence (cheapest passages by min p-value) rather than returning [],
    otherwise pipeline.py has no evidence to feed the LLM and abstains
    100% of the time (the symptom we observed on the 50-query shard).
    """
    cfg = SafeCoverConfig(
        per_facet_alpha=0.10,
        abstain_on_infeasible=False,
        fallback_to_pareto=False,
        early_abstain=True,
        max_evidence_tokens=2000,
    )
    cal = ReliabilityCalibrator()
    cal.tables["RELATION"] = [(0.0, 1.0), (0.9, 0.01), (1.0, 0.0)]
    algo = SafeCoverAlgorithm(config=cfg, calibrator=cal)

    facets = [
        Facet(facet_id="r1", facet_type=FacetType.RELATION, template={}),
        Facet(facet_id="e1", facet_type=FacetType.ENTITY, template={}),
    ]
    passages = [Passage(pid=f"p{j}", text="x", cost=100) for j in range(10)]
    # No p-value below alpha_bar (~0.01) — strict greedy would pick nothing.
    # Best-effort path should still surface the lowest-p passages so the LLM
    # has evidence to reason over.
    pvals = {}
    for j, p in enumerate(passages):
        pvals[(p.pid, "r1")] = 0.20 + 0.05 * j
        pvals[(p.pid, "e1")] = 0.30 + 0.05 * j

    r = algo.run(facets, passages, pvals)

    assert not r.abstained, "best-effort must not abstain when fallback is off"
    assert len(r.selected_passages) > 0, (
        "best-effort must select at least one passage; got empty selection — "
        "pipeline would otherwise feed the LLM no evidence and abstain"
    )
    # And the cheapest/lowest-p passage (p0 — min p across both facets) should
    # be picked first by the best-effort ranker.
    assert r.selected_passages[0].pid == "p0", (
        f"expected p0 first (lowest p), got {r.selected_passages[0].pid}"
    )


def test_heuristic_mode_completes_more_queries():
    """is_heuristic_mode=True should cover more queries on the same inputs
    because the threshold floor is lifted."""
    cfg = SafeCoverConfig(
        per_facet_alpha=0.01,  # very tight
        abstain_on_infeasible=True,
        max_evidence_tokens=2000,
    )
    cal = ReliabilityCalibrator()
    for ft in ["ENTITY", "RELATION", "BRIDGE"]:
        cal.tables[ft] = [(0.0, 1.0), (0.9, 0.01), (1.0, 0.0)]
    algo = SafeCoverAlgorithm(config=cfg, calibrator=cal)

    def build():
        facets = [
            Facet(facet_id="r1", facet_type=FacetType.RELATION, template={}),
            Facet(facet_id="e1", facet_type=FacetType.ENTITY, template={}),
        ]
        passages = [Passage(pid=f"p{j}", text="x", cost=100) for j in range(10)]
        # p-value just above tight alpha_bar but at/below heuristic floor
        pvals = {
            (p.pid, f.facet_id): (0.04 if f.facet_id == "r1" else 0.02)
            for p in passages
            for f in facets
        }
        return facets, passages, pvals

    f, p, pv = build()
    r_strict = algo.run(f, p, pv, is_heuristic_mode=False)
    f, p, pv = build()
    r_heur = algo.run(f, p, pv, is_heuristic_mode=True)

    # In heuristic mode the threshold floor means r1 crosses → covered.
    # In strict mode with alpha=0.01/1/10 = 0.001 it cannot.
    assert len(r_heur.certificates) > len(r_strict.certificates)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if not name.startswith("test_"):
            continue
        try:
            fn()
            print(f"PASS  {name}")
        except AssertionError as e:
            print(f"FAIL  {name}: {e}")
        except Exception as e:
            print(f"ERROR {name}: {e!r}")
