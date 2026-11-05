from trident.candidates import Passage
from trident.facets import Facet, FacetType
from trident.pareto import ParetoKnapsackOptimizer
from trident.presets import available_presets, get_preset


def test_final_preset_selects_within_budget() -> None:
    config = get_preset()
    facets = [Facet("f1", FacetType.RELATION, {"predicate": "created_by"})]
    passages = [
        Passage("cheap", "support", cost=600),
        Passage("expensive", "support", cost=700),
    ]
    p_values = {("cheap", "f1"): 0.01, ("expensive", "f1"): 0.01}

    result = ParetoKnapsackOptimizer(config).optimize(facets, passages, p_values)

    assert [passage.pid for passage in result.selected_passages] == ["cheap"]
    assert config.max_evidence_tokens is not None
    assert result.total_cost <= config.max_evidence_tokens


def test_only_validated_pareto_presets_are_exposed() -> None:
    assert available_presets() == (
        "pareto_match_1500_alpha06",
        "pareto_match_500_alpha06",
        "pareto_match_400",
    )
