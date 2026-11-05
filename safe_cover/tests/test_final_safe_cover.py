from trident.calibration import ReliabilityCalibrator
from trident.candidates import Passage
from trident.facets import Facet, FacetType
from trident.presets import available_presets, get_preset
from trident.safe_cover import SafeCoverAlgorithm


def test_final_preset_produces_a_certificate() -> None:
    config = get_preset()
    calibrator = ReliabilityCalibrator(version="release-test")
    calibrator.tables["RELATION"] = [(0.0, 1.0), (0.9, 0.01), (1.0, 0.0)]
    facet = Facet("f1", FacetType.RELATION, {"predicate": "created_by"})
    passage = Passage("p1", "support", cost=10)

    result = SafeCoverAlgorithm(config, calibrator).run(
        [facet], [passage], {("p1", "f1"): 0.001}
    )

    assert [item.pid for item in result.selected_passages] == ["p1"]
    assert result.certificates[0].calibrator_version == "release-test"


def test_only_validated_safe_cover_presets_are_exposed() -> None:
    assert available_presets() == (
        "safe_cover_equal_2000",
        "safe_cover_loose_4000",
    )
