"""Check global aggregation of independently scored Zatom-1 chunks."""

from scripts.evaluate_geom_zatom1 import PB_KEYS, aggregate


def test_global_uniqueness_and_native_posebusters_denominators():
    passed = dict.fromkeys(PB_KEYS, True)
    disconnected = passed | {
        "all_atoms_connected": False,
        "posebusters_rate": False,
        "internal_energy": None,
    }
    rows = [
        {"valid": True, "smiles": "C", "checks": passed},
        {"valid": True, "smiles": "C", "checks": disconnected},
        {"valid": True, "smiles": "O", "checks": passed},
        {"valid": False, "smiles": None, "checks": {}},
    ]
    result = aggregate(rows, ["O"])

    assert result["rates"]["valid_rate"] == 3 / 4
    assert result["rates"]["unique_rate"] == 2 / 3
    assert result["rates"]["novel_rate"] == 1 / 3
    assert result["rates"]["posebusters_rate"] == 2 / 3
    assert result["rates"]["posebusters_valid_over_all_generated"] == 1 / 2
    assert result["rates"]["all_atoms_connected"] == 2 / 3
    assert result["denominators"]["internal_energy"] == 2
    assert result["counts"]["posebusters_missing_checks"]["internal_energy"] == 1


def test_unreadable_samples_remain_in_validity_denominator():
    rows = [{"valid": False, "checks": {}, "read_error": "bad coordinates"}]
    result = aggregate(rows, [])

    assert result["counts"]["read_errors"] == 1
    assert result["denominators"]["valid_rate"] == 1
    assert result["rates"]["valid_rate"] == 0
    assert result["rates"]["unique_rate"] is None
    assert result["rates"]["posebusters_rate"] is None


def test_empty_sample_list_has_no_defined_rates():
    result = aggregate([], [])

    assert result["counts"]["generated"] == 0
    assert all(rate is None for rate in result["rates"].values())
