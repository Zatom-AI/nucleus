"""Tests for human-readable evaluation reports."""

from zevals.utils.reporting import print_categorized_report


def test_print_categorized_report_groups_task_metric_namespaces(capsys):
    print_categorized_report(
        {
            "de_novo/geometry/clash_free_rate": 0.75,
            "de_novo/geometry/chain_continuity_rate": 1.0,
            "de_novo/novelty/novel_rate": 0.5,
        },
        title="RNA results",
    )

    output = capsys.readouterr().out
    assert "RNA results" in output
    assert output.count("de_novo/geometry") == 1
    assert output.count("de_novo/novelty") == 1
    assert "clash_free_rate" in output
    assert "novel_rate" in output
