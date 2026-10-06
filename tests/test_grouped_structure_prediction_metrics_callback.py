from types import SimpleNamespace

import pytest


def _records(*, domain="molecules", generated, reference):
    return {
        "domain": domain,
        "generated": generated,
        "reference": reference,
    }


def test_grouped_zevals_merge_discards_cross_rank_sampler_padding() -> None:
    """A padded target keeps one complete candidate/reference record."""
    from nucleus.callbacks.metrics_logging import (
        LogGroupedZevalsStructurePredictionMetricsCallback,
    )

    payloads = [
        {
            "omol25_structure_prediction": _records(
                generated={"target-000": ["primary-000"], "target-016": ["p16"]},
                reference={"target-000": ["ref-000"], "target-016": ["ref-016"]},
            )
        },
        {
            "omol25_structure_prediction": _records(
                generated={"target-000": ["padding-000"], "target-017": ["p17"]},
                reference={"target-000": ["ref-000"], "target-017": ["ref-017"]},
            )
        },
    ]

    merged = LogGroupedZevalsStructurePredictionMetricsCallback._merge_payloads(
        payloads
    )["omol25_structure_prediction"]

    assert dict(merged["generated"]) == {
        "target-000": ["primary-000"],
        "target-016": ["p16"],
        "target-017": ["p17"],
    }
    assert merged["reference"] == {
        "target-000": ["ref-000"],
        "target-016": ["ref-016"],
        "target-017": ["ref-017"],
    }


def test_grouped_zevals_merge_rejects_incomplete_target_records() -> None:
    """Candidate and reference keys must remain paired during aggregation."""
    from nucleus.callbacks.metrics_logging import (
        LogGroupedZevalsStructurePredictionMetricsCallback,
    )

    payloads = [
        {
            "omol25_structure_prediction": _records(
                generated={"target-000": ["candidate"]},
                reference={},
            )
        }
    ]

    with pytest.raises(ValueError, match="target IDs do not match"):
        LogGroupedZevalsStructurePredictionMetricsCallback._merge_payloads(payloads)


@pytest.mark.parametrize("rank", [0, 1])
def test_grouped_zevals_rank_zero_failure_is_raised_on_every_rank(
    monkeypatch: pytest.MonkeyPatch,
    rank: int,
) -> None:
    """Every rank observes an exception before leaving the callback."""
    import nucleus.callbacks.metrics_logging as metrics_logging

    trainer = SimpleNamespace(fabric=SimpleNamespace(global_rank=rank, world_size=2))
    local_error = ValueError("invalid grouped payload") if rank == 0 else None

    def fake_broadcast_object_list(payload, src):
        assert src == 0
        if rank == 0:
            assert payload == [
                {"type": "ValueError", "message": "invalid grouped payload"}
            ]
        else:
            payload[0] = {
                "type": "ValueError",
                "message": "invalid grouped payload",
            }

    monkeypatch.setattr(
        metrics_logging.torch.distributed,
        "broadcast_object_list",
        fake_broadcast_object_list,
    )

    expected_error = ValueError if rank == 0 else RuntimeError
    with pytest.raises(expected_error, match="invalid grouped payload"):
        metrics_logging.LogGroupedZevalsStructurePredictionMetricsCallback._synchronize_rank_zero_exception(
            trainer,
            local_error,
            operation="merging grouped zevals validation records",
        )


def test_grouped_zevals_nonzero_rank_waits_for_rank_zero_compute(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Nonzero ranks synchronize even though only rank zero computes metrics."""
    import nucleus.callbacks.metrics_logging as metrics_logging

    callback = metrics_logging.LogGroupedZevalsStructurePredictionMetricsCallback(
        enabled=True
    )
    trainer = SimpleNamespace(fabric=SimpleNamespace(global_rank=1, world_size=2))
    monkeypatch.setattr(callback, "_gather_to_rank_zero", lambda trainer: (False, {}))
    broadcasts = []

    def fake_broadcast_object_list(payload, src):
        broadcasts.append((payload.copy(), src))

    monkeypatch.setattr(
        metrics_logging.torch.distributed,
        "broadcast_object_list",
        fake_broadcast_object_list,
    )

    callback.on_validation_epoch_end(trainer)

    assert broadcasts == [([None], 0)]


def test_grouped_zevals_rank_zero_compute_failure_is_synchronized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A scoring failure is broadcast before rank zero reraises it."""
    import nucleus.callbacks.metrics_logging as metrics_logging

    callback = metrics_logging.LogGroupedZevalsStructurePredictionMetricsCallback(
        enabled=True
    )
    trainer = SimpleNamespace(fabric=SimpleNamespace(global_rank=0, world_size=2))
    monkeypatch.setattr(
        callback,
        "_gather_to_rank_zero",
        lambda trainer: (True, {"omol25_structure_prediction": {}}),
    )

    def fail_compute(records):
        del records
        raise ValueError("scoring failed")

    monkeypatch.setattr(callback, "_compute_dataset_metrics", fail_compute)
    broadcasts = []

    def fake_broadcast_object_list(payload, src):
        broadcasts.append((payload.copy(), src))

    monkeypatch.setattr(
        metrics_logging.torch.distributed,
        "broadcast_object_list",
        fake_broadcast_object_list,
    )

    with pytest.raises(ValueError, match="scoring failed"):
        callback.on_validation_epoch_end(trainer)

    assert broadcasts == [([{"type": "ValueError", "message": "scoring failed"}], 0)]
