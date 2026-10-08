"""The W&B service handle must not block a forked dataloader worker."""

from __future__ import annotations

import multiprocessing

from wandb.sdk.lib.service import service_connection

from nucleus.utils import wandb_isolation


def _report_cleanup_blocks(queue) -> None:
    connection = service_connection.ServiceConnection.__new__(
        service_connection.ServiceConnection
    )
    if hasattr(connection, "api_cleanup_request"):
        connection.api_cleanup_request("api-id")
    connection.teardown(0)
    queue.put("returned")


def test_forked_children_do_not_wait_on_the_service(monkeypatch):
    """In the child, an inherited cleanup request returns instead of hanging."""

    def _hang(self, *args, **kwargs):
        raise AssertionError("the child sent a request to the parent's service")

    monkeypatch.setattr(
        service_connection.ServiceConnection,
        "api_cleanup_request",
        _hang,
        raising=False,
    )
    monkeypatch.setattr(service_connection.ServiceConnection, "teardown", _hang)
    monkeypatch.setattr(wandb_isolation, "_installed", False)
    wandb_isolation.isolate_wandb_service_from_forked_children()

    context = multiprocessing.get_context("fork")
    queue = context.Queue()
    child = context.Process(target=_report_cleanup_blocks, args=(queue,))
    child.start()
    child.join(timeout=30)

    assert child.exitcode == 0
    assert queue.get(timeout=5) == "returned"


def test_parent_behaviour_is_unchanged(monkeypatch):
    """Only children are neutralised; the parent keeps its real connection."""
    monkeypatch.setattr(wandb_isolation, "_installed", False)
    original_teardown = getattr(
        service_connection.ServiceConnection, "teardown", None
    )
    original_cleanup = getattr(
        service_connection.ServiceConnection, "api_cleanup_request", None
    )

    wandb_isolation.isolate_wandb_service_from_forked_children()

    assert (
        getattr(service_connection.ServiceConnection, "teardown", None)
        is original_teardown
    )
    assert (
        getattr(service_connection.ServiceConnection, "api_cleanup_request", None)
        is original_cleanup
    )
