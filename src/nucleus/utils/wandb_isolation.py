"""Keep inherited W&B service handles from stalling forked child processes."""

from __future__ import annotations

import os

_installed = False


def _disable_inherited_service_calls() -> None:
    """Turn this child's inherited service requests into no-ops."""
    try:
        from wandb.sdk.lib.service.service_connection import ServiceConnection
    except ImportError:
        return

    ServiceConnection.api_cleanup_request = lambda self, api_id: None
    ServiceConnection.teardown = lambda self, exit_code: None


def isolate_wandb_service_from_forked_children() -> None:
    """Stop forked children from waiting on replies only the parent can receive.

    A dataloader worker inherits the parent's W&B service connection but not
    the asyncio thread that drives it. Any request the child then makes — a
    garbage-collected API handle running its cleanup finalizer, or the atexit
    teardown — blocks forever, which stalls that worker, its DDP rank, and
    finally the whole job when the other ranks time out in a collective.
    """
    global _installed
    if not _installed:
        os.register_at_fork(after_in_child=_disable_inherited_service_calls)
        _installed = True
