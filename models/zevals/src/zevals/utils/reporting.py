"""Evaluation report generation.

Formats EvalResults into human-readable tables (rich) and machine-readable
JSON for downstream logging or comparison scripts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from zevals.pipeline.base import EvalResults


def _sanitize_for_json(obj: Any) -> Any:
    """Recursively replace non-finite floats (inf/-inf/nan) with None.

    `json.dumps` happily emits the bare tokens `Infinity`/`-Infinity`/`NaN`
    for these by default (valid Python, invalid JSON) — metrics like
    designability's `mean_min_scRMSD` are `inf` whenever nothing is
    designable, so this is a real, not hypothetical, case. `None` round-trips
    through strict JSON parsers (e.g. WandB, `JSON.parse`) as `null`.
    """
    if isinstance(obj, float):
        return obj if obj == obj and obj not in (float("inf"), float("-inf")) else None
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    return obj


def print_report(results: EvalResults, title: str = "Eval Results") -> None:
    """Print a formatted table of aggregate metrics to stdout via rich."""
    try:
        from rich.console import Console
        from rich.table import Table
    except ImportError as exc:
        raise ImportError("Install rich: pip install rich") from exc

    console = Console()
    table = Table(title=title, show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="dim")
    table.add_column("Value", justify="right")

    for k, v in sorted(results.aggregate.items()):
        table.add_row(k, f"{v:.4f}" if isinstance(v, float) else str(v))

    console.print(table)


def print_merged_report(
    merged: dict[str, float], title: str = "zevals Results"
) -> None:
    """Print a formatted table of a `ZevalSuite.evaluate()`-style merged
    metrics dict (keys already namespaced as `"{modality}/{metric_name}"`).

    Unlike `print_report`, this takes a plain flat dict rather than an
    `EvalResults`, since a `ZevalSuite` call spans multiple modalities/pipelines
    with no single `EvalResults` to summarise.
    """
    try:
        from rich.console import Console
        from rich.table import Table
    except ImportError as exc:
        raise ImportError("Install rich: pip install rich") from exc

    console = Console()
    table = Table(title=title, show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="dim")
    table.add_column("Value", justify="right")

    for k, v in sorted(merged.items()):
        table.add_row(k, f"{v:.4f}" if isinstance(v, float) else str(v))

    console.print(table)


def print_categorized_report(
    merged: dict[str, float], title: str = "zevals Results"
) -> None:
    """Print one table per task/metric category from namespaced metrics."""
    try:
        from rich.console import Console
        from rich.table import Table
    except ImportError as exc:
        raise ImportError("Install rich: pip install rich") from exc

    console = Console()
    categories: dict[str, list[tuple[str, Any]]] = {}
    for key, value in sorted(merged.items()):
        parts = key.split("/")
        category = "/".join(parts[:2]) if len(parts) >= 2 else "summary"
        metric = "/".join(parts[2:]) if len(parts) >= 3 else parts[-1]
        categories.setdefault(category, []).append((metric, value))

    console.print(f"[bold]{title}[/bold]")
    for category, rows in categories.items():
        table = Table(title=category, show_header=True, header_style="bold cyan")
        table.add_column("Metric", style="dim")
        table.add_column("Value", justify="right")
        for metric, value in rows:
            table.add_row(
                metric,
                f"{value:.4f}" if isinstance(value, float) else str(value),
            )
        console.print(table)


def save_json_dict(merged: dict[str, Any], path: Path) -> None:
    """Serialise a plain flat metrics dict (e.g. from `ZevalSuite.evaluate()`)
    to a JSON file — simpler than `save_json`, which expects an `EvalResults`.

    Args:
        merged: Flat `{metric_name: value}` dict.
        path:   Destination JSON file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_sanitize_for_json(merged), indent=2))


def save_json(results: EvalResults, path: Path) -> None:
    """Serialise EvalResults to a JSON file.

    Args:
        results: Output of any EvalPipeline.run() call.
        path:    Destination JSON file path.
    """
    payload: dict[str, Any] = {
        "aggregate": results.aggregate,
        "metadata": results.metadata,
        "per_sample": results.per_sample,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_sanitize_for_json(payload), indent=2))


def load_json(path: Path) -> EvalResults:
    """Deserialise an EvalResults JSON file."""
    payload = json.loads(path.read_text())
    return EvalResults(
        per_sample=payload.get("per_sample", []),
        aggregate=payload.get("aggregate", {}),
        metadata=payload.get("metadata", {}),
    )


def compare_runs(results: dict[str, EvalResults]) -> dict[str, dict[str, float]]:
    """Compare aggregate metrics across multiple named runs.

    Args:
        results: Mapping of run_name → EvalResults.

    Returns:
        Nested dict: metric_name → {run_name: value}.
    """
    all_keys: set[str] = set()
    for r in results.values():
        all_keys.update(r.aggregate.keys())

    comparison: dict[str, dict[str, float]] = {}
    for key in sorted(all_keys):
        comparison[key] = {
            name: r.aggregate.get(key, float("nan")) for name, r in results.items()
        }

    return comparison
