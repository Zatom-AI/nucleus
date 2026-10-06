"""Pure-torch neighbour search — a dependency-free fallback for `torch_cluster`.

`fast.py` builds the CT model's radius graph and, by default, delegates the
within-cutoff neighbour query to the compiled `torch_cluster` kernels
(`radius_graph` for molecules, `radius` for periodic crystals). Those are the
only `torch_cluster` uses in the whole AFD scoring path.

This module provides signature-compatible pure-torch replacements so AFD can run
without the `torch_cluster` native dependency. They are exercised in the model's
*default* graph-builder mode, in which the cutoff query is intentionally
over-requested and the real per-atom cap is applied downstream by the pure-torch
`get_max_neighbors_mask`. In that regime these functions are exact: they return
every ordered pair within the cutoff, so the resulting candidate edge set is
identical to `torch_cluster`'s before the (shared, downstream) cap and sort.

Distances are compared as squared distances against ``r**2`` (no `cdist` matmul
trick) to keep cutoff-boundary decisions as close to `torch_cluster` as possible.
Work is done per graph to bound memory (no global N x N for batched inputs).
"""

from __future__ import annotations

import torch
from torch import Tensor


def _pairs_within_cutoff(p_a: Tensor, p_b: Tensor, r: float):
    """Indices (a, b) with squared dist(p_a[a], p_b[b]) <= r**2, plus that dist²."""
    d2 = (p_a.unsqueeze(1) - p_b.unsqueeze(0)).pow(2).sum(-1)  # (na, nb)
    a, b = (d2 <= r * r).nonzero(as_tuple=True)
    return a, b, d2[a, b]


def _cap_nearest_per_target(
    target: Tensor, src: Tensor, d2: Tensor, max_num_neighbors: int
):
    """Keep, per `target`, the `max_num_neighbors` smallest-`d2` entries.

    No-op in the model's default path (there `max_num_neighbors` is over-requested
    to exceed every atom's candidate count), but implemented for faithfulness.
    """
    keep = torch.ones_like(target, dtype=torch.bool)
    for t in torch.unique(target):
        m = target == t
        if int(m.sum()) <= max_num_neighbors:
            continue
        idx = m.nonzero(as_tuple=True)[0]
        drop = idx[torch.argsort(d2[idx])[max_num_neighbors:]]
        keep[drop] = False
    return target[keep], src[keep], d2[keep]


def radius_graph(
    pos: Tensor,
    r: float,
    batch: Tensor,
    max_num_neighbors: int = 32,
    loop: bool = False,
    flow: str = "source_to_target",
) -> Tensor:
    """Pure-torch analogue of ``torch_cluster.radius_graph``.

    Returns ``edge_index`` (2, E); with ``flow="source_to_target"`` row 0 is the
    source (neighbour) and row 1 is the target (centre), and the per-target cap is
    ``max_num_neighbors`` (applied by nearest-distance here).
    """
    assert flow == "source_to_target", f"unsupported flow {flow!r}"
    device = pos.device
    src_all, dst_all = [], []
    for g in torch.unique(batch):
        idx = (batch == g).nonzero(as_tuple=True)[0]
        i, j, d2 = _pairs_within_cutoff(pos[idx], pos[idx], r)  # i=centre, j=neigh
        if not loop:
            keep = i != j
            i, j, d2 = i[keep], j[keep], d2[keep]
        if max_num_neighbors is not None:
            i, j, d2 = _cap_nearest_per_target(i, j, d2, max_num_neighbors)
        dst_all.append(idx[i])  # target / centre
        src_all.append(idx[j])  # source / neighbour
    src = (
        torch.cat(src_all)
        if src_all
        else torch.empty(0, dtype=torch.long, device=device)
    )
    dst = (
        torch.cat(dst_all)
        if dst_all
        else torch.empty(0, dtype=torch.long, device=device)
    )
    return torch.stack([src, dst], dim=0)


def radius(
    x: Tensor,
    y: Tensor,
    r: float,
    batch_x: Tensor,
    batch_y: Tensor,
    max_num_neighbors: int = 32,
) -> Tensor:
    """Pure-torch analogue of ``torch_cluster.radius``.

    For each point in ``y`` finds points in ``x`` within ``r`` (same batch).
    Returns ``pair`` (2, E): row 0 indexes ``y`` (the query/target), row 1 indexes
    ``x`` (the database/source). Self / zero-distance pairs are kept (callers
    filter them), matching ``torch_cluster``'s default (no self-loop removal).
    """
    device = x.device
    y_all, x_all = [], []
    for g in torch.unique(batch_y):
        iy = (batch_y == g).nonzero(as_tuple=True)[0]
        ix = (batch_x == g).nonzero(as_tuple=True)[0]
        qi, di, d2 = _pairs_within_cutoff(y[iy], x[ix], r)  # qi->y, di->x
        if max_num_neighbors is not None:
            qi, di, d2 = _cap_nearest_per_target(qi, di, d2, max_num_neighbors)
        y_all.append(iy[qi])
        x_all.append(ix[di])
    yi = torch.cat(y_all) if y_all else torch.empty(0, dtype=torch.long, device=device)
    xi = torch.cat(x_all) if x_all else torch.empty(0, dtype=torch.long, device=device)
    return torch.stack([yi, xi], dim=0)
