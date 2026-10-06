"""
Fast batched radius-graph builder.

Two internal branches:
  - **Non-periodic**: delegates straight to `torch_cluster.radius_graph`
    (C++ kd-tree-ish). One call replaces the per-system Python loop in
    `GraphGenerator`, which is the main source of slowness when AtomisticEval
    processes batches of small molecules.

  - **Periodic**: replicates each system's atoms across the necessary image
    cells (per-axis count derived from cell geometry, identical to the upstream
    `radius_graph_pbc`), then uses `torch_cluster.radius` to query each
    original atom against all replicas at once. The candidate enumeration is
    fully vectorised across the batch — no per-system Python loop.

The returned dictionary has the same keys (`edge_index`, `edge_distance`,
`edge_distance_vec`, `cell_offsets`, `neighbors`) as the existing
`GraphGenerator.__call__`, so this is a drop-in replacement.

Conventions match upstream and the compiled torchmdnet kernel:
  - `edge_index[0]` is the source (`j`), `edge_index[1]` is the target (`i`).
  - `edge_distance_vec[k] = pos[edge_index[0, k]] - pos[edge_index[1, k]] + offset`.
  - With `self_loops=True`, each atom carries a self-edge of distance 0.
  - Both directions of each pair are included (i→j and j→i), matching
    `include_transpose=True` on the compiled extension.
"""

from __future__ import annotations

import os
from typing import Optional

import torch
from torch import Tensor
from torch_geometric.data import Data

from . import torch_neighbors as _torch_neighbors
from .radius_graph_pbc import get_max_neighbors_mask

# Normal crystal batches use O(10-100) offsets. Keep generous headroom for unusual
# valid cells while preventing unbounded allocations in the replicated atom grid.
_MAX_LATTICE_OFFSETS = 1_000_000

try:  # torch_cluster is an optional native dependency (see torch_neighbors.py).
    import torch_cluster as _torch_cluster
except Exception:  # pragma: no cover - import guard
    _torch_cluster = None


def _neighbor_backend():
    """Return the neighbour-search backend (compiled ``torch_cluster`` or pure torch).

    Controlled by ``AFD_GRAPH_BACKEND``: ``auto`` (default — use ``torch_cluster``
    if installed, else pure torch), ``torch_cluster`` (force; error if missing), or
    ``torch`` (force the dependency-free fallback in ``torch_neighbors``). Both
    backends expose signature-compatible ``radius_graph`` and ``radius``.
    """
    choice = os.environ.get("AFD_GRAPH_BACKEND", "auto").lower()
    if choice == "torch":
        return _torch_neighbors
    if choice == "torch_cluster":
        if _torch_cluster is None:
            raise ImportError(
                "AFD_GRAPH_BACKEND=torch_cluster but torch_cluster is not installed"
            )
        return _torch_cluster
    return _torch_cluster if _torch_cluster is not None else _torch_neighbors


def _ensure_batch(data: Data) -> Tensor:
    if hasattr(data, "batch") and data.batch is not None:
        return data.batch
    return torch.zeros(data.pos.size(0), dtype=torch.long, device=data.pos.device)


def _ensure_natoms(data: Data, batch: Tensor) -> Tensor:
    if hasattr(data, "natoms") and data.natoms is not None:
        return data.natoms.to(device=batch.device)
    return torch.bincount(batch)


def _required_reps_per_axis(cell: Tensor, cutoff: float, pbc: Tensor) -> Tensor:
    """Per-system replication count along each axis (shape (n_graphs, 3)).

    For each axis, the smallest distance between adjacent planes is
    ``|cross(b, c)| / V`` (and cyclic permutations). The number of image
    cells we have to look at on each side of the central cell is
    ``ceil(cutoff * |cross(b, c)| / V)``.
    """
    a, b, c = cell[:, 0], cell[:, 1], cell[:, 2]
    cross_bc = torch.cross(b, c, dim=-1)
    cross_ca = torch.cross(c, a, dim=-1)
    cross_ab = torch.cross(a, b, dim=-1)
    vol = (a * cross_bc).sum(dim=-1, keepdim=True).abs()
    # Avoid division by zero on degenerate / unset cells.
    vol = vol.clamp(min=1e-12)

    inv_min_a = cross_bc.norm(dim=-1, keepdim=True) / vol
    inv_min_b = cross_ca.norm(dim=-1, keepdim=True) / vol
    inv_min_c = cross_ab.norm(dim=-1, keepdim=True) / vol

    reps = torch.ceil(cutoff * torch.cat([inv_min_a, inv_min_b, inv_min_c], dim=-1))
    # pbc may be (n_graphs, 3) or broadcastable; either way mask out non-periodic axes.
    if pbc.dim() == 1:
        pbc = pbc.unsqueeze(0).expand(reps.size(0), -1)
    return reps * pbc.to(reps.dtype)


def _build_lattice_offsets(max_reps: Tensor, device, dtype) -> Tensor:
    """Generate the full set of integer lattice offsets up to ``max_reps``.

    Returns a tensor of shape (n_offsets, 3) with integer entries.
    Includes the (0, 0, 0) origin.
    """
    if max_reps.numel() != 3 or not bool(torch.isfinite(max_reps).all()):
        raise ValueError(
            "Periodic lattice repetitions must contain three finite values, "
            f"got {max_reps.detach().cpu().tolist()}."
        )

    repetitions = tuple(int(value) for value in max_reps.detach().cpu().tolist())
    if any(value < 0 for value in repetitions):
        raise ValueError(
            f"Periodic lattice repetitions must be non-negative, got {repetitions}."
        )

    axis_sizes = tuple(2 * value + 1 for value in repetitions)
    n_offsets = axis_sizes[0] * axis_sizes[1] * axis_sizes[2]
    if n_offsets > _MAX_LATTICE_OFFSETS:
        raise ValueError(
            "Periodic radius graph requires "
            f"{n_offsets:,} lattice offsets for repetitions {repetitions}, "
            f"exceeding the safety limit of {_MAX_LATTICE_OFFSETS:,}. "
            "The periodic cell is likely degenerate or too small for the cutoff."
        )

    ranges = [
        torch.arange(-value, value + 1, device=device, dtype=dtype)
        for value in repetitions
    ]
    grid = torch.cartesian_prod(*ranges)
    return grid


@torch.no_grad()
def fast_radius_graph(
    data: Data,
    cutoff: float,
    max_neighbors: Optional[int] = 32,
    self_loops: bool = True,
    cap_to_nearest: bool = True,
) -> dict:
    """Compute the radius graph in one shot for a torch_geometric Batch.

    Args:
        data: torch_geometric Data / Batch with `pos` (N, 3) and `batch` (N,)
            attributes. For periodic systems, also `cell` (n_graphs, 3, 3),
            `pbc` (n_graphs, 3), and `natoms` (n_graphs,).
        cutoff: radius in Å.
        max_neighbors: cap per central atom. Pass ``None`` to disable the cap
            entirely (every candidate within ``cutoff`` is kept).
        self_loops: include (i, i) edges with zero distance.
        cap_to_nearest: how to apply the cap when ``max_neighbors`` is set.
            See ``FastGraphGenerator`` for a detailed discussion of the
            trade-off. Ignored when ``max_neighbors is None``.

    Returns:
        dict with the same keys produced by `models.graph_utils.GraphGenerator`.
    """
    pos = data.pos
    batch = _ensure_batch(data)
    natoms = _ensure_natoms(data, batch)
    n_graphs = int(natoms.shape[0])

    has_pbc = (
        hasattr(data, "pbc") and data.pbc is not None and bool(data.pbc.any().item())
    )

    if not has_pbc:
        return _fast_nonperiodic(
            pos, batch, n_graphs, cutoff, max_neighbors, self_loops, cap_to_nearest
        )

    return _fast_periodic(
        data, batch, natoms, n_graphs, cutoff, max_neighbors, self_loops, cap_to_nearest
    )


def _fast_nonperiodic(
    pos: Tensor,
    batch: Tensor,
    n_graphs: int,
    cutoff: float,
    max_neighbors: Optional[int],
    self_loops: bool,
    cap_to_nearest: bool,
) -> dict:
    # Three cap modes (see FastGraphGenerator docstring for the full trade-off):
    #   - max_neighbors=int, cap_to_nearest=True (default, "parity" path):
    #     over-request torch_cluster (no per-atom cap) and apply
    #     `get_max_neighbors_mask` (tolerance-based K-nearest, 0.01 Å² band).
    #     Slower but bit-identical to upstream `radius_graph_pbc`.
    #   - max_neighbors=int, cap_to_nearest=False ("fast strict" path):
    #     pass `max_neighbors` straight to `torch_cluster.radius_graph`, which
    #     caps in C++ traversal order (NOT distance-sorted). Skips
    #     `get_max_neighbors_mask` entirely.
    #   - max_neighbors=None ("no cap" path): set torch_cluster's cap to
    #     n_atoms so nothing is truncated, also skip `get_max_neighbors_mask`.
    # Self-loops are always appended AFTER the cap (loop=False on the C++
    # call) so they never displace a real neighbour — see fastgraph_notes §5.
    n_atoms = pos.size(0)

    fast_path = (max_neighbors is None) or (not cap_to_nearest)
    if fast_path:
        tc_max = n_atoms if max_neighbors is None else max_neighbors
    else:
        tc_max = max(max_neighbors * 4, n_atoms)

    edge_index = _neighbor_backend().radius_graph(
        pos,
        r=cutoff,
        batch=batch,
        loop=False,
        max_num_neighbors=tc_max,
        flow="source_to_target",
    )
    src, dst = edge_index[0], edge_index[1]
    edge_distance_vec = pos[src] - pos[dst]
    edge_distance_sq = (edge_distance_vec * edge_distance_vec).sum(dim=-1)
    edge_distance = edge_distance_sq.sqrt()

    if not fast_path:
        # Sort by target then apply the upstream cap so the edge set matches
        # `radius_graph_pbc` exactly (incl. degeneracy_tolerance=0.01).
        sort_idx = torch.argsort(dst, stable=True)
        src = src[sort_idx]
        dst = dst[sort_idx]
        edge_distance_vec = edge_distance_vec[sort_idx]
        edge_distance = edge_distance[sort_idx]
        edge_distance_sq = edge_distance_sq[sort_idx]

        natoms_per_graph = torch.bincount(batch, minlength=n_graphs).to(pos.device)
        keep_mask, _ = get_max_neighbors_mask(
            natoms=natoms_per_graph,
            index=dst,
            atom_distance=edge_distance_sq,
            max_num_neighbors_threshold=max_neighbors,
            enforce_max_strictly=False,
        )
        if not bool(keep_mask.all()):
            src = src[keep_mask]
            dst = dst[keep_mask]
            edge_distance_vec = edge_distance_vec[keep_mask]
            edge_distance = edge_distance[keep_mask]

    if self_loops:
        loop_idx = torch.arange(n_atoms, device=pos.device)
        src = torch.cat([src, loop_idx])
        dst = torch.cat([dst, loop_idx])
        edge_distance_vec = torch.cat(
            [
                edge_distance_vec,
                torch.zeros(n_atoms, 3, device=pos.device, dtype=pos.dtype),
            ],
            dim=0,
        )
        edge_distance = torch.cat(
            [edge_distance, torch.zeros(n_atoms, device=pos.device, dtype=pos.dtype)],
            dim=0,
        )

    edge_index = torch.stack([src, dst], dim=0)
    cell_offsets = torch.zeros(
        edge_index.size(1), 3, device=pos.device, dtype=pos.dtype
    )
    neighbors = torch.bincount(batch[dst], minlength=n_graphs)

    return {
        "edge_index": edge_index,
        "edge_distance": edge_distance,
        "edge_distance_vec": edge_distance_vec,
        "cell_offsets": cell_offsets,
        "offset_distances": cell_offsets,
        "neighbors": neighbors,
    }


def _fast_periodic(
    data: Data,
    batch: Tensor,
    natoms: Tensor,
    n_graphs: int,
    cutoff: float,
    max_neighbors: Optional[int],
    self_loops: bool,
    cap_to_nearest: bool,
) -> dict:
    pos = data.pos
    cell = data.cell.to(pos.dtype)  # (n_graphs, 3, 3)
    pbc = data.pbc.to(pos.device)  # (n_graphs, 3)
    device = pos.device

    # 1. Per-system replication counts along each axis.
    reps_per_axis = _required_reps_per_axis(cell, cutoff, pbc)  # (n_graphs, 3)
    max_reps = reps_per_axis.max(dim=0).values  # (3,)

    # 2. Integer lattice offsets — shape (K, 3).
    offsets_int = _build_lattice_offsets(max_reps, device=device, dtype=pos.dtype)
    n_offsets = offsets_int.size(0)

    # 3. Real-space offsets per system: (n_graphs, K, 3).
    real_offsets = torch.einsum("kj,gji->gki", offsets_int, cell)  # (n_graphs, K, 3)
    # Per-atom real offsets: (N_total, K, 3)
    real_offsets_per_atom = real_offsets[batch]

    # 4. Build replicated positions: each atom gets K image copies.
    # replica_pos has shape (N_total * K, 3).
    replica_pos = (pos.unsqueeze(1) + real_offsets_per_atom).reshape(-1, 3)
    replica_batch = batch.repeat_interleave(n_offsets)
    replica_to_atom = torch.arange(pos.size(0), device=device).repeat_interleave(
        n_offsets
    )
    replica_to_offset_idx = (
        torch.arange(n_offsets, device=device)
        .unsqueeze(0)
        .expand(pos.size(0), -1)
        .reshape(-1)
    )

    # 5. Query each original atom against all replicas within `cutoff`.
    # `radius` returns edge_index where row 0 indexes into `x` (query) and
    # row 1 indexes into `y` (database). We use x=replica_pos so that the
    # **target** (receiver, `i`) is the original atom and the **source**
    # (neighbour, `j`) is a replica. That matches the upstream convention.
    #
    # Cap-mode dispatch (see FastGraphGenerator docstring):
    #   - parity path (cap_to_nearest=True with max_neighbors set):
    #     oversample so nothing is silently dropped; tolerance cap applied
    #     downstream via `get_max_neighbors_mask`.
    #   - fast path (cap_to_nearest=False) or no-cap (max_neighbors=None):
    #     pass the cap (or a large value) straight to `torch_cluster.radius`
    #     so the C++ kernel does the truncation in traversal order, then
    #     skip `get_max_neighbors_mask` entirely.
    n_atoms_total = pos.size(0)
    fast_path = (max_neighbors is None) or (not cap_to_nearest)
    if max_neighbors is None:
        tc_max = max(n_atoms_total * max(n_offsets, 1), 1)
    elif fast_path:
        tc_max = max_neighbors
    else:
        tc_max = max(max_neighbors * max(n_offsets, 1), n_atoms_total)
    pair = _neighbor_backend().radius(
        x=replica_pos,
        y=pos,
        r=cutoff,
        batch_x=replica_batch,
        batch_y=batch,
        max_num_neighbors=tc_max,
    )
    # pair[0] = idx into y (= pos) — this is the target atom i
    # pair[1] = idx into x (= replica_pos) — this is the source replica
    target_i = pair[0]
    replica_idx = pair[1]
    source_j = replica_to_atom[replica_idx]  # back to original
    offset_idx_per_edge = replica_to_offset_idx[replica_idx]  # K index

    # 6. Build the edge vector with the PBC offset baked in.
    # delta = pos[source_j] + real_offset - pos[target_i]
    real_offset_per_edge = real_offsets[batch[target_i], offset_idx_per_edge]  # (E, 3)
    edge_distance_vec = pos[source_j] + real_offset_per_edge - pos[target_i]
    # Squared distance computed component-wise — same numeric path as
    # `radius_graph_pbc` so the downstream `get_max_neighbors_mask` sees the
    # same atom_distance values and resolves ties consistently.
    edge_distance_sq = (edge_distance_vec * edge_distance_vec).sum(dim=-1)
    edge_distance = edge_distance_sq.sqrt()

    # 7. Drop zero-distance pairs from the cap candidate pool. Upstream's
    # `radius_graph_pbc` does the same (line 280: `mask_not_same = d² > 1e-4`),
    # so its cap sees only non-self candidates. Periodic self-images at
    # non-zero distance ARE candidate edges and stay in the pool.
    edge_distance_sq = edge_distance_sq[:]  # alias for clarity below
    not_zero = edge_distance_sq > 1e-4
    target_i = target_i[not_zero]
    source_j = source_j[not_zero]
    edge_distance_vec = edge_distance_vec[not_zero]
    edge_distance = edge_distance[not_zero]
    edge_distance_sq = edge_distance_sq[not_zero]
    offset_idx_per_edge = offset_idx_per_edge[not_zero]

    # 8a. Apply the upstream max-neighbours cap. We reuse `get_max_neighbors_mask`
    # so the edge set matches `radius_graph_pbc` exactly. It expects edges
    # sorted by target index, with `atom_distance` supplied as squared distance
    # (matching upstream's choice). Skipped on the fast / no-cap paths — there
    # the C++ kernel above already capped (or didn't, if max_neighbors is None).
    if not fast_path:
        sort_idx = torch.argsort(target_i, stable=True)
        target_i = target_i[sort_idx]
        source_j = source_j[sort_idx]
        edge_distance_vec = edge_distance_vec[sort_idx]
        edge_distance = edge_distance[sort_idx]
        edge_distance_sq = edge_distance_sq[sort_idx]
        offset_idx_per_edge = offset_idx_per_edge[sort_idx]

        keep_mask, _neighbors = get_max_neighbors_mask(
            natoms=natoms,
            index=target_i,
            atom_distance=edge_distance_sq,
            max_num_neighbors_threshold=max_neighbors,
            enforce_max_strictly=False,
        )
        if not bool(keep_mask.all()):
            target_i = target_i[keep_mask]
            source_j = source_j[keep_mask]
            edge_distance_vec = edge_distance_vec[keep_mask]
            edge_distance = edge_distance[keep_mask]
            offset_idx_per_edge = offset_idx_per_edge[keep_mask]

    # 8b. Append a zero-distance self-loop per atom AFTER the cap, mirroring
    # `generate_graph` in upstream. This keeps the self-loop out of the cap's
    # candidate pool so it doesn't displace a real neighbour.
    if self_loops:
        n_atoms_total = pos.size(0)
        loop_idx = torch.arange(n_atoms_total, device=device)
        target_i = torch.cat([target_i, loop_idx])
        source_j = torch.cat([source_j, loop_idx])
        edge_distance_vec = torch.cat(
            [
                edge_distance_vec,
                torch.zeros(n_atoms_total, 3, device=device, dtype=pos.dtype),
            ]
        )
        edge_distance = torch.cat(
            [edge_distance, torch.zeros(n_atoms_total, device=device, dtype=pos.dtype)]
        )
        # offset_idx_per_edge for self-loops: use the (0,0,0) lattice point's
        # index. By construction _build_lattice_offsets includes the origin;
        # find its position.
        origin_idx = int(
            torch.nonzero((offsets_int == 0).all(dim=1)).flatten()[0].item()
        )
        offset_idx_per_edge = torch.cat(
            [
                offset_idx_per_edge,
                torch.full(
                    (n_atoms_total,), origin_idx, dtype=torch.long, device=device
                ),
            ]
        )

    # 8c. edge_index — convention: row 0 source, row 1 target.
    edge_index = torch.stack([source_j, target_i], dim=0)

    # 9. cell_offsets (integer lattice coordinates, (E, 3)).
    cell_offsets = offsets_int[offset_idx_per_edge].to(pos.dtype)

    # 10. Per-graph edge counts (count by target).
    neighbors = torch.bincount(batch[target_i], minlength=n_graphs)

    # 11. Real-space offset distances (for parity with upstream output).
    offset_distances = real_offsets[batch[target_i], offset_idx_per_edge]

    return {
        "edge_index": edge_index,
        "edge_distance": edge_distance,
        "edge_distance_vec": edge_distance_vec,
        "cell_offsets": cell_offsets,
        "offset_distances": offset_distances,
        "neighbors": neighbors,
    }


class FastGraphGenerator:
    """Drop-in replacement for `models.graph_utils.GraphGenerator`.

    Same call signature; same output keys. Uses `fast_radius_graph` under the
    hood. Constructed once and reused across calls.

    Cap modes
    ---------
    The per-atom ``max_neighbors`` cap can be applied three ways, selected by
    ``max_neighbors`` and ``cap_to_nearest``. All three modes are deterministic
    (same input → bit-identical output, run-to-run); they differ only in
    *which* edges survive when an atom has more candidates than the cap allows.

    1. ``max_neighbors=K, cap_to_nearest=True`` (default).
       Tolerance-based K-nearest cap, applied via the upstream
       ``get_max_neighbors_mask`` helper with ``degeneracy_tolerance=0.01 Å²``.
       For an atom with more than ``K`` neighbours, the K closest are kept,
       plus any extras whose squared distance is within 0.01 Å² of the K-th.
       This is the upstream ``radius_graph_pbc`` convention; edges are
       **bit-identical to upstream** on every (cutoff, max_n, dataset) combo
       we tested (see ``fastgraph_validation.ipynb``).

       Pros: parity with upstream; stable under unit-cell / coordinate
       reparametrisation when atoms have near-degenerate neighbours at the
       cap boundary (common in dense crystals).
       Cons: slowest mode. ``get_max_neighbors_mask`` is the dominant cost on
       periodic batches (~6 ms / round on the QM9 benchmark; see
       ``fastgraph_notes.md`` §3a for the per-step breakdown).

    2. ``max_neighbors=K, cap_to_nearest=False`` — fast strict cap.
       The cap is delegated to the C++ kernel: ``torch_cluster.radius_graph``
       (non-periodic) or ``torch_cluster.radius`` (periodic) is called with
       ``max_num_neighbors=K`` directly. ``get_max_neighbors_mask`` is
       skipped entirely.

       Important: ``torch_cluster`` does **not** sort by distance before
       truncating — it fills a fixed-size buffer in spatial-traversal order
       and drops further candidates. So when the cap binds, the K kept are
       the K ``torch_cluster`` happened to find first, NOT the K closest.
       In particular this means:

         - Different from upstream when the cap binds (typically only on
           dense periodic systems / large coordination numbers).
         - Not invariant under perturbations that flip traversal order:
           reordering atoms in the input batch can change which K survive.
           For chemistry data this still gives sensible local neighbourhoods,
           but it can introduce non-determinism with respect to upstream
           ``mol_emb`` outputs on cap-bound atoms.

       Pros: substantially faster, especially on non-periodic batches (no
       Python-side sort / mask / index_copy pass). On the v1 benchmark this
       was ~12× to ~37× faster than the legacy ``GraphGenerator``;
       ``cap_to_nearest=True`` is ~12× on QM9, ~1.8× on matbench. Choose
       this mode when you have a CUDA-graph-style workload, a tight inference
       budget, or otherwise know the cap won't bind on your data (e.g. small
       molecules at cutoff=5 / K=32).
       Cons: not bit-identical to upstream when the cap binds; the dropped
       neighbours may not be the farthest ones.

    3. ``max_neighbors=None`` — no cap.
       Every candidate within ``cutoff`` is kept; ``get_max_neighbors_mask``
       is skipped. ``cap_to_nearest`` is ignored in this mode.

       Pros: fastest path that still produces upstream-parity edges. The
       pre-cap candidate pools are bit-identical to ``radius_graph_pbc``
       (verified in ``fastgraph_notes.md`` §5), so if the cap is what you'd
       otherwise be debugging, ``None`` removes that whole class of
       disagreement.
       Cons: edge counts can blow up on dense periodic systems — memory and
       downstream message-passing compute scale linearly with edge count.
       Also note that the trained ``CT`` checkpoint was fit with
       ``max_neighbors=32``; running inference with ``None`` changes the
       receptive field and may shift the model's predictions even though the
       edge set is more complete.

    Self-loops are always appended **after** the cap on every mode (fixing
    the v1 displacement bug; see ``fastgraph_notes.md`` §5).
    """

    required_keys = ["pos", "cell", "natoms", "pbc"]

    def __init__(
        self,
        cutoff: float,
        max_neighbors: Optional[int] = 32,
        self_loops: bool = True,
        alt_cell_key: str = "box",
        cap_to_nearest: bool = True,
        # Accepted but ignored — kept so this is a literal drop-in replacement
        # for `GraphGenerator`. The fast builder always uses the same
        # algorithm regardless of these.
        neighbor_method: str = "brute",
        enforce_max_neighbors_strictly: bool = False,
    ):
        self.cutoff = cutoff
        self.max_neighbors = max_neighbors
        self.self_loops = self_loops
        self.alt_cell_key = alt_cell_key
        self.cap_to_nearest = cap_to_nearest

    def __call__(self, batch) -> dict:
        if isinstance(batch, dict):
            batch = Data.from_dict(batch)
        if not hasattr(batch, "cell") and hasattr(batch, self.alt_cell_key):
            setattr(batch, "cell", getattr(batch, self.alt_cell_key))
        if not hasattr(batch, "natoms"):
            setattr(batch, "natoms", torch.bincount(batch.batch))
        for key in self.required_keys:
            if not hasattr(batch, key):
                raise ValueError(
                    f"Batch is missing required key '{key}' for graph generation"
                )
        return fast_radius_graph(
            batch,
            self.cutoff,
            self.max_neighbors,
            self.self_loops,
            self.cap_to_nearest,
        )


__all__ = ["fast_radius_graph", "FastGraphGenerator"]
