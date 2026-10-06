"""Vendored LigandMPNN inference code (MIT License, Copyright (c) 2024 Justas
Dauparas — https://github.com/dauparas/LigandMPNN).

Private to `zevals.oracles.inverse_folders.ligand_mpnn`; import the
public `LigandMPNNFolder` oracle instead of reaching into this package.

Only the `ligand_mpnn` model_type is vendored here. The LigandMPNN repo's
`protein_mpnn`/`soluble_mpnn` model_types share the exact same architecture
(and checkpoint schema — verified: `num_edges`/`model_state_dict` keys,
128-dim hidden/node/edge features, 3 encoder/decoder layers, byte-identical
`ProteinFeatures` math) already vendored under
`oracles.inverse_folders._protein_mpnn` — checkpoints from this repo's
`get_model_params.sh` (`proteinmpnn_v_48_0XX.pt`, `solublempnn_v_48_0XX.pt`)
load directly into that module's `ProteinMPNN` class, no new code needed.
"""
