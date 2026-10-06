"""LigandMPNN inverse-folding oracle.

Runs inference in-process using the model architecture vendored under
`_ligand_mpnn/` (ported from https://github.com/dauparas/LigandMPNN, MIT
License). No subprocess or cloned repo is needed at runtime — only a
checkpoint file and the `zevals[proteins]` extra (torch, numpy, prody).

Unlike ProteinMPNN, LigandMPNN conditions sequence design on nearby
non-protein context atoms (ligands, metals, nucleic acids, cofactors — any
non-water HETATM/non-protein record already present in the input PDB), so
it's the better choice whenever the backbone being designed sits in a
pocket that should stay compatible with a bound small molecule.

For plain ProteinMPNN/SolubleMPNN checkpoints from this same repo's
`get_model_params.sh` bundle (`proteinmpnn_v_48_0XX.pt`,
`solublempnn_v_48_0XX.pt`), no new code is needed — they share the exact
architecture already vendored in `oracles.inverse_folders.protein_mpnn`;
just point that oracle's `weights_dir` at the downloaded bundle. See
`configs/oracle/inverse_folder/soluble_mpnn.yaml`.

Checkpoints are not shipped with zevals — run
`zevals setup --checkpoints ligand_mpnn` to fetch the default variant to
`zevals.setup.default_checkpoint_dir()` (already the fallback location
`checkpoint_path` resolves to when left unset). Or download others with
this repo's own `get_model_params.sh <dest_dir>` and point `checkpoint_path`
explicitly at the resulting `ligandmpnn_v_32_0XX_25.pt` (or `_16.pt`) file.

The model is loaded once (lazily, on first `design()` call) and cached on
`self._model` — construct this class once (e.g. via `ZevalSuite`, which does
this for you) and reuse it across an entire training run rather than
re-instantiating it every validation epoch.

Hydra config: configs/oracle/inverse_folder/ligand_mpnn.yaml
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from zevals.oracles.base import BaseInverseFolder, InverseFoldResult, write_fasta
from zevals.oracles.inverse_folders._ligand_mpnn.featurize import (
    featurize_ligand_backbone,
    parse_pdb_with_ligand,
)
from zevals.oracles.inverse_folders._ligand_mpnn.model import LigandMPNN
from zevals.oracles.inverse_folders._protein_mpnn.featurize import (
    decode_sequence,
    sequence_scores,
)
from zevals.utils.devices import resolve_device

_HIDDEN_DIM = 128
_NUM_LAYERS = 3


class LigandMPNNFolder(BaseInverseFolder):
    """In-process LigandMPNN inverse folder with ligand/metal/nucleic-acid context.

    Hydra config: oracle/inverse_folder/ligand_mpnn.yaml
      _target_:       zevals.oracles.inverse_folders.ligand_mpnn.LigandMPNNFolder
      checkpoint_path: /path/to/ligandmpnn_v_32_010_25.pt
      num_sequences:  8
      temperature:    0.1
      batch_size:     null          # null = sample all num_sequences in one batch
      device:         null          # null = cuda if available, else cpu

    Every chain in the input PDB is treated as designable (no fixed
    positions); every non-water heteroatom / non-protein atom already in
    the PDB is a candidate context atom, narrowed per-residue to the
    checkpoint's trained `atom_context_num` nearest neighbours.
    """

    def __init__(
        self,
        checkpoint_path: str | None = None,
        batch_size: int | None = None,
        device: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.checkpoint_path = (
            Path(checkpoint_path).expanduser() if checkpoint_path else None
        )
        self.batch_size = batch_size
        self.device = torch.device(resolve_device(device))
        self._model: LigandMPNN | None = None
        self._atom_context_num: int | None = None

    def design(self, pdb_path: Path, output_dir: Path) -> InverseFoldResult:
        """Sample `num_sequences` ligand-context-conditioned designs.

        Args:
            pdb_path:   Input PDB — protein backbone plus any HETATM/ligand
                        context atoms to condition on (same file).
            output_dir: Directory to write a debug FASTA of the designs to.

        Returns:
            InverseFoldResult with `num_sequences` designed sequences and
            their per-sequence average negative log-likelihood scores.
            `extra["num_context_atoms"]` reports how many real (non-padding)
            ligand-context atoms were found near the designed residues.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        model = self._get_model()

        backbone = parse_pdb_with_ligand(pdb_path)
        feats = featurize_ligand_backbone(backbone, self.device, self._atom_context_num)
        fd = feats.feature_dict

        batch_size = self.batch_size or self.num_sequences
        sequences: list[str] = []
        scores: list[float] = []
        with torch.no_grad():
            while len(sequences) < self.num_sequences:
                b = min(batch_size, self.num_sequences - len(sequences))
                batch_fd = dict(fd)
                batch_fd["batch_size"] = b
                batch_fd["temperature"] = self.temperature
                batch_fd["randn"] = torch.randn(
                    [b, fd["mask"].shape[1]], device=self.device
                )

                out = model.sample(batch_fd)
                mask_for_loss = (fd["mask"] * fd["chain_mask"]).repeat(b, 1)
                batch_scores = sequence_scores(
                    out["S"], out["log_probs"], mask_for_loss
                ).tolist()

                for i in range(b):
                    sequences.append(decode_sequence(out["S"][i], feats.chain_lengths))
                    scores.append(batch_scores[i])

        write_fasta(
            output_dir / f"{backbone.name}.fa", backbone.name, sequences, scores
        )
        return InverseFoldResult(
            sequences=sequences,
            scores=scores,
            extra={"num_context_atoms": feats.num_context_atoms},
        )

    def _get_model(self) -> LigandMPNN:
        """Lazily load (and cache) the checkpoint."""
        if self._model is not None:
            return self._model

        checkpoint_path = self.checkpoint_path
        if checkpoint_path is None:
            # No explicit checkpoint_path — fall back to the standard
            # per-user checkpoint cache `zevals setup --checkpoints
            # ligand_mpnn` downloads to. Only used as a fallback, never
            # silently preferred over an explicit checkpoint_path.
            from zevals.setup import default_checkpoint_dir

            fallback = (
                default_checkpoint_dir() / "ligand_mpnn" / "ligandmpnn_v_32_010_25.pt"
            )
            if fallback.is_file():
                checkpoint_path = fallback
            else:
                raise ValueError(
                    "checkpoint_path must be set to a LigandMPNN checkpoint "
                    "(e.g. ligandmpnn_v_32_010_25.pt from get_model_params.sh) "
                    f"— none found at the default cache location either "
                    f"({fallback}). Run `zevals setup --checkpoints ligand_mpnn` "
                    "to fetch it there."
                )
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"No checkpoint at {checkpoint_path}.")
        self.checkpoint_path = checkpoint_path

        checkpoint = torch.load(
            self.checkpoint_path, map_location=self.device, weights_only=False
        )
        atom_context_num = checkpoint["atom_context_num"]
        model = LigandMPNN(
            node_features=_HIDDEN_DIM,
            edge_features=_HIDDEN_DIM,
            hidden_dim=_HIDDEN_DIM,
            num_encoder_layers=_NUM_LAYERS,
            num_decoder_layers=_NUM_LAYERS,
            k_neighbors=checkpoint["num_edges"],
            atom_context_num=atom_context_num,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)
        model.eval()
        model.requires_grad_(False)

        self._model = model
        self._atom_context_num = atom_context_num
        return model
