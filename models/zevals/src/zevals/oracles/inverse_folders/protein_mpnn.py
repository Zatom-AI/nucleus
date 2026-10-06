"""ProteinMPNN / SolubleMPNN / CA-ProteinMPNN inverse-folding oracle.

Runs inference in-process using the model architecture vendored under
`_protein_mpnn/` (ported from https://github.com/dauparas/ProteinMPNN,
MIT License). No subprocess, conda env, or cloned repo is needed at
runtime — only a checkpoint directory and the `zevals[proteins]`
extra (torch + numpy).

Checkpoints are not shipped with zevals — run
`zevals setup --checkpoints protein_mpnn_{vanilla,soluble,ca}` to fetch them
(all three are bundled directly in the ProteinMPNN repo, no license needed)
to the standard cache location `zevals.setup.default_checkpoint_dir()`
already falls back to. Or point `weights_dir` explicitly at any
`*_model_weights/` folder you already have on disk.

The model is loaded once (lazily, on first `design()` call) and cached on
`self._model` — construct this class once (e.g. via `ZevalSuite`, which does
this for you) and reuse it across an entire training run rather than
re-instantiating it every validation epoch.

Hydra config: configs/oracle/inverse_folder/protein_mpnn.yaml
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from zevals.oracles.base import BaseInverseFolder, InverseFoldResult, write_fasta
from zevals.oracles.inverse_folders._protein_mpnn.featurize import (
    ALPHABET,
    decode_sequence,
    featurize_backbone,
    parse_pdb,
    sequence_scores,
)
from zevals.oracles.inverse_folders._protein_mpnn.model import ProteinMPNN
from zevals.utils.devices import resolve_device

ModelVariant = Literal["vanilla", "soluble", "ca_only"]

_VARIANT_TO_WEIGHTS_DIRNAME: dict[ModelVariant, str] = {
    "vanilla": "vanilla_model_weights",
    "soluble": "soluble_model_weights",
    "ca_only": "ca_model_weights",
}
_HIDDEN_DIM = 128
_NUM_LAYERS = 3


class ProteinMPNNFolder(BaseInverseFolder):
    """In-process ProteinMPNN / SolubleMPNN / CA-ProteinMPNN inverse folder.

    Hydra config: oracle/inverse_folder/protein_mpnn.yaml
      _target_:       zevals.oracles.inverse_folders.protein_mpnn.ProteinMPNNFolder
      weights_dir:    /path/to/{vanilla,soluble,ca}_model_weights   # folder holding v_48_0XX.pt
      model_name:     v_48_020      # v_48_002 | v_48_010 | v_48_020 | v_48_030
      model_variant:  vanilla       # vanilla | soluble | ca_only
      num_sequences:  8
      temperature:    0.1
      backbone_noise:  0.02
      batch_size:     null          # null = sample all num_sequences in one batch
      device:         null          # null = cuda if available, else cpu

    `ca_only` should be used whenever the input backbones are Cα-only
    traces (the common case for generated backbones evaluated against the
    designability metric in proteins/METRICS.md) — it selects both the
    CA-only featurizer and the checkpoint trained for it.
    """

    def __init__(
        self,
        weights_dir: str | None = None,
        model_name: str = "v_48_020",
        model_variant: ModelVariant = "vanilla",
        batch_size: int | None = None,
        device: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.weights_dir = Path(weights_dir).expanduser() if weights_dir else None
        self.model_name = model_name
        self.model_variant = model_variant
        self.batch_size = batch_size
        self.device = torch.device(resolve_device(device))
        self._model: ProteinMPNN | None = None
        self._model_key: tuple[str, str] | None = None

    def design(self, pdb_path: Path, output_dir: Path) -> InverseFoldResult:
        """Sample `num_sequences` designs for the backbone(s) in `pdb_path`.

        Every chain present in the PDB is treated as designable (no fixed
        positions), matching upstream ProteinMPNN's behaviour when no
        `--chain_id_jsonl` is supplied.

        Args:
            pdb_path:   Input backbone file, PDB or CIF (single structure/
                        model). `_protein_mpnn`'s vendored parser is a
                        hand-rolled fixed-column PDB text reader with no CIF
                        analog (it's a faithful port of upstream
                        ProteinMPNN's own parser — not being rewritten), so
                        CIF input is converted to a temp PDB first via
                        `utils.io.convert_cif_to_pdb` rather than parsed
                        directly.
            output_dir: Directory to write a debug FASTA of the designs to
                        (and, for CIF input, the intermediate converted PDB).

        Returns:
            InverseFoldResult with `num_sequences` designed sequences and
            their per-sequence average negative log-likelihood scores
            (lower = more confident under the model).
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        ca_only = self.model_variant == "ca_only"

        if pdb_path.suffix == ".cif":
            from zevals.utils.io import convert_cif_to_pdb

            pdb_path = convert_cif_to_pdb(pdb_path, output_dir / f"{pdb_path.stem}.pdb")

        model = self._get_model(ca_only=ca_only)
        backbone = parse_pdb(pdb_path, ca_only=ca_only)
        feats = featurize_backbone(backbone, self.device, ca_only=ca_only)

        omit_AAs_np = np.array([aa == "X" for aa in ALPHABET], dtype=np.float32)
        bias_AAs_np = np.zeros(len(ALPHABET), dtype=np.float32)

        batch_size = self.batch_size or self.num_sequences
        sequences: list[str] = []
        scores: list[float] = []
        with torch.no_grad():
            while len(sequences) < self.num_sequences:
                b = min(batch_size, self.num_sequences - len(sequences))
                X = feats.X.repeat(b, *([1] * (feats.X.dim() - 1)))
                S = feats.S.repeat(b, 1)
                mask = feats.mask.repeat(b, 1)
                chain_M = feats.chain_M.repeat(b, 1)
                chain_M_pos = feats.chain_M_pos.repeat(b, 1)
                chain_encoding_all = feats.chain_encoding_all.repeat(b, 1)
                residue_idx = feats.residue_idx.repeat(b, 1)
                omit_AA_mask = feats.omit_AA_mask.repeat(b, 1, 1)
                bias_by_res = feats.bias_by_res.repeat(b, 1, 1)

                randn = torch.randn(chain_M.shape, device=self.device)
                sample_dict = model.sample(
                    X,
                    randn,
                    S,
                    chain_M,
                    chain_encoding_all,
                    residue_idx,
                    mask=mask,
                    temperature=self.temperature,
                    omit_AAs_np=omit_AAs_np,
                    bias_AAs_np=bias_AAs_np,
                    chain_M_pos=chain_M_pos,
                    omit_AA_mask=omit_AA_mask,
                    bias_by_res=bias_by_res,
                )
                S_sample = sample_dict["S"]

                log_probs = model(
                    X,
                    S_sample,
                    mask,
                    chain_M * chain_M_pos,
                    residue_idx,
                    chain_encoding_all,
                    randn,
                )
                mask_for_loss = mask * chain_M * chain_M_pos
                batch_scores = sequence_scores(
                    S_sample, log_probs, mask_for_loss
                ).tolist()

                for i in range(b):
                    sequences.append(decode_sequence(S_sample[i], feats.chain_lengths))
                    scores.append(batch_scores[i])

        write_fasta(
            output_dir / f"{backbone.name}.fa", backbone.name, sequences, scores
        )
        return InverseFoldResult(
            sequences=sequences,
            scores=scores,
            extra={"model_name": self.model_name, "model_variant": self.model_variant},
        )

    def _get_model(self, ca_only: bool) -> ProteinMPNN:
        """Load (and cache) the checkpoint for the current model_name/variant."""
        key = (self.model_name, self.model_variant)
        if self._model is not None and self._model_key == key:
            return self._model

        weights_dir = self._resolve_weights_dir()
        checkpoint_path = weights_dir / f"{self.model_name}.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"No checkpoint at {checkpoint_path}. Check `weights_dir` and `model_name`."
            )

        checkpoint = torch.load(
            checkpoint_path, map_location=self.device, weights_only=False
        )
        model = ProteinMPNN(
            ca_only=ca_only,
            num_letters=21,
            node_features=_HIDDEN_DIM,
            edge_features=_HIDDEN_DIM,
            hidden_dim=_HIDDEN_DIM,
            num_encoder_layers=_NUM_LAYERS,
            num_decoder_layers=_NUM_LAYERS,
            augment_eps=self.backbone_noise,
            k_neighbors=checkpoint["num_edges"],
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)
        model.eval()
        model.requires_grad_(False)

        self._model, self._model_key = model, key
        return model

    def _resolve_weights_dir(self) -> Path:
        if self.weights_dir is not None:
            return self.weights_dir

        # No explicit weights_dir — fall back to the standard per-user
        # checkpoint cache `zevals setup --checkpoints ...` downloads to.
        # Only used as a fallback, never silently preferred over an
        # explicit weights_dir. Two cases, matching zevals.setup._CHECKPOINTS:
        #   - soluble_mpnn.yaml's `solublempnn_v_48_0XX` naming (LigandMPNN-
        #     repo-hosted checkpoint, kept in its own cache subdir since it's
        #     a differently-named file, not one of ProteinMPNN's own
        #     vanilla/soluble/ca_only _model_weights/ directories).
        #   - Everything else: ProteinMPNN's own repo-bundled checkpoints,
        #     one subdirectory per model_variant.
        from zevals.setup import default_checkpoint_dir

        if self.model_name.startswith("solublempnn_"):
            fallback = default_checkpoint_dir() / "soluble_mpnn"
            fetch_hint = "soluble_mpnn"
        else:
            fallback = (
                default_checkpoint_dir()
                / "protein_mpnn"
                / _VARIANT_TO_WEIGHTS_DIRNAME[self.model_variant]
            )
            fetch_hint = f"protein_mpnn_{'ca' if self.model_variant == 'ca_only' else self.model_variant}"

        if (fallback / f"{self.model_name}.pt").is_file():
            return fallback

        raise ValueError(
            "weights_dir must be set to a folder containing "
            f"{self.model_name}.pt (e.g. ProteinMPNN's "
            f"{_VARIANT_TO_WEIGHTS_DIRNAME[self.model_variant]}/, or a "
            "checkpoint directory downloaded separately for the 'vanilla' "
            f"variant) — none found at the default cache location either "
            f"({fallback}). Run `zevals setup --checkpoints {fetch_hint}` "
            "to fetch it there."
        )
