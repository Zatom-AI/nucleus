"""NA-MPNN protein + DNA/RNA joint inverse-folding oracle.

Runs inference in-process using the model architecture vendored under
`_na_mpnn/` (ported from https://github.com/baker-laboratory/NA-MPNN, MIT
License). No subprocess or cloned repo is needed at runtime — only a
checkpoint file and the `zevals[proteins]` extra (torch, numpy, prody).

Only upstream's "design" mode is vendored (autoregressive full-backbone
sequence sampling for protein and/or DNA/RNA residues) — "specificity" mode
(predicting a protein-DNA binding PPM for a *fixed* protein context) is a
different task from inverse folding and isn't exposed via `BaseInverseFolder`.
Every chain in the input PDB is treated as designable, matching
`ProteinMPNNFolder`'s convention — point this at a generated RNA/DNA
backbone (optionally with a fixed protein partner threaded into the same
PDB as context) to design a sequence for it.

NA-MPNN's checkpoints were trained with DNA/RNA one-letter tokens sharing
the same softmax slots (`na_shared_tokens=1`, upstream's default) — see
`_na_mpnn/featurize.py` for how the model's DNA-token output is mapped back
to RNA one-letter codes post-hoc for originally-RNA positions. Designed
sequences use NA-MPNN's own non-standard one-letter alphabet (uppercase =
protein; lowercase a/c/g/t/x = DNA; lowercase b/d/h/u/y = RNA — see
`_na_mpnn/featurize.py`'s module docstring for the full mapping) rather than
the plain A/C/G/U used elsewhere in zevals.

Checkpoints are not shipped with zevals — run
`zevals setup --checkpoints na_mpnn_design` to fetch the design-mode
checkpoint (bundled directly in the NA-MPNN repo, no license needed) to
`zevals.setup.default_checkpoint_dir()` (already the fallback location
`checkpoint_path` resolves to when left unset).

The model is loaded once (lazily, on first `design()` call) and cached on
`self._model` — construct this class once (e.g. via `ZevalSuite`, which does
this for you) and reuse it across an entire training run rather than
re-instantiating it every validation epoch.

Hydra config: configs/oracle/inverse_folder/na_mpnn.yaml
"""

from __future__ import annotations

import hashlib
import hmac
from pathlib import Path
from typing import Any

import torch

from zevals.oracles.base import BaseInverseFolder, InverseFoldResult, write_fasta
from zevals.oracles.inverse_folders._na_mpnn.featurize import (
    ATOM_ORDER,
    POLYTYPE_TO_INT,
    RESTYPE_TO_INT,
    decode_sequence,
    featurize_na_backbone,
    parse_pdb_na,
)
from zevals.oracles.inverse_folders._na_mpnn.model import NAMPNN
from zevals.oracles.inverse_folders._protein_mpnn.featurize import sequence_scores
from zevals.utils.devices import resolve_device

_HIDDEN_DIM = 128
_NUM_LAYERS = 3
_NUM_LETTERS = 33
_K_NEIGHBORS = 32  # upstream's fixed default for model_type="na_mpnn"; not stored in the checkpoint.


class NAMPNNFolder(BaseInverseFolder):
    """In-process NA-MPNN inverse folder (protein + DNA/RNA joint design).

    Hydra config: oracle/inverse_folder/na_mpnn.yaml
      _target_:        zevals.oracles.inverse_folders.na_mpnn.NAMPNNFolder
      checkpoint_path: /path/to/s_19137.pt   # design-mode checkpoint
      num_sequences:   8
      temperature:     0.1
      batch_size:      null          # null = sample all num_sequences in one batch
      device:          null          # null = cuda if available, else cpu
    """

    def __init__(
        self,
        checkpoint_path: str | None = None,
        batch_size: int | None = None,
        device: str | None = None,
        seed: int = 20260812,
        expected_checkpoint_sha256: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.checkpoint_path = (
            Path(checkpoint_path).expanduser() if checkpoint_path else None
        )
        self.batch_size = batch_size
        self.device = torch.device(resolve_device(device))
        self.seed = seed
        self.expected_checkpoint_sha256 = expected_checkpoint_sha256
        self._checkpoint_validated = False
        self._model: NAMPNN | None = None

    def design(self, pdb_path: Path, output_dir: Path) -> InverseFoldResult:
        """Sample `num_sequences` designs for the residues in `pdb_path`.

        Args:
            pdb_path:   Input backbone file, PDB or CIF (dispatched by
                        `.suffix` in `parse_pdb_na`) — protein and/or
                        nucleic-acid chains, all treated as designable.
            output_dir: Directory to write a debug FASTA of the designs to.

        Returns:
            InverseFoldResult with `num_sequences` designed sequences
            (NA-MPNN's non-standard one-letter alphabet — see module
            docstring) and their per-sequence average negative
            log-likelihood scores.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        model = self._get_model()

        backbone = parse_pdb_na(pdb_path)
        feats = featurize_na_backbone(backbone, self.device)
        fd = feats.feature_dict

        batch_size = self.batch_size or self.num_sequences
        sequences: list[str] = []
        scores: list[float] = []
        sample_seed = self._seed_for_backbone(pdb_path)
        cuda_devices: list[int] = []
        if self.device.type == "cuda":
            cuda_devices = [
                self.device.index
                if self.device.index is not None
                else torch.cuda.current_device()
            ]
        # Model sampling uses the process-global torch RNG internally. Fork it
        # so each backbone receives a stable stream without perturbing callers.
        with torch.random.fork_rng(devices=cuda_devices), torch.no_grad():
            torch.manual_seed(sample_seed)
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
                    sequences.append(
                        decode_sequence(
                            out["S"][i],
                            feats.rna_mask_for_token_conversion,
                            feats.chain_letters,
                        )
                    )
                    scores.append(batch_scores[i])

        write_fasta(
            output_dir / f"{backbone.name}.fa", backbone.name, sequences, scores
        )
        return InverseFoldResult(
            sequences=sequences,
            scores=scores,
            extra={"seed": sample_seed},
        )

    def _seed_for_backbone(self, pdb_path: Path) -> int:
        """Derive a stable, checkpoint-independent RNG seed from a sample name."""
        payload = f"{self.seed}\0{pdb_path.name}".encode()
        digest = hashlib.blake2b(payload, digest_size=8).digest()
        return int.from_bytes(digest, byteorder="little") % (2**63 - 1)

    def _get_model(self) -> NAMPNN:
        """Lazily load (and cache) the checkpoint."""
        if self._model is not None:
            return self._model

        checkpoint_path = self.checkpoint_path
        if checkpoint_path is None:
            # No explicit checkpoint_path — fall back to the standard
            # per-user checkpoint cache `zevals setup --checkpoints
            # na_mpnn_design` downloads to. Only used as a fallback, never
            # silently preferred over an explicit checkpoint_path.
            from zevals.setup import default_checkpoint_dir

            fallback = default_checkpoint_dir() / "na_mpnn" / "s_19137.pt"
            if fallback.is_file():
                checkpoint_path = fallback
            else:
                raise ValueError(
                    "checkpoint_path must be set to NA-MPNN's design-mode "
                    "checkpoint (s_19137.pt) — none found at the default "
                    f"cache location either ({fallback}). Run `zevals setup "
                    "--checkpoints na_mpnn_design` to fetch it there."
                )
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"No checkpoint at {checkpoint_path}.")
        self.checkpoint_path = checkpoint_path.expanduser().resolve()
        if self.expected_checkpoint_sha256 and not self._checkpoint_validated:
            digest = hashlib.sha256()
            with self.checkpoint_path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            actual_sha256 = digest.hexdigest()
            if not hmac.compare_digest(actual_sha256, self.expected_checkpoint_sha256):
                raise RuntimeError(
                    "NA-MPNN checkpoint SHA-256 mismatch: expected "
                    f"{self.expected_checkpoint_sha256}, found {actual_sha256}"
                )
            self._checkpoint_validated = True

        checkpoint = torch.load(
            self.checkpoint_path, map_location=self.device, weights_only=False
        )
        model = NAMPNN(
            restype_to_int=RESTYPE_TO_INT,
            atom_dict=ATOM_ORDER,
            polytype_to_int=POLYTYPE_TO_INT,
            num_letters=_NUM_LETTERS,
            node_features=_HIDDEN_DIM,
            edge_features=_HIDDEN_DIM,
            hidden_dim=_HIDDEN_DIM,
            num_encoder_layers=_NUM_LAYERS,
            num_decoder_layers=_NUM_LAYERS,
            vocab=_NUM_LETTERS,
            k_neighbors=_K_NEIGHBORS,
            augment_eps=self.backbone_noise,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)
        model.eval()
        model.requires_grad_(False)

        self._model = model
        return model
