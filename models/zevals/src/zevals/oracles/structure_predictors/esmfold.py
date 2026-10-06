"""ESMFold v1 structure predictor oracle (HuggingFace `transformers` port).

Runs `facebook/esmfold_v1` in-process via `transformers.EsmForProteinFolding`
— no subprocess, conda env, or cloned repo needed at runtime, only the
`zevals[proteins]` extra (`transformers`, `accelerate`, `torch`).
Usage follows the official HF port: https://huggingface.co/facebook/esmfold_v1

Each sequence in the input FASTA is folded independently (no multi-chain
poly-glycine-linker complex folding — out of scope for the single-chain
designability workflow this oracle is used for).

The model is loaded once (lazily, on first `predict()` call) and cached on
`self._model` — construct this class once (e.g. via `ZevalSuite`, which does
this for you) and reuse it across an entire training run rather than
re-instantiating it every validation epoch, or you'll pay the checkpoint
load cost repeatedly and lose the stable GPU device assignment from
`utils.devices.resolve_device`.

Hydra config: configs/oracle/structure_predictor/esmfold.yaml
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from zevals.oracles.base import BaseStructurePredictor, StructurePrediction
from zevals.utils.devices import resolve_device

_DEFAULT_MODEL_NAME = "facebook/esmfold_v1"


class ESMFoldPredictor(BaseStructurePredictor):
    """In-process ESMFold v1 structure predictor via HuggingFace `transformers`.

    Hydra config: oracle/structure_predictor/esmfold.yaml
      _target_:       zevals.oracles.structure_predictors.esmfold.ESMFoldPredictor
      model_name:     facebook/esmfold_v1
      num_recycles:   4       # null = model config default (max_recycles)
      chunk_size:     null    # int (e.g. 64) to cut VRAM on long sequences / <=16GB GPUs
      half_precision: null    # null = auto (fp16 ESM stem on CUDA, fp32 on CPU)
      batch_size:     1       # keep at 1 for faithful per-sequence pTM (see predict())
      device:         null    # null = cuda if available, else cpu
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL_NAME,
        chunk_size: int | None = None,
        half_precision: bool | None = None,
        batch_size: int = 1,
        device: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.half_precision = half_precision
        self.batch_size = batch_size
        self.device = torch.device(resolve_device(device))
        self._tokenizer = None
        self._model = None

    def predict(self, fasta_path: Path, output_dir: Path) -> list[StructurePrediction]:
        """Fold every sequence in `fasta_path` with ESMFold.

        Args:
            fasta_path: FASTA file with one or more sequences.
            output_dir: Directory to write one PDB file per sequence to.

        Returns:
            One StructurePrediction per input sequence. `num_samples` is
            ignored — ESMFold is deterministic given fixed weights and
            `num_recycles`, so repeated samples would be identical.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        tokenizer, model = self._get_model()
        records = _read_fasta(fasta_path)

        predictions: list[StructurePrediction] = []
        for start in range(0, len(records), self.batch_size):
            batch = records[start : start + self.batch_size]
            seq_ids = [seq_id for seq_id, _ in batch]
            seqs = [seq for _, seq in batch]

            tokenized = tokenizer(
                seqs, return_tensors="pt", padding=True, add_special_tokens=False
            )
            input_ids = tokenized["input_ids"].to(self.device)
            attention_mask = tokenized["attention_mask"].to(self.device)

            with torch.no_grad():
                output = model(
                    input_ids,
                    attention_mask=attention_mask,
                    num_recycles=self.num_recycles,
                )

            pdb_strings = model.output_to_pdb(output)
            plddt = output["plddt"].to("cpu")
            atom_mask = output["atom37_atom_exists"].to("cpu").bool()
            ptm = output["ptm"].to("cpu")
            # pLDDT is 0-1 in some ESMFold checkpoints/versions and 0-100 in
            # others (same ambiguity noted in the official HF example) —
            # detect and normalise to the [0, 100] scale this codebase uses.
            plddt_scale = 100.0 if plddt.max() <= 1.0 else 1.0
            # ESMFold's pTM head (`compute_tm`) reduces over the whole batch
            # to a single scalar rather than one value per sequence — a
            # long-standing upstream quirk, not batch-size-dependent code
            # here. It's only a faithful per-sequence pTM when batch_size=1;
            # for larger batches every sample in the batch gets this same
            # shared scalar.
            batch_ptm = float(ptm)

            for i, (seq_id, seq) in enumerate(zip(seq_ids, seqs)):
                pdb_path = output_dir / f"{seq_id}.pdb"
                pdb_path.write_text(pdb_strings[i])

                mask_i = atom_mask[i]
                mean_plddt = (
                    (plddt[i][mask_i].mean().item() * plddt_scale)
                    if mask_i.any()
                    else float("nan")
                )

                predictions.append(
                    StructurePrediction(
                        sequence=seq,
                        pdb_path=pdb_path,
                        plddt=mean_plddt,
                        ptm=batch_ptm,
                    )
                )

        return predictions

    def _get_model(self) -> tuple[Any, Any]:
        """Lazily load (and cache) the tokenizer + model."""
        if self._model is not None:
            return self._tokenizer, self._model

        from transformers import AutoTokenizer, EsmForProteinFolding

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = EsmForProteinFolding.from_pretrained(
            self.model_name, low_cpu_mem_usage=True
        )
        model = model.to(self.device)
        model.eval()
        model.requires_grad_(False)

        half = (
            self.half_precision
            if self.half_precision is not None
            else self.device.type == "cuda"
        )
        if half:
            model.esm = model.esm.half()

        if self.chunk_size is not None:
            model.trunk.set_chunk_size(self.chunk_size)

        self._tokenizer, self._model = tokenizer, model
        return tokenizer, model


def _read_fasta(path: Path) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    seq_id, seq_parts = None, []
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            if seq_id:
                records.append((seq_id, "".join(seq_parts)))
            seq_id = line[1:].split()[0]
            seq_parts = []
        else:
            seq_parts.append(line.strip())
    if seq_id:
        records.append((seq_id, "".join(seq_parts)))
    return records
