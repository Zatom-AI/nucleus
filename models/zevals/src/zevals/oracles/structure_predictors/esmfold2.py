"""ESMFold2 structure predictor oracle (Biohub `esm` + HuggingFace `transformers`).

Runs `biohub/ESMFold2` (or the inference-optimised `biohub/ESMFold2-Fast`)
in-process via `transformers.models.esmfold2.ESMFold2Model` plus the input
builders from the `esm` package — no subprocess, conda env, or HPC binary
needed at runtime, only the `zevals[proteins-esmfold2]` extra.
Usage follows the official model card: https://huggingface.co/biohub/ESMFold2

``predict`` folds FASTA records independently for benchmark throughput.
``predict_complex`` accepts ESMFold2's native ``StructurePredictionInput`` so
multi-chain polymers, ligands, modifications, MSAs, pockets, distograms, and
covalent bonds remain available to callers.

Unlike ESMFold v1 (`esmfold.py`), ESMFold2 is a **diffusion** model: it is
stochastic, so `num_samples` (with a per-sample `seed` offset) produces
genuinely different structures rather than identical repeats, and the folding
knobs are `num_loops` / `num_sampling_steps` rather than v1's `num_recycles`.

The model is loaded once (lazily, on first `predict()` call) and cached on
`self._model` — construct this class once (e.g. via `ZevalSuite`, which does
this for you) and reuse it across an entire training run rather than
re-instantiating it every validation epoch, or you'll pay the (large — this
is a 6B-parameter ESMC stem) checkpoint load cost repeatedly and lose the
stable GPU device assignment from `utils.devices.resolve_device`.

NOTE: the `esm` package (`pip install esm@git+https://github.com/Biohub/esm.git@main`,
no PyPI release yet) post-dates this code's authoring; the builder's return
attributes used below (`result.complex.to_mmcif()`, `result.plddt`,
`result.ptm`) are taken verbatim from the published model card. If a pinned
`esm` release renames them, this wrapper is the only place to update.

Hydra config: configs/oracle/structure_predictor/esmfold2.yaml
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch

from zevals.oracles.base import (
    BaseStructurePredictor,
    ComplexPrediction,
    StructurePrediction,
)
from zevals.utils.devices import resolve_device

# ESMFold2-Fast is single-sequence (no MSA) and inference-optimised — the right
# default for the high-throughput, single-chain designability loop. Switch to
# "biohub/ESMFold2" via config for the larger, optionally MSA-conditioned model.
_DEFAULT_MODEL_NAME = "biohub/ESMFold2-Fast"
_DEFAULT_MODEL_REVISION = "c6c7958d63f5f2f1f0fed0bb9462316f8ccceea6"
_ALPHABETS = {
    "protein": frozenset("ACDEFGHIKLMNPQRSTVWY"),
    "rna": frozenset("ACGU"),
    "dna": frozenset("ACGT"),
}


def _compat_esmfold2_config(
    config_cls: Any,
    model_name: str,
    revision: str,
    esmc_model_name: str | None = None,
) -> Any:
    """Load release configs exported by newer Biohub Transformers versions.

    ESMFold2 release snapshots serialize nested ``PretrainedConfig`` metadata
    and renamed dimension fields. Older Biohub Transformers builds expose
    strict dataclasses for those nested configs, so pass them a normalized
    representation while leaving the checkpoint weights untouched.
    """
    config_dict, _ = config_cls.get_config_dict(model_name, revision=revision)
    if esmc_model_name:
        # Keep the ESMC stem compatible with the Transformers runtime and the
        # ESMFold2 wrapper snapshot. Biohub's current main snapshot changed
        # the checkpoint format after the pinned ESMFold2 runtime was built.
        config_dict["esmc_id"] = esmc_model_name

    structure_head = dict(config_dict.get("structure_head") or {})
    diffusion_module = dict(structure_head.get("diffusion_module") or {})
    atom_encoder = dict(diffusion_module.get("atom_encoder") or {})
    diffusion_module = {
        "sigma_data": diffusion_module.get("sigma_data", 16.0),
        "c_atom": diffusion_module.get("c_atom", atom_encoder.get("hidden_size", 128)),
        "c_token": diffusion_module.get(
            "c_token", diffusion_module.get("token_hidden_size", 768)
        ),
        "c_z": diffusion_module.get("c_z", 256),
        "c_s_inputs": diffusion_module.get("c_s_inputs", 451),
        "fourier_dim": diffusion_module.get("fourier_dim", 256),
        "relpos_r_max": diffusion_module.get("relpos_r_max", 32),
        "relpos_s_max": diffusion_module.get("relpos_s_max", 2),
        "atom_num_blocks": diffusion_module.get(
            "atom_num_blocks", atom_encoder.get("num_hidden_layers", 3)
        ),
        "atom_num_heads": diffusion_module.get(
            "atom_num_heads", atom_encoder.get("num_attention_heads", 4)
        ),
        "token_num_blocks": diffusion_module.get("token_num_blocks", 12),
        "token_num_heads": diffusion_module.get("token_num_heads", 16),
        "transition_multiplier": diffusion_module.get("transition_multiplier", 2),
    }
    config_dict["structure_head"] = {
        "diffusion_module": diffusion_module,
        "distogram_bins": structure_head.get(
            "distogram_bins", structure_head.get("num_distogram_bins", 128)
        ),
        "train_noise_log_mean": structure_head.get("train_noise_log_mean", -1.2),
        "train_noise_log_std": structure_head.get("train_noise_log_std", 1.5),
        "gamma_0": structure_head.get("gamma_0", 0.605),
        "gamma_min": structure_head.get("gamma_min", 1.107),
        "noise_scale": structure_head.get("noise_scale", 0.0),
        "step_scale": structure_head.get("step_scale", 1.0),
        "inference_s_max": structure_head.get(
            "inference_s_max", structure_head.get("inference_sigma_max_ratio", 160.0)
        ),
        "inference_s_min": structure_head.get(
            "inference_s_min", structure_head.get("inference_sigma_min_ratio", 4e-4)
        ),
        "inference_p": structure_head.get(
            "inference_p", structure_head.get("inference_exponent", 8.0)
        ),
        "inference_num_steps": structure_head.get("inference_num_steps", 68),
    }

    confidence_head = dict(config_dict.get("confidence_head") or {})
    config_dict["confidence_head"] = {
        "enabled": confidence_head.get("enabled", True),
        "num_plddt_bins": confidence_head.get("num_plddt_bins", 50),
        "num_pde_bins": confidence_head.get("num_pde_bins", 64),
        "num_pae_bins": confidence_head.get("num_pae_bins", 64),
        "min_dist": confidence_head.get("min_dist", 2.0),
        "max_dist": confidence_head.get("max_dist", 52.0),
        "distogram_bins": confidence_head.get("distogram_bins", 128),
    }

    msa_encoder = dict(config_dict.get("msa_encoder") or {})
    config_dict["msa_encoder"] = {
        "enabled": msa_encoder.get("enabled", False),
        "d_msa": msa_encoder.get("d_msa", msa_encoder.get("hidden_size", 128)),
        "d_hidden": msa_encoder.get(
            "d_hidden", msa_encoder.get("outer_hidden_size", 32)
        ),
        "n_layers": msa_encoder.get(
            "n_layers", msa_encoder.get("num_hidden_layers", 4)
        ),
        "n_heads_msa": msa_encoder.get(
            "n_heads_msa", msa_encoder.get("num_attention_heads", 8)
        ),
        "msa_head_width": msa_encoder.get(
            "msa_head_width", msa_encoder.get("head_width", 32)
        ),
    }

    lm_encoder = dict(config_dict.get("lm_encoder") or {})
    config_dict["lm_encoder"] = {
        "enabled": lm_encoder.get("enabled", True),
        "n_layers": lm_encoder.get("n_layers", lm_encoder.get("num_hidden_layers", 4)),
        "lm_dropout": lm_encoder.get("lm_dropout", 0.25),
        "per_loop_lm_dropout": lm_encoder.get("per_loop_lm_dropout", True),
    }
    return config_cls.from_dict(config_dict)


class ESMFold2Predictor(BaseStructurePredictor):
    """In-process ESMFold2 structure predictor via Biohub `esm` + `transformers`.

    Hydra config: oracle/structure_predictor/esmfold2.yaml
      _target_:            zevals.oracles.structure_predictors.esmfold2.ESMFold2Predictor
      model_name:          biohub/ESMFold2-Fast
      num_loops:           3       # ESMFold2 trunk loops (accuracy vs. speed)
      num_sampling_steps:  50      # diffusion sampling steps
      num_samples:         1       # independent diffusion samples per sequence
      seed:                0       # base seed; sample i uses seed + i
      device:              null    # null = cuda if available, else cpu

    Precision note: the checkpoint is loaded in its native (fp32) precision and
    NOT cast — ESMFold2 keeps fp32 master weights and applies bf16 autocast to
    the heavy matmuls *internally* (see modeling_esmfold2's
    `torch.amp.autocast(..., dtype=bfloat16)` blocks). Force-casting the weights
    to bf16 (e.g. via `torch_dtype`) breaks the fold: the processor emits fp32
    input tensors, which then hit bf16 Linear weights and raise
    "mat1 and mat2 must have the same dtype". This mirrors the model card's
    `ESMFold2Model.from_pretrained(...).cuda().eval()` (no dtype cast).
    """

    def __init__(
        self,
        entity_type: str = "protein",
        model_name: str = _DEFAULT_MODEL_NAME,
        model_revision: str = _DEFAULT_MODEL_REVISION,
        num_loops: int = 3,
        num_sampling_steps: int = 50,
        num_samples: int = 1,
        seed: int = 0,
        device: str | None = None,
        esmc_model_name: str | None = None,
        **kwargs: Any,
    ) -> None:
        # num_samples flows through to BaseStructurePredictor; num_recycles is
        # irrelevant to ESMFold2 (it uses num_loops) so we don't forward it.
        super().__init__(num_samples=num_samples, **kwargs)
        if entity_type not in _ALPHABETS:
            raise ValueError(
                "ESMFold2 entity_type must be protein, rna, or dna; "
                f"received {entity_type!r}"
            )
        self.entity_type = entity_type
        self.model_name = model_name
        self.model_revision = model_revision
        self.num_loops = num_loops
        self.num_sampling_steps = num_sampling_steps
        self.seed = seed
        self.esmc_model_name = esmc_model_name or os.environ.get(
            "ESMFOLD2_STEM_MODEL_NAME"
        )
        self.use_compat_config = os.environ.get(
            "ESMFOLD2_USE_COMPAT_CONFIG", "0"
        ).lower() in {"1", "true", "yes", "on"}
        self.use_svd_retry = os.environ.get("ESMFOLD2_SVD_RETRY", "0").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self.device = torch.device(resolve_device(device))
        self._model = None
        self._builder = None
        self._input_types: dict[str, Any] | None = None

    def predict(self, fasta_path: Path, output_dir: Path) -> list[StructurePrediction]:
        """Fold every sequence in `fasta_path` with ESMFold2.

        Args:
            fasta_path: FASTA file with one or more sequences.
            output_dir: Directory to write one mmCIF file per (sequence, sample)
                        to. ESMFold2 emits mmCIF (`.cif`), not PDB; downstream
                        designability dispatches on the file suffix so this is
                        transparent.

        Returns:
            One StructurePrediction per input sequence × `num_samples`. Because
            ESMFold2 is a diffusion model, distinct samples (seed + i) are
            genuinely different structures.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        model, builder, input_types = self._get_model()
        entity_input = input_types[self.entity_type]
        structure_input = input_types["structure"]
        records = _read_fasta(fasta_path)

        predictions: list[StructurePrediction] = []
        for seq_id, seq in records:
            _validate_sequence(seq, self.entity_type, seq_id)
            for sample in range(self.num_samples):
                spi = structure_input(sequences=[entity_input(id="A", sequence=seq)])
                result = self._fold(builder, model, spi, sample)

                # Only tag the filename with the sample index when there's more
                # than one, so the common num_samples=1 path matches v1's naming.
                suffix = "" if self.num_samples == 1 else f"_sample{sample}"
                cif_path = output_dir / f"{seq_id}{suffix}.cif"
                cif_path.write_text(result.complex.to_mmcif())

                predictions.append(
                    StructurePrediction(
                        sequence=seq,
                        pdb_path=cif_path,
                        plddt=_mean_plddt(result.plddt),
                        # ESMFold2 returns ptm as a Python float (or None if the
                        # pTM head is disabled); StructurePrediction uses nan
                        # when a confidence metric is unavailable.
                        ptm=float(result.ptm)
                        if result.ptm is not None
                        else float("nan"),
                    )
                )

        return predictions

    def predict_complex(
        self, structure_input: Any, output_dir: Path, name: str = "complex"
    ) -> list[ComplexPrediction]:
        """Fold an unrestricted native ESMFold2 ``StructurePredictionInput``.

        The input object is passed to Biohub's builder unchanged. This is the
        full-capability path for complexes and conditioning; the FASTA API is a
        deliberately simpler benchmark convenience method.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        model, builder, input_types = self._get_model()
        if not isinstance(structure_input, input_types["structure"]):
            raise TypeError(
                "structure_input must be an ESMFold2 StructurePredictionInput"
            )

        predictions: list[ComplexPrediction] = []
        for sample in range(self.num_samples):
            result = self._fold(builder, model, structure_input, sample)
            suffix = "" if self.num_samples == 1 else f"_sample{sample}"
            cif_path = output_dir / f"{name}{suffix}.cif"
            cif_path.write_text(result.complex.to_mmcif())
            predictions.append(
                ComplexPrediction(
                    structure_path=cif_path,
                    plddt=_mean_plddt(result.plddt),
                    ptm=_optional_float(result.ptm),
                    iptm=_optional_float(getattr(result, "iptm", None)),
                    extra={
                        "backend": self.model_name,
                        "pair_chains_iptm": getattr(result, "pair_chains_iptm", None),
                    },
                )
            )
        return predictions

    def _fold(self, builder: Any, model: Any, structure_input: Any, sample: int) -> Any:
        if not self.use_svd_retry:
            with torch.no_grad():
                result = builder.fold(
                    model,
                    structure_input,
                    num_loops=self.num_loops,
                    num_sampling_steps=self.num_sampling_steps,
                    num_diffusion_samples=1,
                    seed=self.seed + sample,
                )
            return self._unwrap_fold_result(result)

        # ESMFold2's rigid-alignment SVD explicitly requests CUDA ``gesvd``.
        # For some low-complexity sequences, cuSOLVER reports non-convergence
        # even though the more robust Jacobi driver can solve the same matrix.
        original_svd = torch.linalg.svd

        def svd_with_retry(matrix: Any, *args: Any, **kwargs: Any) -> Any:
            try:
                return original_svd(matrix, *args, **kwargs)
            except RuntimeError as exc:
                if (
                    not getattr(matrix, "is_cuda", False)
                    or kwargs.get("driver") != "gesvd"
                    or "failed to converge" not in str(exc)
                ):
                    raise
                retry_kwargs = dict(kwargs)
                retry_kwargs["driver"] = "gesvdj"
                return original_svd(matrix, *args, **retry_kwargs)

        # The Biohub transformer calls torch.linalg.svd internally, so keep
        # this compatibility fallback scoped to one fold invocation.
        torch.linalg.svd = svd_with_retry
        try:
            with torch.no_grad():
                result = builder.fold(
                    model,
                    structure_input,
                    num_loops=self.num_loops,
                    num_sampling_steps=self.num_sampling_steps,
                    num_diffusion_samples=1,
                    seed=self.seed + sample,
                )
        finally:
            torch.linalg.svd = original_svd
        return self._unwrap_fold_result(result)

    @staticmethod
    def _unwrap_fold_result(result: Any) -> Any:
        # ``fold`` may return one result or a list of diffusion samples.
        if isinstance(result, (list, tuple)):
            if len(result) != 1:
                raise RuntimeError(
                    "ESMFold2 returned an unexpected number of results for "
                    "num_diffusion_samples=1"
                )
            result = result[0]
        return result

    def _get_model(self) -> tuple[Any, Any, dict[str, Any]]:
        """Lazily load (and cache) the model + input builder.

        Returns the model, builder, and typed ESMFold2 input classes.
        """
        if self._model is not None:
            assert self._input_types is not None
            return self._model, self._builder, self._input_types

        from esm.models.esmfold2 import (
            DNAInput,
            ESMFold2InputBuilder,
            LigandInput,
            ProteinInput,
            RNAInput,
            StructurePredictionInput,
        )
        from transformers.models.esmfold2.configuration_esmfold2 import (
            ESMFold2Config,
        )
        from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model

        # Load in native (fp32) precision — do NOT pass a dtype. ESMFold2
        # autocasts to bf16 internally; casting the weights breaks folding
        # (see the precision note in the class docstring).
        config = None
        # A pinned historical ESMC stem is used with the older validated
        # Transformers runtime. Normalize nested configs in that case too;
        # passing the raw release config exposes newer metadata fields such as
        # ``architectures`` to strict legacy nested config classes.
        if self.use_compat_config or self.esmc_model_name:
            config = _compat_esmfold2_config(
                ESMFold2Config,
                self.model_name,
                self.model_revision,
                self.esmc_model_name,
            )

        model_kwargs: dict[str, Any] = {"revision": self.model_revision}
        if config is not None:
            model_kwargs["config"] = config
        model = ESMFold2Model.from_pretrained(self.model_name, **model_kwargs)
        model = model.to(self.device)
        model.eval()
        model.requires_grad_(False)

        self._model = model
        self._builder = ESMFold2InputBuilder()
        self._input_types = {
            "protein": ProteinInput,
            "rna": RNAInput,
            "dna": DNAInput,
            "ligand": LigandInput,
            "structure": StructurePredictionInput,
        }
        return self._model, self._builder, self._input_types


def _validate_sequence(sequence: str, entity_type: str, sequence_id: str) -> None:
    invalid = sorted(set(sequence.upper()) - _ALPHABETS[entity_type])
    if invalid:
        raise ValueError(
            f"ESMFold2 {entity_type} sequence {sequence_id!r} contains "
            f"unsupported symbols: {invalid}"
        )


def _mean_plddt(plddt: Any) -> float:
    """Mean pLDDT normalised to the [0, 100] scale this codebase uses.

    ESMFold checkpoints report pLDDT on either a 0-1 or 0-100 scale depending
    on version (same ambiguity flagged in the ESMFold HF examples); detect and
    rescale so the returned value is always [0, 100]."""
    if plddt is None:
        return float("nan")
    t = torch.as_tensor(plddt).float()
    if t.numel() == 0:
        return float("nan")
    scale = 100.0 if float(t.max()) <= 1.0 else 1.0
    return float(t.mean()) * scale


def _optional_float(value: Any) -> float:
    return float(value) if value is not None else float("nan")


def _read_fasta(path: Path) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    seq_id, seq_parts = None, []
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            if seq_id:
                records.append((seq_id, "".join(seq_parts).upper()))
            seq_id = line[1:].split()[0]
            seq_parts = []
        else:
            seq_parts.append(line.strip())
    if seq_id:
        records.append((seq_id, "".join(seq_parts).upper()))
    return records
