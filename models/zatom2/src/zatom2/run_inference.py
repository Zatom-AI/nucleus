import json
import logging
import os
import pickle
import time
from collections.abc import Mapping
from dataclasses import dataclass
from os import PathLike
from pathlib import Path

import hydra
import torch
from atomworks.io.utils.io_utils import to_cif_file
from biotite.structure import AtomArray, AtomArrayStack
from dotenv import load_dotenv
from foundry.utils.torch import assert_no_nans
from lightning_utilities import apply_to_collection
from nucleus.inference_engines.rfd3_compat import load_rfd3_engine
from nucleus.inference_input_parsing import (
    _coerce_zatom_design_specification,
    install_molecule_contig_support,
    uniquify_atom_names_for_cif,
)
from nucleus.utils.atom_array import _atom_array_to_smiles
from nucleus.utils.batching import SparseCollate
from nucleus.utils.bond_sidecar import BondSidecarASEAtomsLoader
from nucleus.utils.inference import (
    copy_live_weights_to_ema_shadow,
    make_rank_local_inference_model,
)
from nucleus.utils.materials import (
    add_fractional_cif_fields,
    material_cif_extra_categories,
)
from nucleus.utils.torch import cast_float64_tensors_to_float32
from omegaconf import DictConfig, OmegaConf
from rdkit import Chem
from rfd3.constants import SAVED_CONDITIONING_ANNOTATIONS
from rfd3.inference.input_parsing import DesignInputSpecification

_rfd3_engine = load_rfd3_engine()
RFD3InferenceConfig = _rfd3_engine.RFD3InferenceConfig
RFD3InferenceEngine = _rfd3_engine.RFD3InferenceEngine
RFD3Output = _rfd3_engine.RFD3Output
process_input = _rfd3_engine.process_input

load_dotenv(override=True)

logger = logging.getLogger(__name__)

_TRAINING_ONLY_CHECKPOINT_TRAINER_KEYS = (
    "log_shared_backbone_gradient_cosines",
    "shared_backbone_gradient_cosine_log_every_n_steps",
)

# For pip-installed package, configs should be relative to this file
# Adjust this path based on where configs are bundled in the package
_config_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "configs"
)


def _select(cfg: DictConfig, key: str, default=None):
    """Helper to select a config value and convert to primitive if it's a config itself."""
    value = OmegaConf.select(cfg, key, default=default)
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=False)
    return value


def _select_train_transform(cfg: DictConfig, key: str, default=None):
    """Select a value from active train dataset transforms, falling back to default."""
    train_cfg = OmegaConf.select(cfg, "datasets.train", default=None)
    if train_cfg is None:
        return default

    values = []
    for dataset_cfg in train_cfg.values():
        if dataset_cfg is None:
            continue
        value = OmegaConf.select(dataset_cfg, f"dataset.transform.{key}", default=None)
        if value is not None:
            values.append(value)

    if not values:
        return default
    value = values[0]
    if any(other != value for other in values[1:]):
        logger.warning(
            "Multiple train dataset transform values found for %s; using %r.",
            key,
            value,
        )
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=False)
    return value


def _build_inference_transform_from_checkpoint_cfg(cfg: DictConfig) -> DictConfig:
    """Build a validation-style transform when an overfit checkpoint has no val set."""
    return OmegaConf.create(
        {
            "_target_": _select(cfg, "datasets.pipeline_target"),
            "is_inference": True,
            "return_atom_array": True,
            "diffusion_batch_size": _select(
                cfg,
                "datasets.diffusion_batch_size_inference",
                _select(cfg, "datasets.diffusion_batch_size_train", 1),
            ),
            "randomly_augment": _select(cfg, "datasets.randomly_augment", False),
            "enable_atomistic_property_targets": _select(
                cfg,
                "datasets.enable_atomistic_property_targets",
                False,
            ),
            "remove_hydrogens": _select_train_transform(
                cfg,
                "remove_hydrogens",
                _select(
                    cfg,
                    "datasets.global_transform_args.remove_hydrogens",
                    True,
                ),
            ),
            "sigma_data": _select(cfg, "model.net.diffusion_module.sigma_data"),
            "central_atom": _select(cfg, "datasets.global_transform_args.central_atom"),
            "n_atoms_per_token": _select(
                cfg, "datasets.global_transform_args.n_atoms_per_token"
            ),
            "association_scheme": _select(
                cfg, "datasets.global_transform_args.association_scheme"
            ),
            "center_option": _select(
                cfg, "datasets.global_transform_args.center_option", "all"
            ),
            "generate_conformers": _select_train_transform(
                cfg,
                "generate_conformers",
                _select(
                    cfg, "datasets.global_transform_args.generate_conformers", True
                ),
            ),
            "generate_conformers_for_non_protein_only": _select_train_transform(
                cfg,
                "generate_conformers_for_non_protein_only",
                _select(
                    cfg,
                    "datasets.global_transform_args.generate_conformers_for_non_protein_only",
                    True,
                ),
            ),
            "provide_reference_conformer_when_unmasked": _select_train_transform(
                cfg,
                "provide_reference_conformer_when_unmasked",
                _select(
                    cfg,
                    "datasets.global_transform_args.provide_reference_conformer_when_unmasked",
                    True,
                ),
            ),
            "ground_truth_conformer_policy": _select(
                cfg,
                "datasets.global_transform_args.ground_truth_conformer_policy",
                "IGNORE",
            ),
            "provide_elements_for_unindexed_components": _select(
                cfg,
                "datasets.global_transform_args.provide_elements_for_unindexed_components",
                False,
            ),
            "use_element_for_atom_names_of_atomized_tokens": _select(
                cfg,
                "datasets.global_transform_args.use_element_for_atom_names_of_atomized_tokens",
                True,
            ),
            "residue_cache_dir": _select(cfg, "paths.data.residue_cache_dir"),
            "atom_1d_features": _select(
                cfg, "model.net.token_initializer.atom_1d_features"
            ),
            "token_1d_features": _select(
                cfg, "model.net.token_initializer.token_1d_features"
            ),
        }
    )


def _rewrite_inputless_molecule_contig(spec):
    """Route inputless molecule/material contigs through RFD3's length path."""
    if not isinstance(spec, dict):
        return spec
    if spec.get("input") or spec.get("atom_array_input"):
        return spec
    contig = spec.get("contig")
    if isinstance(contig, str) and any(
        part.endswith(("m", "M")) for part in contig.split(",")
    ):
        spec = dict(spec)
        spec["length"] = spec.pop("contig")
    return spec


def _valid_rdkit_mol_from_atom_array(
    atom_array: AtomArray,
    *,
    example_id: str,
) -> Chem.Mol | None:
    try:
        mol = _atom_array_to_smiles(
            atom_array,
            example_id=example_id,
            return_mol_only=True,
        )
    except Exception:
        return None

    if mol is None:
        return None

    try:
        smiles = Chem.MolToSmiles(mol)
    except Exception:
        return None

    if not smiles:
        return None

    try:
        round_tripped_mol = Chem.MolFromSmiles(smiles)
    except Exception:
        return None

    if round_tripped_mol is None or round_tripped_mol.GetNumAtoms() == 0:
        return None

    return mol


def _write_rdkit_mol_pickle(output_path: Path, mols: list[Chem.Mol]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_output_path = output_path.with_name(f".{output_path.name}.tmp")
    try:
        with tmp_output_path.open("wb") as handle:
            pickle.dump(mols, handle, protocol=pickle.HIGHEST_PROTOCOL)
        tmp_output_path.replace(output_path)
    except Exception:
        if tmp_output_path.exists():
            tmp_output_path.unlink()
        raise


def _molecule_pickle_path(out_dir: Path) -> Path:
    output_dir_name = out_dir.name or "output"
    return out_dir / f"{output_dir_name}_molecules.pkl"


@dataclass
class Zatom2Output(RFD3Output):
    """Inference output that writes material cell metadata to CIF files."""

    save_molecules: bool = False
    save_cifs: bool = True

    @staticmethod
    def _material_extra_fields(structure: AtomArray | AtomArrayStack):
        if isinstance(structure, AtomArrayStack):
            categories = structure.get_annotation_categories()
        else:
            categories = structure.get_annotation_categories()
        extra_fields = list(SAVED_CONDITIONING_ANNOTATIONS)
        for field in ("fract_x", "fract_y", "fract_z"):
            if field in categories and field not in extra_fields:
                extra_fields.append(field)
        return extra_fields

    @staticmethod
    def _prepare_structure(structure: AtomArray | AtomArrayStack):
        structure = structure.copy()
        if isinstance(structure, AtomArray):
            structure = add_fractional_cif_fields(structure)
        return structure

    def dump(self, out_dir, verbose=True):
        base_path = Path(out_dir, self.example_id).absolute()
        if self.save_cifs:
            atom_array = self._prepare_structure(self.atom_array)
            to_cif_file(
                atom_array,
                base_path,
                file_type="cif.gz",
                include_entity_poly=False,
                extra_fields=self._material_extra_fields(atom_array),
                extra_categories=material_cif_extra_categories(atom_array),
            )
        if self.metadata:
            with open(f"{base_path}.json", "w") as f:
                json.dump(self.metadata, f, indent=4)

        if self.save_cifs:
            prefix = str(base_path)[:-1].rstrip("_model_")
            suffix = str(base_path)[-1]
            if self.denoised_trajectory_stack is not None:
                to_cif_file(
                    self.denoised_trajectory_stack,
                    "_denoised_model_".join([prefix, suffix]),
                    file_type="cif.gz",
                    include_entity_poly=False,
                    extra_categories=material_cif_extra_categories(atom_array),
                )

            if self.noisy_trajectory_stack is not None:
                to_cif_file(
                    self.noisy_trajectory_stack,
                    "_noisy_model_".join([prefix, suffix]),
                    file_type="cif.gz",
                    include_entity_poly=False,
                    extra_categories=material_cif_extra_categories(atom_array),
                )

        if verbose:
            logger.info("Outputs for %s written to %s.", self.example_id, base_path)


class Float64ToFloat32Pipeline:
    """Pipeline wrapper to cast float64 tensors to float32."""

    def __init__(self, pipeline):
        self.pipeline = pipeline

    def __call__(self, data):
        return cast_float64_tensors_to_float32(self.pipeline(data))


def _unwrap_single_example_id(value, *, field_name: str):
    """Unwrap singleton collate containers while rejecting batched IDs."""
    while isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError(
                f"{field_name} must identify one source example before collation; "
                f"got {len(value)} values."
            )
        value = value[0]
    return value


class BondSidecarPipeline:
    """Attach validation bond graphs before running an inference transform."""

    def __init__(self, pipeline, *, bond_sidecar_path: str | Path):
        self.pipeline = pipeline
        self.bond_sidecar_loader = BondSidecarASEAtomsLoader(
            bond_sidecar_path=bond_sidecar_path
        )

    def __call__(self, data):
        if not isinstance(data, Mapping):
            raise TypeError(
                "Bond-sidecar inference expects a mapping returned by "
                "DesignInputSpecification.to_pipeline_input()."
            )
        if "atom_array" not in data:
            raise KeyError("Bond-sidecar inference input is missing `atom_array`.")

        specification = data.get("specification")
        source_example_id = (
            specification.get("example_id")
            if isinstance(specification, Mapping)
            else None
        )
        source_example_id = _unwrap_single_example_id(
            source_example_id,
            field_name="specification.example_id",
        )
        if source_example_id in (None, ""):
            source_example_id = _unwrap_single_example_id(
                data.get("example_id"),
                field_name="example_id",
            )
        if source_example_id in (None, ""):
            raise ValueError(
                "Bond-sidecar inference requires `specification.example_id` or "
                "an outer input example ID to select the LMDB record."
            )

        data["has_disconnected_fragments"] = self.bond_sidecar_loader.attach_bonds(
            data["atom_array"],
            example_id=str(source_example_id),
        )
        return self.pipeline(data)


class Zatom2InferenceEngine(RFD3InferenceEngine):
    """Custom inference engine to handle the case where an overfit checkpoint has no val dataset or pipeline config."""

    def __init__(
        self,
        transform_overrides=None,
        bond_sidecar_path: str | PathLike | None = None,
        material_coordinate_output_source=None,
        wrap_material_euclidean_coordinates=None,
        use_ema_weights=True,
        save_molecules=False,
        save_cifs=True,
        inference_batch_size: int = 1,
        inference_num_workers: int = 0,
        inference_prefetch_factor: int = 2,
        property_statistics_path: str | PathLike | None = None,
        **kwargs,
    ):
        for name, value, minimum in (
            ("inference_batch_size", inference_batch_size, 1),
            ("inference_num_workers", inference_num_workers, 0),
            ("inference_prefetch_factor", inference_prefetch_factor, 1),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}.")
        self.inference_batch_size = inference_batch_size
        self.inference_num_workers = inference_num_workers
        self.inference_prefetch_factor = inference_prefetch_factor
        self._inference_low_memory_mode = kwargs.get("low_memory_mode", False)
        install_molecule_contig_support()
        self.bond_sidecar_path = bond_sidecar_path
        self.property_statistics_path = property_statistics_path
        super().__init__(**kwargs)
        self.use_ema_weights = use_ema_weights
        self.save_molecules = save_molecules
        self.save_cifs = save_cifs
        self._reset_molecule_collection()
        if not self.save_cifs and self.skip_existing:
            logger.warning(
                "save_cifs=False leaves no CIF files for skip_existing to detect; "
                "reruns may not skip JSON-only outputs."
            )
        if material_coordinate_output_source is not None:
            self._assign_override(
                "trainer.material_coordinate_output_source",
                material_coordinate_output_source,
            )
        if wrap_material_euclidean_coordinates is not None:
            self._assign_override(
                "trainer.wrap_material_euclidean_coordinates",
                wrap_material_euclidean_coordinates,
            )
        if transform_overrides:
            self.transform_overrides = OmegaConf.to_container(
                OmegaConf.merge(self.transform_overrides, transform_overrides),
                resolve=True,
            )

    def _construct_trainer(self, cfg, checkpoint=None):
        """Override to copy live model weights into EMA shadow for inference if requested."""
        super()._construct_trainer(cfg, checkpoint=checkpoint)
        if self.use_ema_weights:
            return
        if copy_live_weights_to_ema_shadow(self.trainer.state["model"]):
            logger.warning(
                "use_ema_weights=false: copied live model weights into EMA shadow "
                "for this inference run."
            )
        else:
            logger.warning(
                "use_ema_weights=false was requested, but no EMA wrapper was found; "
                "using the loaded model as-is."
            )

    def _override_checkpoint_config(self, cfg):
        """Apply local artifact paths and normalize training-only checkpoint options."""
        cfg = super()._override_checkpoint_config(cfg)
        property_head = OmegaConf.select(
            cfg, "model.net.diffusion_module.atomistic_property_head"
        )
        if property_head is not None and property_head.get("enabled", bool(property_head)):
            statistics_path = self.property_statistics_path
            saved_path = property_head.get("statistics_path")
            if statistics_path is None and saved_path and not Path(saved_path).is_file():
                # Published checkpoints may retain the training machine's path.
                # Only relocate the versioned artifact shipped with this release.
                if Path(saved_path).name == "atomistic_property_statistics.v1.json":
                    import nucleus

                    package_root = Path(nucleus.__file__).parent
                    candidates = [
                        package_root / "data" / Path(saved_path).name,
                        package_root.parents[1] / "data" / Path(saved_path).name,
                    ]
                    statistics_path = next((p for p in candidates if p.is_file()), None)
            if statistics_path is not None:
                OmegaConf.update(
                    property_head,
                    "statistics_path",
                    str(statistics_path),
                    force_add=True,
                )
        trainer_cfg = OmegaConf.select(cfg, "trainer", default=None)
        if trainer_cfg is None:
            return cfg

        # The base inference override recursively merges ``metrics={}``, which
        # leaves the checkpoint's nested validation metrics intact. Neither
        # metrics nor losses are used by design inference.
        trainer_cfg.metrics = {}
        trainer_cfg.loss = None

        for key in _TRAINING_ONLY_CHECKPOINT_TRAINER_KEYS:
            if key in trainer_cfg:
                del trainer_cfg[key]
                logger.info(
                    "Ignoring training-only checkpoint trainer option `%s` during "
                    "inference.",
                    key,
                )
        if trainer_cfg.get("validate_loader_every_n_epochs"):
            trainer_cfg.validate_loader_every_n_epochs = {}
            logger.info(
                "Ignoring the checkpoint's per-loader validation schedule during "
                "inference."
            )
        return cfg

    def _reset_molecule_collection(self) -> None:
        self._molecule_candidates_seen = 0
        self._valid_molecules: list[Chem.Mol] = []

    def _collect_valid_molecules(self, outputs: list[Zatom2Output]) -> None:
        if not self.save_molecules:
            return

        self._molecule_candidates_seen += len(outputs)
        for output in outputs:
            mol = _valid_rdkit_mol_from_atom_array(
                output.atom_array,
                example_id=output.example_id,
            )
            if mol is not None:
                self._valid_molecules.append(mol)

    def _molecule_collection_rank_and_world_size(self) -> tuple[int, int]:
        fabric = getattr(getattr(self, "trainer", None), "fabric", None)
        if fabric is None:
            return 0, 1
        return int(getattr(fabric, "global_rank", 0)), int(
            getattr(fabric, "world_size", 1)
        )

    def _gather_molecule_collection_to_rank_zero(
        self,
    ) -> tuple[bool, int, list[Chem.Mol]]:
        rank, world_size = self._molecule_collection_rank_and_world_size()
        local_payload = (self._molecule_candidates_seen, self._valid_molecules)
        if world_size <= 1:
            return True, local_payload[0], list(local_payload[1])

        if (
            not torch.distributed.is_available()
            or not torch.distributed.is_initialized()
        ):
            raise RuntimeError(
                "save_molecules=True with distributed inference requires an "
                "initialized torch.distributed process group to aggregate molecules."
            )

        gathered_payloads = [None] * world_size if rank == 0 else None
        torch.distributed.gather_object(local_payload, gathered_payloads, dst=0)
        if rank != 0:
            return False, 0, []

        total_candidates_seen = 0
        valid_molecules: list[Chem.Mol] = []
        for candidates_seen, mols in gathered_payloads:
            total_candidates_seen += candidates_seen
            valid_molecules.extend(mols)
        return True, total_candidates_seen, valid_molecules

    def _dump_valid_molecule_pickle(self) -> None:
        if not self.save_molecules:
            return

        if self.out_dir is None:
            logger.warning(
                "save_molecules=True but out_dir is not set; aggregate molecule "
                "pickle was not written."
            )
            return

        should_write, candidates_seen, valid_molecules = (
            self._gather_molecule_collection_to_rank_zero()
        )
        if not should_write:
            return

        if candidates_seen == 0:
            logger.info(
                "No molecule outputs were generated; aggregate molecule pickle "
                "was not written."
            )
            return

        output_path = _molecule_pickle_path(Path(self.out_dir))
        _write_rdkit_mol_pickle(output_path, valid_molecules)
        logger.info(
            "Wrote %d valid RDKit molecules to %s (%d invalid molecules skipped).",
            len(valid_molecules),
            output_path,
            candidates_seen - len(valid_molecules),
        )

    def run(
        self,
        *,
        inputs: str | PathLike | AtomArray | DesignInputSpecification,
        n_batches: int | None = None,
        out_dir: str | PathLike | None = None,
    ):
        self._reset_molecule_collection()
        outputs = super().run(inputs=inputs, n_batches=n_batches, out_dir=out_dir)
        self._dump_valid_molecule_pickle()
        return outputs

    def _canonicalize_inputs(self, inputs):
        """Override to add support for JSON file or list of JSON files as input, which is common for structure generation tasks."""
        is_json_like = (isinstance(inputs, (str, PathLike, Path))) or (
            isinstance(inputs, list)
            and all([isinstance(i, (str, PathLike, Path)) for i in inputs])
        )
        if not is_json_like:
            return super()._canonicalize_inputs(inputs)

        inputs = process_input(
            inputs,
            json_keys_subset=self.json_keys_subset,
            global_prefix=self.global_prefix,
            specification_overrides=self.specification_overrides,
            validate=False,
        )
        inputs = {
            key: _coerce_zatom_design_specification(
                _rewrite_inputless_molecule_contig(value)
            )
            for key, value in inputs.items()
        }
        if self.prevalidate_inputs:
            for spec in inputs.values():
                if not isinstance(spec, DesignInputSpecification):
                    DesignInputSpecification.safe_init(**spec)
        return inputs

    def _construct_pipeline(self, cfg):
        """Override to construct the pipeline, casting float64 tensors to float32 if necessary."""
        force_tasks_key = "mean_force_norm_conditioning_tasks"
        global_transform_args = OmegaConf.select(
            cfg, "datasets.global_transform_args", default={}
        )
        if (
            force_tasks_key not in self.transform_overrides
            and global_transform_args is not None
            and force_tasks_key in global_transform_args
        ):
            # A validation loader may restrict conditioning to its own task;
            # inference should default to the checkpoint's training scope.
            self.transform_overrides = {
                **self.transform_overrides,
                force_tasks_key: _select(
                    cfg, f"datasets.global_transform_args.{force_tasks_key}"
                ),
            }

        if OmegaConf.select(cfg, "datasets.val", default=None) is not None:
            # Remove val datasets with null config to avoid issues with pipeline construction
            cfg.datasets.val = {
                k: v for k, v in cfg.datasets.val.items() if v is not None
            }
            super()._construct_pipeline(cfg)
            self.pipeline = self._wrap_pipeline(self.pipeline)
            return

        logger.warning(
            "Checkpoint train_cfg.datasets.val is null; constructing an inference "
            "transform from the checkpoint model/dataset settings."
        )
        transform = _build_inference_transform_from_checkpoint_cfg(cfg)
        transform = OmegaConf.merge(transform, self.transform_overrides)
        self.pipeline = self._wrap_pipeline(hydra.utils.instantiate(transform))

    def _wrap_pipeline(self, pipeline):
        """Apply raw-input sidecars before validation transforms and dtype cleanup."""
        if self.bond_sidecar_path is not None:
            pipeline = BondSidecarPipeline(
                pipeline,
                bond_sidecar_path=self.bond_sidecar_path,
            )
        return Float64ToFloat32Pipeline(pipeline)

    def _run_multi(self, specs):
        """Pack independent inputs, retaining one output group per specification."""
        if self.inference_batch_size == 1 and self.inference_num_workers == 0:
            return super()._run_multi(specs)

        # The serial path does not depend on the optional batching helpers.
        try:
            from zatom2.inference_batching import (
                batching_fallback_reason,
                build_inference_loader,
            )
        except ModuleNotFoundError as error:
            if error.name != "zatom2.inference_batching":
                raise
            raise RuntimeError(
                "This checkout is missing zatom2.inference_batching. Use "
                "inference_batch_size=1 and inference_num_workers=0, or restore "
                "the batching module before enabling parallel inference."
            ) from error

        fabric = self.trainer.fabric
        loader = build_inference_loader(
            specs,
            pipeline=self.pipeline,
            rank=int(fabric.global_rank),
            world_size=int(fabric.world_size),
            batch_size=self.inference_batch_size,
            num_workers=self.inference_num_workers,
            prefetch_factor=self.inference_prefetch_factor,
            seed=self.seed,
        )
        # Uneven shards and single-example fallbacks must not synchronize DDP
        # forwards. Retain Fabric's precision and the checkpoint's EMA dispatch.
        original_model = self.trainer.state["model"]
        local_model = make_rank_local_inference_model(
            original_model,
            strategy=fabric.strategy,
            world_size=int(fabric.world_size),
        )
        sampler_cfg = OmegaConf.select(
            self.trainer.state.get("train_cfg", OmegaConf.create({})),
            "model.net.inference_sampler",
            default=self.inference_sampler_overrides,
        )
        serial_reason = None
        if sampler_cfg.get("allow_realignment", False):
            serial_reason = "motif realignment"
        elif sampler_cfg.get("kind", "default") != "default":
            serial_reason = "symmetry sampling"
        elif self.dump_trajectories:
            serial_reason = "trajectory export"
        elif self._inference_low_memory_mode:
            serial_reason = "low-memory tokenization"

        outputs = {}
        warned_reasons = set()
        self.trainer.state["model"] = local_model
        try:
            for examples in loader:
                started = time.monotonic()
                reason = serial_reason or batching_fallback_reason(examples)
                if reason:
                    if reason not in warned_reasons:
                        logger.warning(
                            "Using single-example inference for %s; sampler settings "
                            "are preserved.",
                            reason,
                        )
                        warned_reasons.add(reason)
                    groups = [self._model_forward(example) for example in examples]
                else:
                    groups = self._model_forward_batch(examples)
                for example, output_list in zip(examples, groups, strict=True):
                    if self.out_dir:
                        for output in output_list:
                            output.dump(out_dir=self.out_dir)
                    else:
                        outputs[example["example_id"]] = output_list
                logger.info(
                    "Rank %d: generated %d examples in %.2f seconds.",
                    fabric.global_rank,
                    len(examples),
                    time.monotonic() - started,
                )
        finally:
            self.trainer.state["model"] = original_model
        return outputs

    @torch.no_grad()
    def _model_forward_batch(self, examples):
        """Run one packed rollout and use the existing writer for each example."""
        from zatom2.inference_batching import split_prediction_outputs

        # Collate on CPU before transfer, avoiding many small per-example GPU
        # copies and CPU synchronizations while SparseCollate builds indices.
        collated = SparseCollate()(examples)
        network_input = self.trainer._assemble_network_inputs(collated)
        if self.trainer.fabric.device.type == "cuda":
            network_input = apply_to_collection(
                network_input, torch.Tensor, lambda tensor: tensor.pin_memory()
            )
        network_input = self.trainer.fabric.to_device(network_input)
        model = self.trainer.state["model"]
        assert not model.training, "Inference requires an eval-mode model."
        assert_no_nans(network_input, msg="Batched inference inputs")
        network_output = model.forward(
            input=network_input,
            coord_atom_lvl_to_be_noised=network_input["X_clean_L"],
        )
        assert_no_nans(network_output, msg="Batched inference outputs")

        groups = []
        for example, prediction in zip(
            examples, split_prediction_outputs(network_output, examples), strict=True
        ):
            # Singleton collation retains the established metadata representation
            # and each specification's optional fields, including bond sidecars.
            single = SparseCollate()([example])
            single["atom_array"] = single["atom_array"][0]
            arrays, metadata = self.trainer._build_predicted_atom_array_stack(
                prediction, single
            )
            outputs = []
            for index, atom_array in enumerate(arrays):
                if self.dump_prediction_metadata_json:
                    metadata[index]["ckpt_path"] = str(self.ckpt_path)
                    metadata[index]["seed"] = self.seed
                outputs.append(
                    Zatom2Output(
                        example_id=f"{example['example_id']}_model_{index}",
                        atom_array=atom_array,
                        metadata=metadata[index]
                        if self.dump_prediction_metadata_json
                        else {},
                        save_cifs=self.save_cifs,
                    )
                )
            self._collect_valid_molecules(outputs)
            groups.append(outputs)
        return groups

    def _model_forward(self, pipeline_output):
        """Override to dump trajectories with atom names if dump_trajectories is enabled."""
        output_list = super()._model_forward(pipeline_output)
        if self.dump_trajectories:
            # Upstream RFD3 currently assigns the noisy and denoised trajectory
            # tensors to the opposite output fields after reshaping.
            # See https://github.com/RosettaCommons/foundry/issues/335 for more details.
            for output in output_list:
                (
                    output.denoised_trajectory_stack,
                    output.noisy_trajectory_stack,
                ) = (
                    output.noisy_trajectory_stack,
                    output.denoised_trajectory_stack,
                )

            trajectory_atom_array = uniquify_atom_names_for_cif(
                pipeline_output["atom_array"]
            )
            for output in output_list:
                if output.denoised_trajectory_stack is not None:
                    output.denoised_trajectory_stack = (
                        output.denoised_trajectory_stack.copy()
                    )
                    output.denoised_trajectory_stack.atom_name = (
                        trajectory_atom_array.atom_name.copy()
                    )
                    if "atom_id" in trajectory_atom_array.get_annotation_categories():
                        output.denoised_trajectory_stack.atom_id = (
                            trajectory_atom_array.atom_name.copy()
                        )
                if output.noisy_trajectory_stack is not None:
                    output.noisy_trajectory_stack = output.noisy_trajectory_stack.copy()
                    output.noisy_trajectory_stack.atom_name = (
                        trajectory_atom_array.atom_name.copy()
                    )
                    if "atom_id" in trajectory_atom_array.get_annotation_categories():
                        output.noisy_trajectory_stack.atom_id = (
                            trajectory_atom_array.atom_name.copy()
                        )
        outputs = [
            Zatom2Output(
                example_id=output.example_id,
                atom_array=output.atom_array,
                metadata=output.metadata,
                denoised_trajectory_stack=output.denoised_trajectory_stack,
                noisy_trajectory_stack=output.noisy_trajectory_stack,
                save_cifs=self.save_cifs,
            )
            for output in output_list
        ]
        self._collect_valid_molecules(outputs)
        return outputs


@hydra.main(
    config_path=_config_path,
    config_name="inference",
    version_base="1.3",
)
def run_inference(cfg: DictConfig) -> None:
    """Execute the specified inference pipeline"""

    run_params_set = {"inputs", "n_batches", "out_dir"}
    run_params = {k: v for k, v in cfg.items() if k in run_params_set}

    engine = build_inference_engine(cfg)
    engine.run(**run_params)


def build_inference_engine(cfg: DictConfig) -> Zatom2InferenceEngine:
    """Construct the checkpoint-backed engine shared by design and MLIP eval."""
    excluded_keys = {"_target_", "inputs", "n_batches", "out_dir", "evaluation"}
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    init_cfg_dict = {
        key: value for key, value in cfg_dict.items() if key not in excluded_keys
    }
    transform_overrides = init_cfg_dict.pop("transform_overrides", {})
    transform_overrides = {
        key: value for key, value in transform_overrides.items() if value is not None
    }
    bond_sidecar_path = init_cfg_dict.pop("bond_sidecar_path", None)
    property_statistics_path = init_cfg_dict.pop("property_statistics_path", None)
    material_coordinate_output_source = init_cfg_dict.pop(
        "material_coordinate_output_source", None
    )
    wrap_material_euclidean_coordinates = init_cfg_dict.pop(
        "wrap_material_euclidean_coordinates", None
    )
    use_ema_weights = init_cfg_dict.pop("use_ema_weights", True)
    save_molecules = init_cfg_dict.pop("save_molecules", False)
    save_cifs = init_cfg_dict.pop("save_cifs", True)
    batching_kwargs = {
        key: init_cfg_dict.pop(key)
        for key in (
            "inference_batch_size",
            "inference_num_workers",
            "inference_prefetch_factor",
        )
        if key in init_cfg_dict
    }
    return Zatom2InferenceEngine(
        **RFD3InferenceConfig(**init_cfg_dict),
        transform_overrides=transform_overrides,
        bond_sidecar_path=bond_sidecar_path,
        property_statistics_path=property_statistics_path,
        material_coordinate_output_source=material_coordinate_output_source,
        wrap_material_euclidean_coordinates=wrap_material_euclidean_coordinates,
        use_ema_weights=use_ema_weights,
        save_molecules=save_molecules,
        save_cifs=save_cifs,
        **batching_kwargs,
    )


if __name__ == "__main__":
    run_inference()
