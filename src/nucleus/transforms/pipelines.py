"""Nucleus pipeline builders that adapt RFD3 pipelines without vendoring them."""

from functools import partial

from atomworks.ml.transforms.base import ConditionalRoute, Identity
from atomworks.ml.transforms.bonds import AddAF3TokenBondFeatures

from nucleus.transforms.atomistic_property_transforms import (
    AddAtomisticPropertyTargets,
    AddElectronicStateConditioning,
    FinalizeAtomisticPropertyTargets,
)
from nucleus.transforms.atomized_ligand_transforms import (
    AddAtomizedLigandAdditional1dFeatures,
    AddAtomizedLigandIsXFeatures,
    EncodeAtomizedLigandTokenFeatures,
    FastAtomizedLigandUnindex,
    FeaturizeUnconditionedAtomizedLigandAtoms,
    InitializeAtomizedLigand,
)
from nucleus.transforms.design_transforms import (
    CanonicalizeMLIPInferenceCoordinates,
    MotifCentering,
    SubsampleToPolymerOrLigandTypes,
)
from nucleus.transforms.diffusion import SampleFlowMatchingNoise
from nucleus.transforms.hbplus import (
    DisabledCalculateHbondsPlus,
    SafeCalculateHbondsPlus,
)
from nucleus.transforms.material_transforms import (
    AddMaterialCrystalFeatures,
    AddMaterialPredictionTargets,
    MaterialAwareMotifCenterRandomAugmentation,
    RandomPeriodicCellOriginShift,
)
from nucleus.transforms.molecule_transforms import (
    AddMeanForceNormConditioning,
    AddUnifiedTokenTargets,
    EnsureMotifAtomAnnotation,
    MoleculeAwarePadTokensWithVirtualAtoms,
)
from nucleus.transforms.na_geometry import SafeCalculateNucleicAcidGeomFeats
from nucleus.transforms.polymer_mapping import (
    ContextAwareAssignTypes,
    MapPolymerNucleotideParents,
)
from nucleus.utils.forces import (
    FORCE_LOG_EPSILON,
    force_conditioning_method_from_features,
)

try:
    from rfd3na.transforms import pipelines
    from rfd3na.transforms.training_conditions import PPICondition
except ImportError:
    raise ImportError(
        "Could not import RFD3NA transforms. Please ensure that the `rc-foundry[all]` package is installed and up to date."
    )


def _is_inference(data):
    """Condition function to determine if the current data is for inference or training."""
    return data["is_inference"]


def _is_structure_prediction(data):
    """Return whether this example uses structure-prediction conditioning.

    Training examples receive an explicit condition name from
    ``SampleConditioningType``. Inference bypasses that training-only transform,
    so identify structure prediction from its canonical conditioning masks: the
    complete sequence is fixed while all coordinates are diffused.
    """
    if data.get("sampled_condition_name") in {
        "structure_prediction",
        "sequence_fixed_structure_design",
    }:
        return True
    if not data.get("is_inference", False):
        return False

    atom_array = data.get("atom_array")
    if atom_array is None:
        return False
    required_annotations = {
        "is_motif_atom_with_fixed_seq",
        "is_motif_atom_with_fixed_coord",
    }
    if not required_annotations.issubset(atom_array.get_annotation_categories()):
        return False

    fixed_sequence = atom_array.is_motif_atom_with_fixed_seq.astype(bool)
    fixed_coordinates = atom_array.is_motif_atom_with_fixed_coord.astype(bool)
    return bool(
        fixed_sequence.size > 0 and fixed_sequence.all() and not fixed_coordinates.any()
    )


class StructurePredictionAddAF3TokenBondFeatures(AddAF3TokenBondFeatures):
    """Add token bonds only when the sampled task is structure prediction.

    The class name intentionally retains ``AddAF3TokenBondFeatures`` so legacy
    AtomWorks transform-history requirements continue to recognize this stage.
    """

    def check_input(self, data):
        if _is_structure_prediction(data):
            super().check_input(data)

    def forward(self, data):
        if not _is_structure_prediction(data):
            return data
        return super().forward(data)


def _condition_is_enabled(data, *, condition):
    """Condition function to determine if a specific condition is enabled."""
    return data["conditions"][condition]


def _is_ppi_with_hotspots(data):
    """Condition function to determine if the current data is for a PPI with hotspots."""
    return (
        data["sampled_condition_name"] == "ppi"
        and data["conditions"]["add_ppi_hotspots"]
    )


def _add_global_is_non_loopy_feature(data):
    """Condition function to determine if the global is_non_loopy feature should be added."""
    return data["conditions"]["add_global_is_non_loopy_feature"]


def _use_full_binder_spatial_crop(
    data, *, keep_full_binder_in_spatial_crop, max_binder_length
):
    """Condition function to determine if the full binder should be kept in the spatial crop."""
    sampled_condition_cls = data.get("sampled_condition_cls")
    is_binder_design_condition = isinstance(sampled_condition_cls, type) and issubclass(
        sampled_condition_cls, PPICondition
    )
    return (
        keep_full_binder_in_spatial_crop
        and is_binder_design_condition
        and pipelines.get_token_count(
            data["atom_array"][data["atom_array"].is_binder_pn_unit]
        )
        < max_binder_length
        and data["conditions"]["full_binder_crop"]
    )


def _has_symmetry_id(data):
    """Condition function to determine if the current data has a symmetry ID."""
    return "symmetry_id" in data["atom_array"].get_annotation_categories()


def _training_route(transform):
    """Route that applies the given transform only during training."""
    return ConditionalRoute(
        condition_func=_is_inference,
        transform_map={True: Identity(), False: transform},
    )


def _inference_route(transform):
    """Route that applies the given transform only during inference."""
    return ConditionalRoute(
        condition_func=_is_inference,
        transform_map={False: Identity(), True: transform},
    )


def _training_condition_route(condition, transform):
    """Route that applies the given transform during training only if the specified condition is enabled."""
    return _training_route(
        ConditionalRoute(
            condition_func=partial(_condition_is_enabled, condition=condition),
            transform_map={
                True: transform,
                False: Identity(),
            },
        )
    )


def _replace_condition_func(transform):
    """Replace unpicklable lambda condition functions in the given transform with named functions."""
    condition_func = getattr(transform, "condition_func", None)
    code = getattr(condition_func, "__code__", None)
    if code is None or condition_func.__name__ != "<lambda>":
        return

    consts = set(code.co_consts)
    if "full_binder_crop" in consts:
        closure_values = {
            name: cell.cell_contents
            for name, cell in zip(code.co_freevars, condition_func.__closure__)
        }
        transform.condition_func = partial(
            _use_full_binder_spatial_crop,
            keep_full_binder_in_spatial_crop=closure_values[
                "keep_full_binder_in_spatial_crop"
            ],
            max_binder_length=closure_values["max_binder_length"],
        )
    elif "add_ppi_hotspots" in consts:
        transform.condition_func = _is_ppi_with_hotspots
    elif "add_global_is_non_loopy_feature" in consts:
        transform.condition_func = _add_global_is_non_loopy_feature
    elif "symmetry_id" in consts:
        transform.condition_func = _has_symmetry_id


def _replace_unpicklable_condition_funcs(transform):
    """Recursively replace unpicklable lambda condition functions in the given transform and its children with named functions."""
    _replace_condition_func(transform)

    for child in getattr(transform, "transforms", []):
        _replace_unpicklable_condition_funcs(child)

    for child in getattr(transform, "transform_map", {}).values():
        _replace_unpicklable_condition_funcs(child)


def _gate_token_bonds_to_structure_prediction(pipeline):
    """Only construct token-bond features for structure-prediction examples."""
    replacements = 0
    for index, transform in enumerate(pipeline.transforms):
        if transform.__class__.__name__ != "AddAF3TokenBondFeatures":
            continue
        pipeline.transforms[index] = StructurePredictionAddAF3TokenBondFeatures(
            distance_cutoff=transform.distance_cutoff
        )
        replacements += 1

    if replacements != 1:
        raise RuntimeError(
            "Expected exactly one AddAF3TokenBondFeatures in the RFD3 pipeline, "
            f"found {replacements}."
        )


def _replace_edm_noising_with_flow_matching(
    transform,
    *,
    sigma_data,
    diffusion_batch_size,
    beta_alpha,
    uniform_probability,
    min_time,
    sigma_perturb_com,
):
    """Replace the EDM noising pair while retaining the pipeline's public keys."""
    replacements = {"sample": 0, "augment": 0}

    for parent in [transform, *getattr(transform, "transforms", [])]:
        children = getattr(parent, "transforms", None)
        if not isinstance(children, list):
            continue
        for index, child in enumerate(children):
            if child.__class__.__name__ == "SampleEDMNoise":
                # Flow interpolation depends on the final augmented clean frame,
                # so defer all sampling to the later AugmentNoise position.
                children[index] = Identity()
                replacements["sample"] += 1
            elif child.__class__.__name__ == "AugmentNoise":
                children[index] = SampleFlowMatchingNoise(
                    sigma_data=sigma_data,
                    diffusion_batch_size=diffusion_batch_size,
                    beta_alpha=beta_alpha,
                    uniform_probability=uniform_probability,
                    min_time=min_time,
                    sigma_perturb_com=sigma_perturb_com,
                )
                replacements["augment"] += 1

    if replacements != {"sample": 1, "augment": 1}:
        raise RuntimeError(
            "Expected exactly one SampleEDMNoise and one AugmentNoise in the "
            f"RFD3 pipeline, found {replacements}."
        )


def _replace_random_augmentation_with_centering(pipeline):
    """Retain motif-aware centering while disabling random augmentation."""
    replacements = 0

    for index, child in enumerate(pipeline.transforms):
        if child.__class__.__name__ == "MotifCenterRandomAugmentation":
            pipeline.transforms[index] = MotifCentering(
                center_option=child.center_option
            )
            replacements += 1

    if replacements != 1:
        raise RuntimeError(
            "Expected exactly one MotifCenterRandomAugmentation in the RFD3 "
            f"pipeline, found {replacements}."
        )


def _replace_random_augmentation_with_material_aware_rotation(
    pipeline,
    *,
    record_rotation_pairs: bool = False,
):
    """Use cell-aware rotations for materials and preserve RFD3 for molecules."""
    replacements = 0

    for index, child in enumerate(pipeline.transforms):
        if child.__class__.__name__ == "MotifCenterRandomAugmentation":
            pipeline.transforms[index] = MaterialAwareMotifCenterRandomAugmentation(
                batch_size=child.batch_size,
                sigma_perturb=child.sigma_perturb,
                center_option=child.center_option,
                record_rotation_pairs=record_rotation_pairs,
            )
            replacements += 1

    if replacements != 1:
        raise RuntimeError(
            "Expected exactly one MotifCenterRandomAugmentation in the RFD3 "
            f"pipeline, found {replacements}."
        )


def _insert_random_periodic_cell_origin_shift(pipeline, *, batch_size: int) -> None:
    """Insert periodic cell-cut augmentation before centering and flow noising."""
    insertion_indices = [
        index
        for index, transform in enumerate(pipeline.transforms)
        if transform.__class__.__name__
        in {"MaterialAwareMotifCenterRandomAugmentation", "MotifCentering"}
    ]
    if len(insertion_indices) != 1:
        raise RuntimeError(
            "Expected exactly one material-aware augmentation or centering "
            f"transform, found {len(insertion_indices)}."
        )
    pipeline.transforms.insert(
        insertion_indices[0],
        RandomPeriodicCellOriginShift(batch_size=batch_size),
    )


def _insert_mlip_inference_centering(pipeline) -> None:
    """Canonicalize direct property inputs after the augmentation stage."""
    insertion_indices = [
        index + 1
        for index, transform in enumerate(pipeline.transforms)
        if transform.__class__.__name__
        in {"MaterialAwareMotifCenterRandomAugmentation", "MotifCentering"}
    ]
    if len(insertion_indices) != 1:
        raise RuntimeError(
            "Expected exactly one material-aware augmentation or centering "
            f"transform, found {len(insertion_indices)}."
        )
    augmentation = pipeline.transforms[insertion_indices[0] - 1]
    pipeline.transforms.insert(
        insertion_indices[0],
        CanonicalizeMLIPInferenceCoordinates(center_option=augmentation.center_option),
    )


def _nested_transform_names(transform):
    """Return class names for a transform and all routed children."""
    names = {transform.__class__.__name__}
    for child in getattr(transform, "transforms", []):
        names.update(_nested_transform_names(child))
    for child in getattr(transform, "transform_map", {}).values():
        names.update(_nested_transform_names(child))
    return names


def _disable_transform_bookkeeping(transform):
    """Disable validation/history already covered by the fast-path contract."""
    transform.validate_input = False
    transform._track_transform_history = False
    for child in getattr(transform, "transforms", []):
        _disable_transform_bookkeeping(child)
    for child in getattr(transform, "transform_map", {}).values():
        _disable_transform_bookkeeping(child)


def _optimize_atomized_ligand_pipeline(
    pipeline,
    *,
    generate_conformers,
    meta_conditioning_probabilities,
):
    """Specialize the unified pipeline for resolved ASE ligand records."""
    replacements = {
        "AssignTypes": lambda transform: InitializeAtomizedLigand(),
        "ContextAwareAssignTypes": lambda transform: InitializeAtomizedLigand(),
        "UnindexFlaggedTokens": lambda transform: FastAtomizedLigandUnindex(),
        "EncodeAF3TokenLevelFeatures": lambda transform: (
            EncodeAtomizedLigandTokenFeatures(
                sequence_encoding=transform.sequence_encoding,
                encode_residues_to=transform.encode_residues_to,
            )
        ),
        "AddIsXFeats": lambda transform: AddAtomizedLigandIsXFeatures(
            X=transform.X,
            central_atom=transform.central_atom,
        ),
        "AddAdditional1dFeaturesToFeats": lambda transform: (
            AddAtomizedLigandAdditional1dFeatures(
                token_1d_features=transform.token_1d_features,
                atom_1d_features=transform.atom_1d_features,
                autofill_zeros_if_not_present_in_atomarray=transform.autofill,
                association_scheme=transform.association_scheme,
            )
        ),
    }

    skip_top_level = {
        "RemoveTerminalOxygen",
        "RemoveUnresolvedPNUnits",
        "MaskPolymerResiduesWithUnresolvedFrameAtoms",
        "FlagAndReassignCovalentModifications",
        "FlagNonPolymersForAtomization",
        "AddGlobalAtomIdAnnotation",
        "AtomizeByCCDName",
        "RemoveNucleicAcidTerminalOxygen",
        "AddWithinChainInstanceResIdx",
        "AddWithinPolyResIdxAnnotation",
        "AddProteinTerminiAnnotation",
        "SubsampleToPolymerOrLigandTypes",
        "MoleculeAwarePadTokensWithVirtualAtoms",
    }
    if not generate_conformers:
        skip_top_level.add("LoadCachedResidueLevelData")
    skip_routed = {
        "StrtoBoolforIsXFeatures",
        "SetOccToZeroOnBfactor",
        "RemovePolymersWithTooFewResolvedResidues",
        "HandleUndesiredResTokens",
        "RemoveUnresolvedLigandAtomsIfTooMany",
        "RemoveTokensWithoutCorrespondingCentralAtom",
        "PlaceUnresolvedTokenAtomsOnRepresentativeAtom",
        "PlaceUnresolvedTokenOnClosestResolvedTokenInSequence",
    }
    probabilities = meta_conditioning_probabilities or {}
    if probabilities.get("calculate_rasa", 0.0) == 0.0:
        skip_routed.add("CalculateRASA")
    if probabilities.get("calculate_hbonds", 0.0) == 0.0:
        skip_routed.add("CalculateHbondsPlus")
    if probabilities.get("add_ppi_hotspots", 0.0) == 0.0:
        skip_routed.add("AddPPIHotspotFeature")
    if probabilities.get("add_1d_ss_features", 0.0) == 0.0:
        skip_routed.add("Add1DSSFeature")
    if probabilities.get("add_global_is_non_loopy_feature", 0.0) == 0.0:
        skip_routed.add("AddGlobalIsNonLoopyFeature")

    unconditioned_atom_features = (
        probabilities.get("calculate_rasa", 0.0) == 0.0
        and probabilities.get("calculate_hbonds", 0.0) == 0.0
    )
    optimized = []
    for transform in pipeline.transforms:
        transform_name = transform.__class__.__name__
        if transform_name in skip_top_level:
            continue
        if _nested_transform_names(transform) & skip_routed:
            continue
        if transform_name == "FeaturizeAtoms" and unconditioned_atom_features:
            transform = FeaturizeUnconditionedAtomizedLigandAtoms()
        elif transform_name in replacements:
            transform = replacements[transform_name](transform)
        optimized.append(transform)

    pipeline.transforms = optimized
    _disable_transform_bookkeeping(pipeline)
    return pipeline


def build_atom14_base_pipeline(
    *args,
    remove_hydrogens=True,
    mean_force_norm_conditioning_probability=0.8,
    mean_force_norm_conditioning_tasks=None,
    force_conditioning_method=None,
    force_conditioning_epsilon=FORCE_LOG_EPSILON,
    randomly_translate_fractional_coordinates=False,
    optimize_atomized_ligand_pipeline=False,
    **kwargs,
):
    """Build the RFD3 pipeline with optional flow and material augmentation."""

    noise_process = kwargs.pop("noise_process", "edm")
    flow_matching_beta_alpha = kwargs.pop("flow_matching_beta_alpha", 1.9)
    flow_matching_uniform_probability = kwargs.pop(
        "flow_matching_uniform_probability", 0.02
    )
    flow_matching_min_time = kwargs.pop("flow_matching_min_time", 1e-3)
    flow_matching_sigma_perturb_com = kwargs.get("sigma_perturb_com", 0.0)
    # Retain compatibility with configurations from the former Cartesian COM
    # augmentation. A uniform fractional-origin shift has no length scale.
    kwargs.pop("material_sigma_perturb_com", None)
    atomistic_property_domain = kwargs.pop("atomistic_property_domain", None)
    enable_atomistic_property_targets = kwargs.pop(
        "enable_atomistic_property_targets", False
    )
    record_rotation_pairs = kwargs.pop("record_rotation_pairs", False)
    if record_rotation_pairs and not enable_atomistic_property_targets:
        raise ValueError(
            "record_rotation_pairs requires enable_atomistic_property_targets=True."
        )
    enable_hbplus_features = kwargs.pop("enable_hbplus_features", True)
    token_1d_features = kwargs.get("token_1d_features")
    if force_conditioning_method is None:
        force_conditioning_method = force_conditioning_method_from_features(
            token_1d_features
        )
    if hasattr(token_1d_features, "items"):
        # Hydra uses null entries to disable the mutually exclusive legacy/new
        # feature family. Upstream feature transforms expect only integer widths.
        kwargs["token_1d_features"] = {
            key: value for key, value in token_1d_features.items() if value is not None
        }
    original_subsample_to_types = pipelines.SubsampleToTypes
    original_pad_tokens = pipelines.PadTokensWithVirtualAtoms
    original_add_ground_truth_sequence = pipelines.AddGroundTruthSequence
    original_training_route = pipelines.TrainingRoute
    original_inference_route = pipelines.InferenceRoute
    original_training_condition_route = pipelines.TrainingConditionRoute
    original_assign_types = pipelines.AssignTypes
    original_calculate_hbonds_plus = pipelines.CalculateHbondsPlus
    original_calculate_na_geom = pipelines.CalculateNucleicAcidGeomFeats

    pipelines.SubsampleToTypes = SubsampleToPolymerOrLigandTypes
    pipelines.PadTokensWithVirtualAtoms = MoleculeAwarePadTokensWithVirtualAtoms
    pipelines.AddGroundTruthSequence = AddUnifiedTokenTargets
    pipelines.TrainingRoute = _training_route
    pipelines.InferenceRoute = _inference_route
    pipelines.TrainingConditionRoute = _training_condition_route
    pipelines.AssignTypes = ContextAwareAssignTypes
    pipelines.CalculateHbondsPlus = (
        SafeCalculateHbondsPlus
        if enable_hbplus_features
        else DisabledCalculateHbondsPlus
    )
    pipelines.CalculateNucleicAcidGeomFeats = SafeCalculateNucleicAcidGeomFeats
    try:
        pipeline = pipelines.build_atom14_base_pipeline(*args, **kwargs)
        _replace_unpicklable_condition_funcs(pipeline)
        _gate_token_bonds_to_structure_prediction(pipeline)
        for i, transform in enumerate(pipeline.transforms):
            if transform.__class__.__name__ == "RemoveHydrogens":
                pipeline.transforms.insert(i + 1, MapPolymerNucleotideParents())
                break
        else:
            raise RuntimeError("Expected RemoveHydrogens in the RFD3 pipeline.")
        for i, transform in enumerate(pipeline.transforms):
            if transform.__class__.__name__ in {
                "AssignTypes",
                "ContextAwareAssignTypes",
            }:
                early_transforms = [AddMaterialCrystalFeatures()]
                if enable_atomistic_property_targets:
                    early_transforms.append(
                        AddAtomisticPropertyTargets(
                            diffusion_batch_size=kwargs["diffusion_batch_size"],
                            domain=atomistic_property_domain,
                        )
                    )
                pipeline.transforms[i + 1 : i + 1] = early_transforms
                break
        for i, transform in enumerate(pipeline.transforms):
            if transform.__class__.__name__ == "CreateDesignReferenceFeatures":
                pipeline.transforms.insert(i, EnsureMotifAtomAnnotation())
                break
        for i, transform in enumerate(pipeline.transforms):
            if transform.__class__.__name__ == "AddUnifiedTokenTargets":
                target_transforms = [
                    AddMeanForceNormConditioning(
                        conditioning_probability=(
                            mean_force_norm_conditioning_probability
                        ),
                        conditioning_tasks=mean_force_norm_conditioning_tasks,
                        conditioning_method=force_conditioning_method,
                        epsilon=force_conditioning_epsilon,
                    ),
                    AddMaterialPredictionTargets(),
                ]
                if enable_atomistic_property_targets:
                    target_transforms.append(
                        FinalizeAtomisticPropertyTargets(
                            diffusion_batch_size=kwargs["diffusion_batch_size"]
                        )
                    )
                else:
                    # Electronic state is an input for every OMol task, while
                    # energy/force targets remain opt-in.
                    target_transforms.append(
                        AddElectronicStateConditioning(
                            domain=atomistic_property_domain,
                        )
                    )
                pipeline.transforms[i + 1 : i + 1] = target_transforms
                break
        # Periodic samples rotate with their cells and use a synchronized
        # zero-centroid Cartesian/fractional gauge. Preserve the original RFD3
        # behavior for non-periodic samples.
        if kwargs.get("randomly_augment", True):
            _replace_random_augmentation_with_material_aware_rotation(
                pipeline,
                record_rotation_pairs=record_rotation_pairs,
            )
        else:
            _replace_random_augmentation_with_centering(pipeline)
        if enable_atomistic_property_targets:
            _insert_mlip_inference_centering(pipeline)
        if not remove_hydrogens:
            pipeline.transforms = [
                transform
                for transform in pipeline.transforms
                if transform.__class__.__name__ != "RemoveHydrogens"
            ]
        if noise_process == "flow_matching":
            _replace_edm_noising_with_flow_matching(
                pipeline,
                sigma_data=kwargs["sigma_data"],
                diffusion_batch_size=kwargs["diffusion_batch_size"],
                beta_alpha=flow_matching_beta_alpha,
                uniform_probability=flow_matching_uniform_probability,
                min_time=flow_matching_min_time,
                sigma_perturb_com=flow_matching_sigma_perturb_com,
            )
        if randomly_translate_fractional_coordinates:
            _insert_random_periodic_cell_origin_shift(
                pipeline,
                batch_size=kwargs["diffusion_batch_size"],
            )
        if optimize_atomized_ligand_pipeline:
            if kwargs.get("is_inference", args[0] if args else False):
                raise ValueError("The atomized-ligand fast path is training-only.")
            pipeline = _optimize_atomized_ligand_pipeline(
                pipeline,
                generate_conformers=kwargs.get("generate_conformers", False),
                meta_conditioning_probabilities=kwargs.get(
                    "meta_conditioning_probabilities", {}
                ),
            )
        return pipeline
    finally:
        pipelines.SubsampleToTypes = original_subsample_to_types
        pipelines.PadTokensWithVirtualAtoms = original_pad_tokens
        pipelines.AddGroundTruthSequence = original_add_ground_truth_sequence
        pipelines.TrainingRoute = original_training_route
        pipelines.InferenceRoute = original_inference_route
        pipelines.TrainingConditionRoute = original_training_condition_route
        pipelines.AssignTypes = original_assign_types
        pipelines.CalculateHbondsPlus = original_calculate_hbonds_plus
        pipelines.CalculateNucleicAcidGeomFeats = original_calculate_na_geom
