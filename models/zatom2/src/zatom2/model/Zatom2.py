import os
from typing import Literal

import hydra
import torch
from foundry.utils.ddp import RankedLogger
from nucleus.model.inference_sampler import ZatomConditionalDiffusionSampler
from omegaconf import DictConfig
from rfd3.model.cfg_utils import (
    strip_f,
)
from torch import nn

ranked_logger = RankedLogger(__name__, rank_zero_only=True)

FORCE_CONDITIONING_CFG_FEATURES = (
    "mean_force_norm",
    "has_mean_force_norm",
    "force_rms_p95",
    "has_force_rms_p95",
)


class Zatom2(nn.Module):
    """
    Simplified model for generation. This module level serves to wrap
    the diffusion module of AF3 to be roughly equivalent to the AF3
    model w/o trunk processing. This allows the same sampler to be used.

    The model builds upon RFD3 components, but is generalized to support
    a unified representation of molecule, material, and protein systems.

    Args:
        c_s: int, dimension of 'global' features (e.g. sequence-level features)
        c_z: int, dimension of 'global' features (e.g. sequence-level features)
        c_atom: int, dimension of atom-level features
        c_atompair: int | None, dimension of atom-pair-level features
        token_initializer: DictConfig or dict, config for the token initializer module
        diffusion_module: DictConfig or dict, config for the diffusion module (e.g. AF3's denoiser)
        inference_sampler: DictConfig or dict, config for the inference sampler, which performs a full diffusion rollout during inference
    """

    def __init__(
        self,
        *,
        # Channel dimensions ('global' features)
        c_s: int,
        c_z: int,
        c_atom: int,
        c_atompair: int | None = None,
        # Arguments for modules that will be instantiated
        token_initializer: DictConfig | dict,
        diffusion_module: DictConfig | dict,
        inference_sampler: DictConfig | dict,
        **_,
    ):
        super().__init__()
        # Check for chunked P_LL mode via environment variable
        use_chunked_pll = os.environ.get("ZATOM2_LOW_MEMORY_MODE", None) == "1"
        ranked_logger.info(f"Zatom-2 initialized with chunked_pll={use_chunked_pll}")

        # Simple constant-feature initializer
        self.token_initializer = hydra.utils.instantiate(
            token_initializer,
            c_s=c_s,
            c_z=c_z,
            c_atom=c_atom,
            c_atompair=c_atompair,
            use_chunked_pll=use_chunked_pll,
        )

        # Diffusion module instantiated to allow for config scripting
        self.diffusion_module = hydra.utils.instantiate(
            diffusion_module,
            c_atom=c_atom,
            c_atompair=c_atompair,
            c_s=c_s,
            c_z=c_z,
        )

        # Work on a shallow copy because Hydra reuses the model config when
        # constructing checkpoint-backed inference models.
        inference_sampler = dict(inference_sampler)
        self.use_classifier_free_guidance = (
            inference_sampler["use_classifier_free_guidance"]
            and inference_sampler["cfg_scale"] != 1.0
        )
        inference_sampler["use_classifier_free_guidance"] = (
            self.use_classifier_free_guidance
        )
        self.cfg_feature_mode = inference_sampler.pop("cfg_feature_mode", "configured")
        self.cfg_features = self._resolve_cfg_features(
            self.cfg_feature_mode,
            inference_sampler.pop("cfg_features", []),
        )

        # ... initialize the inference sampler, which performs a full diffusion rollout during inference
        self.inference_sampler = ZatomConditionalDiffusionSampler(**inference_sampler)

    def _build_self_embedding_features(self, input: dict) -> dict:
        """Build a clean-target feature view for the SCD embedding pass."""
        f = input["f"]
        ground_truth = input.get("self_embedding_ground_truth") or {}
        f_self = dict(f)

        if "sequence_gt_I" in ground_truth and "restype" in f_self:
            sequence_gt_I = ground_truth["sequence_gt_I"].to(
                device=f_self["restype"].device,
            )
            if f_self["restype"].ndim == 1:
                if sequence_gt_I.ndim > 1:
                    sequence_gt_I = sequence_gt_I.argmax(dim=-1)
                restype_gt_I = sequence_gt_I.to(dtype=f_self["restype"].dtype)
            elif sequence_gt_I.ndim == f_self["restype"].ndim:
                restype_gt_I = sequence_gt_I.to(dtype=f_self["restype"].dtype)
            else:
                n_restype_classes = f_self["restype"].shape[-1]
                restype_gt_I = torch.nn.functional.one_hot(
                    sequence_gt_I.long().clamp(min=0, max=n_restype_classes - 1),
                    num_classes=n_restype_classes,
                ).to(dtype=f_self["restype"].dtype)
            f_self["restype"] = restype_gt_I

        if "atom_type_gt_I" in ground_truth and "ref_element" in f_self:
            tok_idx = f["atom_to_token_map"]
            atom_type_gt_I = ground_truth["atom_type_gt_I"].to(
                device=f_self["ref_element"].device,
            )
            atom_type_gt_L = atom_type_gt_I[tok_idx]
            if f_self["ref_element"].ndim == 1:
                ref_element_gt_L = atom_type_gt_L.to(dtype=f_self["ref_element"].dtype)
            else:
                n_element_classes = f_self["ref_element"].shape[-1]
                ref_element_gt_L = torch.nn.functional.one_hot(
                    atom_type_gt_L.long().clamp(min=0, max=n_element_classes - 1),
                    num_classes=n_element_classes,
                ).to(dtype=f_self["ref_element"].dtype)
            if "atom_type_valid_mask" in ground_truth:
                atom_type_mask_L = ground_truth["atom_type_valid_mask"].to(
                    device=f_self["ref_element"].device,
                    dtype=torch.bool,
                )[tok_idx]
                while atom_type_mask_L.ndim < ref_element_gt_L.ndim:
                    atom_type_mask_L = atom_type_mask_L.unsqueeze(-1)
                f_self["ref_element"] = torch.where(
                    atom_type_mask_L,
                    ref_element_gt_L,
                    f_self["ref_element"],
                )
            else:
                f_self["ref_element"] = ref_element_gt_L

        return f_self

    def _use_self_embedding_guidance(self, input: dict) -> bool:
        """Determine whether to use self-embedding guidance based on training mode, diffusion module settings, and optional input overrides."""
        if not self.training:
            return False
        if not getattr(self.diffusion_module, "self_embedding_guidance_enabled", False):
            return False

        override = input.get("use_self_embedding_guidance")
        if override is not None:
            return bool(override)

        probability = getattr(
            self.diffusion_module,
            "self_embedding_guidance_probability",
            0.0,
        )
        if probability <= 0.0:
            return False
        return torch.rand((), device=input["X_noisy_L"].device).item() < probability

    @staticmethod
    def _copy_property_features(input: dict) -> dict:
        """Copy model features and make system metadata available to the model."""
        f = dict(input["f"])
        ground_truth = input.get("ground_truth") or {}
        metadata_keys = (
            "domain_id_B",
            "domain_valid_mask_B",
            "total_charge_B",
            "spin_multiplicity_B",
            "charge_valid_mask_B",
            "spin_valid_mask_B",
            "property_valid_mask_B",
            "retained_atom_count_B",
            "source_atom_count_B",
        )
        for key in metadata_keys:
            if key not in f:
                if key in input:
                    f[key] = input[key]
                elif key in ground_truth:
                    f[key] = ground_truth[key]
        return f

    @staticmethod
    def _neutralize_mean_force_conditioning(f: dict) -> None:
        """Remove ground-truth force-statistic conditioning from a feature copy."""
        for key in FORCE_CONDITIONING_CFG_FEATURES:
            if key in f:
                f[key] = torch.zeros_like(f[key])

    @staticmethod
    def _resolve_cfg_features(
        cfg_feature_mode: str,
        cfg_features,
    ) -> list[str]:
        """Resolve the feature set used to construct the CFG reference branch."""
        if cfg_feature_mode == "force_only":
            return list(FORCE_CONDITIONING_CFG_FEATURES)
        if cfg_feature_mode == "configured":
            if cfg_features is None:
                return []
            if isinstance(cfg_features, str):
                raise TypeError("cfg_features must be a sequence of feature names.")
            return list(cfg_features)
        raise ValueError(
            "cfg_feature_mode must be either 'configured' or 'force_only', "
            f"got {cfg_feature_mode!r}."
        )

    @staticmethod
    def _zero_cfg_features(f: dict, cfg_features: list[str]) -> dict:
        """Zero selected tensors without changing any other conditioning."""
        cfg_features = set(cfg_features)
        return {
            key: torch.zeros_like(value)
            if key in cfg_features and isinstance(value, torch.Tensor)
            else value
            for key, value in f.items()
        }

    @staticmethod
    def _strip_cfg_features(f: dict, cfg_features: list[str]) -> dict:
        """Strip CFG tensor features while preserving non-tensor metadata.

        Upstream ``strip_f`` also crops guidepost tensor features, but assumes
        every feature value exposes ``Tensor.shape``. Sparse inference batches
        may additionally carry list-valued metadata that the denoiser does not
        consume. Keep that metadata unchanged and delegate the tensor handling
        to the upstream implementation.
        """
        tensor_features = {
            key: value for key, value in f.items() if isinstance(value, torch.Tensor)
        }
        stripped_features = strip_f(tensor_features, cfg_features)
        return {**f, **stripped_features}

    @classmethod
    def _build_cfg_reference_features(
        cls,
        f: dict,
        *,
        cfg_feature_mode: str,
        cfg_features: list[str],
    ) -> dict:
        """Construct a CFG reference feature dictionary.

        The legacy ``configured`` mode delegates to RFdiffusion3's CFG helper,
        which can also crop unindexed guideposts. ``force_only`` deliberately
        avoids that behavior: it removes only force values and their presence
        flags, so task, graph, composition, charge, spin, and motif conditions
        are identical in the conditional and reference branches.
        """
        if cfg_feature_mode == "force_only":
            return cls._zero_cfg_features(f, cfg_features)
        return cls._strip_cfg_features(f, cfg_features)

    @staticmethod
    def _expose_property_electronic_metadata(f: dict) -> None:
        """Broadcast system charge/spin into token features for the MLIP API."""
        batch_token_lvl = f["batch_token_lvl"].to(dtype=torch.long)
        metadata = (
            (
                "total_charge_B",
                "charge_valid_mask_B",
                "total_charge",
                "has_total_charge",
            ),
            (
                "spin_multiplicity_B",
                "spin_valid_mask_B",
                "spin_multiplicity",
                "has_spin_multiplicity",
            ),
        )
        for value_key_B, valid_key_B, token_value_key, token_valid_key in metadata:
            if value_key_B not in f:
                continue
            values_B = torch.as_tensor(
                f[value_key_B], device=batch_token_lvl.device
            ).reshape(-1)
            if batch_token_lvl.numel() and int(batch_token_lvl.max()) >= len(values_B):
                raise ValueError(f"{value_key_B} does not cover every system.")
            valid_B = torch.isfinite(values_B)
            if valid_key_B in f:
                valid_B &= torch.as_tensor(
                    f[valid_key_B],
                    device=batch_token_lvl.device,
                    dtype=torch.bool,
                ).reshape(-1)
            f[token_value_key] = values_B[batch_token_lvl].unsqueeze(-1)
            f[token_valid_key] = (
                valid_B[batch_token_lvl].to(dtype=torch.float32).unsqueeze(-1)
            )

    @staticmethod
    def _mask_property_atom_types(f: dict) -> None:
        """Mirror training-time atom-type masking without mutating caller data."""
        if "atom_to_token_map" not in f:
            raise ValueError("Masked atom-type inference requires atom_to_token_map.")
        atom_to_token_map = f["atom_to_token_map"]
        if "restype" in f:
            n_tokens = len(f["restype"])
        elif atom_to_token_map.numel() > 0:
            n_tokens = int(atom_to_token_map.max().item()) + 1
        else:
            n_tokens = 0

        f["atom_type_generation_mask"] = torch.ones(
            n_tokens,
            dtype=torch.bool,
            device=atom_to_token_map.device,
        )
        for key in ("ref_element", "ref_atom_name_chars"):
            if key in f:
                f[key] = torch.zeros_like(f[key])
        if "token_bonds" in f:
            f["token_bonds"] = torch.zeros_like(f["token_bonds"])

    @staticmethod
    def _validate_known_property_atom_types(f: dict) -> None:
        """Reject missing or masked element inputs in known-type mode."""
        if "ref_element" not in f or "batch_atom_lvl" not in f:
            raise ValueError(
                "Known atom-type inference requires exact, unmasked ref_element "
                "features and batch_atom_lvl."
            )
        ref_element = torch.as_tensor(f["ref_element"])
        n_atoms = int(torch.as_tensor(f["batch_atom_lvl"]).numel())
        if ref_element.numel() == n_atoms:
            valid = torch.isfinite(ref_element) & (ref_element > 0)
            valid = valid.reshape(-1)
        elif ref_element.ndim >= 2 and ref_element.shape[-1] == 128:
            element_vectors = ref_element.reshape(-1, 128)
            valid = torch.isfinite(element_vectors).all(dim=-1) & (
                element_vectors.abs().sum(dim=-1) > 0
            )
        else:
            raise ValueError(
                "Known ref_element must contain atomic numbers [L] or "
                "128-channel element vectors [L, 128]."
            )
        if valid.numel() != n_atoms or not valid.all():
            raise ValueError(
                "Known atom-type inference received missing or masked element "
                "features; provide exact atom types or use atom_types='masked'."
            )

    @staticmethod
    def _has_mean_force_conditioning(f: dict) -> bool:
        """Return whether any system exposes force-statistic conditioning."""
        return any(
            bool(torch.as_tensor(f[key], dtype=torch.bool).any().item())
            for key in ("has_mean_force_norm", "has_force_rms_p95")
            if key in f
        )

    @staticmethod
    def _resolve_system_metadata_validity(
        f: dict,
        *,
        value_key_B: str,
        valid_key_B: str,
        token_value_key: str,
        token_valid_key: str,
        batch_size: int,
        require_positive: bool = False,
        require_integer: bool = False,
    ) -> torch.Tensor:
        """Resolve per-system metadata presence from scalar or token features."""
        device = f["batch_token_lvl"].device
        if value_key_B in f:
            values_B = torch.as_tensor(f[value_key_B], device=device).reshape(-1)
            if values_B.numel() != batch_size:
                raise ValueError(f"{value_key_B} must contain one value per system.")
            valid_B = torch.isfinite(values_B)
            if require_positive:
                valid_B &= values_B > 0
            if require_integer:
                valid_B &= torch.isclose(
                    values_B,
                    values_B.round(),
                    rtol=0.0,
                    atol=1e-6,
                )
            if valid_key_B in f:
                explicit_valid_B = torch.as_tensor(
                    f[valid_key_B], device=device, dtype=torch.bool
                ).reshape(-1)
                if explicit_valid_B.numel() != batch_size:
                    raise ValueError(
                        f"{valid_key_B} must contain one value per system."
                    )
                valid_B &= explicit_valid_B
            return valid_B

        if token_value_key not in f:
            return torch.zeros(batch_size, device=device, dtype=torch.bool)

        batch_token_lvl = f["batch_token_lvl"].to(device=device, dtype=torch.long)
        values_I = torch.as_tensor(f[token_value_key], device=device).reshape(
            batch_token_lvl.numel(), -1
        )
        valid_I = torch.isfinite(values_I).all(dim=-1)
        if require_positive:
            valid_I &= (values_I > 0).all(dim=-1)
        if require_integer:
            valid_I &= torch.isclose(
                values_I,
                values_I.round(),
                rtol=0.0,
                atol=1e-6,
            ).all(dim=-1)
        if token_valid_key in f:
            explicit_valid_I = torch.as_tensor(
                f[token_valid_key], device=device, dtype=torch.bool
            ).reshape(batch_token_lvl.numel(), -1)
            valid_I &= explicit_valid_I.all(dim=-1)

        valid_count_B = torch.zeros(batch_size, device=device, dtype=torch.long)
        token_count_B = torch.zeros_like(valid_count_B)
        valid_count_B.index_add_(0, batch_token_lvl, valid_I.long())
        token_count_B.index_add_(0, batch_token_lvl, torch.ones_like(batch_token_lvl))
        return (token_count_B > 0) & (valid_count_B == token_count_B)

    def _validate_omol_property_metadata(self, f: dict, *, batch_size: int) -> None:
        """Require total charge and spin multiplicity for OMol systems."""
        domain_id_B = self.diffusion_module._property_domain_ids(
            f,
            batch_size=batch_size,
            device=f["batch_token_lvl"].device,
        )
        domain_names = tuple(self.diffusion_module.property_domain_names)
        omol_domain_ids = {
            index for index, name in enumerate(domain_names) if name.lower() == "omol25"
        }
        if not omol_domain_ids:
            return
        is_omol_B = torch.zeros_like(domain_id_B, dtype=torch.bool)
        for domain_id in omol_domain_ids:
            is_omol_B |= domain_id_B == domain_id
        if not is_omol_B.any():
            return

        charge_valid_B = self._resolve_system_metadata_validity(
            f,
            value_key_B="total_charge_B",
            valid_key_B="charge_valid_mask_B",
            token_value_key="total_charge",
            token_valid_key="has_total_charge",
            batch_size=batch_size,
            require_integer=True,
        )
        spin_valid_B = self._resolve_system_metadata_validity(
            f,
            value_key_B="spin_multiplicity_B",
            valid_key_B="spin_valid_mask_B",
            token_value_key="spin_multiplicity",
            token_valid_key="has_spin_multiplicity",
            batch_size=batch_size,
            require_positive=True,
            require_integer=True,
        )
        invalid_omol_B = is_omol_B & ~(charge_valid_B & spin_valid_B)
        if invalid_omol_B.any():
            invalid = invalid_omol_B.nonzero(as_tuple=False).flatten().tolist()
            raise ValueError(
                "OMol25 property inference requires valid total charge and spin "
                f"multiplicity metadata; missing for systems {invalid}."
            )

    @staticmethod
    def _generated_property_atom_type_mode(
        f: dict,
    ) -> Literal["known", "masked"] | None:
        """Return an unambiguous atom-type mode for generated property decoding.

        Hybrid batches require combining exact and predicted elemental counts,
        which the current reconstruction helper deliberately does not infer.
        """
        generation_mask = f.get("atom_type_generation_mask")
        if generation_mask is None:
            return None
        generation_mask = torch.as_tensor(generation_mask, dtype=torch.bool).reshape(-1)
        if generation_mask.numel() == 0:
            return None
        if generation_mask.all():
            return "masked"
        if not generation_mask.any():
            return "known"
        return None

    def _add_generated_physical_properties(self, sample: dict, f: dict) -> None:
        """Decode captured final-step property outputs when type counts are safe."""
        if not getattr(
            self.diffusion_module, "atomistic_property_prediction_enabled", False
        ):
            return
        if not {
            "normalized_energy_B",
            "normalized_forces_L",
        }.issubset(sample):
            return
        if self._has_mean_force_conditioning(f):
            ranked_logger.info(
                "Skipping generated property reconstruction because force-statistic "
                "conditioning is present."
            )
            return

        atom_type_mode = self._generated_property_atom_type_mode(f)
        if atom_type_mode is None:
            return
        if atom_type_mode == "masked" and "atom_type_logits_I" not in sample:
            return
        if atom_type_mode == "known" and "ref_element" not in f:
            return

        try:
            batch_size = int(sample["normalized_energy_B"].shape[-1])
            self._validate_omol_property_metadata(f, batch_size=batch_size)
            physical_outputs = self.diffusion_module.reconstruct_atomistic_properties(
                sample,
                f,
                atom_types=atom_type_mode,
            )
        except (KeyError, ValueError) as error:
            # Generation inputs need not carry valid property-domain metadata or
            # elemental references. Keep normalized heads available in that case.
            ranked_logger.info(
                "Skipping generated property reconstruction: " f"{error}"
            )
            return
        sample.update(physical_outputs)

    @torch.no_grad()
    def predict_properties(
        self,
        input: dict,
        *,
        atom_types: Literal["known", "masked"] = "known",
        return_embeddings: bool = False,
        allow_missing_elemental_references: bool = False,
    ) -> dict:
        """Predict energy and forces with one clean-endpoint model evaluation.

        This path bypasses the generative sampler, evaluates the shared backbone
        at flow-matching ``t=1``, and always removes force-statistic conditioning.
        Exact atom types are used by default. ``atom_types='masked'`` applies the
        same leakage-prevention masks used in joint generative training and uses
        soft predicted elemental counts to reconstruct total energy.
        """
        if atom_types not in ("known", "masked"):
            raise ValueError("atom_types must be either 'known' or 'masked'.")
        if not getattr(
            self.diffusion_module, "atomistic_property_prediction_enabled", False
        ):
            raise RuntimeError("Atomistic property heads are not enabled.")
        if getattr(self.diffusion_module, "f_pred", None) != "flow_matching":
            raise RuntimeError(
                "predict_properties uses t=1 as the clean endpoint and therefore "
                "requires a flow-matching diffusion module."
            )

        f = self._copy_property_features(input)
        self._neutralize_mean_force_conditioning(f)
        self._expose_property_electronic_metadata(f)
        if atom_types == "masked":
            self._mask_property_atom_types(f)
        else:
            self._validate_known_property_atom_types(f)

        X_clean_L = input.get("X_clean_L")
        if X_clean_L is None:
            X_clean_L = input.get("X_noisy_L")
        if X_clean_L is None:
            X_clean_L = input.get("X_L")
        if X_clean_L is None:
            X_clean_L = input.get("coord_atom_lvl_to_be_noised")
        if X_clean_L is None:
            raise ValueError(
                "predict_properties requires clean coordinates in X_clean_L, "
                "X_noisy_L, X_L, or coord_atom_lvl_to_be_noised."
            )
        if X_clean_L.ndim == 2:
            X_clean_L = X_clean_L.unsqueeze(0)
        if X_clean_L.ndim != 3 or X_clean_L.shape[-1] != 3:
            raise ValueError("Clean coordinates must have shape [L, 3] or [D, L, 3].")

        diffusion_batch_size = X_clean_L.shape[0]
        f["diffusion_batch_size"] = diffusion_batch_size
        if "batch_size" in f:
            batch_size_value = f["batch_size"]
            batch_size = (
                int(batch_size_value.item())
                if torch.is_tensor(batch_size_value)
                else int(batch_size_value)
            )
        elif f["batch_token_lvl"].numel() == 0:
            batch_size = 0
            f["batch_size"] = batch_size
        else:
            batch_size = int(f["batch_token_lvl"].max().item()) + 1
            f["batch_size"] = batch_size
        self._validate_omol_property_metadata(f, batch_size=batch_size)

        was_training = self.training
        self.eval()
        try:
            initializer_outputs = self.token_initializer(f)
            clean_time_B = torch.ones(
                (diffusion_batch_size, batch_size),
                device=X_clean_L.device,
                dtype=X_clean_L.dtype,
            )
            outputs = self.diffusion_module(
                X_noisy_L=X_clean_L,
                t=clean_time_B,
                f=f,
                n_recycle=1,
                return_property_embeddings=return_embeddings,
                **initializer_outputs,
            )
            reconstruction_kwargs = {"atom_types": atom_types}
            if allow_missing_elemental_references:
                reconstruction_kwargs["allow_missing_elemental_references"] = True
            physical_outputs = self.diffusion_module.reconstruct_atomistic_properties(
                outputs,
                f,
                **reconstruction_kwargs,
            )
        finally:
            self.train(was_training)

        result = {
            "normalized_energy_B": outputs["normalized_energy_B"],
            "normalized_forces_L": outputs["normalized_forces_L"],
            **physical_outputs,
        }
        if atom_types == "masked":
            result["atom_type_logits_I"] = outputs["atom_type_logits_I"]
            result["atom_type_indices_I"] = outputs["atom_type_indices_I"]
        if return_embeddings:
            result.update({key: outputs[key] for key in ("Q_sys", "Q_I", "Q_L")})

        # The usual property API has one clean view, so remove only that leading
        # view dimension while retaining the system/atom dimensions.
        if diffusion_batch_size == 1:
            for key, value in tuple(result.items()):
                if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == 1:
                    result[key] = value.squeeze(0)
        return result

    def forward(
        self,
        input: dict,
        coord_atom_lvl_to_be_noised: torch.Tensor = None,
        n_cycle=None,
        property_prediction_atom_types: Literal["known", "masked"] | None = None,
        allow_missing_property_elemental_references: bool = False,
        **_,
    ) -> dict:
        """
        Forward method that performs either a single denoising step (during training) or a full diffusion rollout (during inference).

        Args:
            input: dict, should contain keys 'f' (features), 'X_noisy_L' (noisy coordinates), 't' (diffusion timestep), and optionally 't_L' and 't_I' for the diffusion module.
            coord_atom_lvl_to_be_noised: torch.Tensor, indices of atom-level coordinates to be noised during inference (used for the sampler).
            n_cycle: int, number of recycling cycles (used for the diffusion module during training).
            property_prediction_atom_types: If set during inference, bypass the
                generative sampler and call :meth:`predict_properties` once at
                the clean endpoint using known or masked atom types. Routing this
                through ``forward`` preserves EMA shadow-model dispatch.

        Returns:
            During training: denoised coordinates of shape [D, L, 3].
            During inference: generated coordinates of shape [D, L, 3] after a full diffusion rollout.
        """
        if property_prediction_atom_types is not None:
            return self.predict_properties(
                input,
                atom_types=property_prediction_atom_types,
                allow_missing_elemental_references=(
                    allow_missing_property_elemental_references
                ),
            )

        use_self_embedding_guidance = self._use_self_embedding_guidance(input)
        initializer_outputs = self.token_initializer(input["f"])

        if self.training:
            self_embedding_kwargs = {}
            if use_self_embedding_guidance:
                if "X_clean_L" not in input:
                    raise ValueError(
                        "Self-embedding guidance requires `X_clean_L` in the model input."
                    )
                # f_self_embedding is input["f"] plus ground-truth restoration for restype and ref_element, if available
                f_self_embedding = self._build_self_embedding_features(input)
                self_embedding_kwargs = {
                    "use_self_embedding_guidance": True,
                    "X_self_embedding_L": input["X_clean_L"],
                    "f_self_embedding": f_self_embedding,
                    "self_embedding_initializer_outputs": self.token_initializer(
                        f_self_embedding
                    ),
                }

            # Single denoising step
            return self.diffusion_module(
                X_noisy_L=input["X_noisy_L"],
                t=input["t"],
                f=input["f"],
                n_recycle=2 if use_self_embedding_guidance else n_cycle,
                t_L=input.get("t_L", None),
                t_I=input.get("t_I", None),
                **self_embedding_kwargs,
                **initializer_outputs,
            )  # [D, L, 3]
        else:
            if self.use_classifier_free_guidance:
                f_ref = self._build_cfg_reference_features(
                    input["f"],
                    cfg_feature_mode=self.cfg_feature_mode,
                    cfg_features=self.cfg_features,
                )
                ref_initializer_outputs = self.token_initializer(f_ref)
            else:
                f_ref = None
                ref_initializer_outputs = None

            sample = self.inference_sampler.sample_diffusion_like_af3(
                f=input["f"],
                f_ref=f_ref,  # for cfg
                diffusion_module=self.diffusion_module,
                diffusion_batch_size=coord_atom_lvl_to_be_noised.shape[0],
                coord_atom_lvl_to_be_noised=coord_atom_lvl_to_be_noised,
                # Forwarded as **kwargs:
                initializer_outputs=initializer_outputs,
                ref_initializer_outputs=ref_initializer_outputs,  # for cfg
            )
            self._add_generated_physical_properties(sample, input["f"])
            return sample
