import importlib.util
import pathlib
import sys
import types

import pytest
import torch
from torch import nn

ROOT = pathlib.Path(__file__).resolve().parents[3]


def _load_module_from_path(module_name: str, path: pathlib.Path):
    """Load a Python module from a given file path."""
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_zatom2(monkeypatch):
    """Load the Zatom2 class from the source file, mocking dependencies that are not available in the test environment."""

    def _register(name: str, module: types.ModuleType) -> None:
        monkeypatch.setitem(sys.modules, name, module)

    hydra = types.ModuleType("hydra")
    hydra.utils = types.ModuleType("hydra.utils")
    hydra.utils.instantiate = lambda *args, **kwargs: None
    _register("hydra", hydra)
    _register("hydra.utils", hydra.utils)

    foundry_ddp = types.ModuleType("foundry.utils.ddp")

    class RankedLogger:
        def __init__(self, *args, **kwargs):
            pass

        def info(self, *args, **kwargs):
            pass

    foundry_ddp.RankedLogger = RankedLogger
    _register("foundry", types.ModuleType("foundry"))
    _register("foundry.utils", types.ModuleType("foundry.utils"))
    _register("foundry.utils.ddp", foundry_ddp)

    nucleus_sampler = types.ModuleType("nucleus.model.inference_sampler")
    nucleus_sampler.ZatomConditionalDiffusionSampler = object
    nucleus_encoders = types.ModuleType("nucleus.model.layers.encoders")
    nucleus_encoders.TokenInitializer = object
    _register("nucleus", types.ModuleType("nucleus"))
    _register("nucleus.model", types.ModuleType("nucleus.model"))
    _register("nucleus.model.inference_sampler", nucleus_sampler)
    _register("nucleus.model.layers", types.ModuleType("nucleus.model.layers"))
    _register("nucleus.model.layers.encoders", nucleus_encoders)

    omegaconf = types.ModuleType("omegaconf")
    omegaconf.DictConfig = dict
    _register("omegaconf", omegaconf)

    rfd3_cfg_utils = types.ModuleType("rfd3.model.cfg_utils")
    rfd3_cfg_utils.strip_f = lambda f, features: f
    _register("rfd3", types.ModuleType("rfd3"))
    _register("rfd3.model", types.ModuleType("rfd3.model"))
    _register("rfd3.model.cfg_utils", rfd3_cfg_utils)

    return _load_module_from_path(
        "test_zatom2_model_wrapper",
        ROOT / "models/zatom2/src/zatom2/model/Zatom2.py",
    ).Zatom2


def _load_conditioner():
    """Load the SelfEmbeddingConditioner class from the source file."""
    return _load_module_from_path(
        "test_self_embedding_guidance_module",
        ROOT / "src/nucleus/model/self_embedding_guidance.py",
    ).SelfEmbeddingConditioner


def test_self_embedding_conditioner_sum_pools_token_embeddings():
    """Test that the SelfEmbeddingConditioner correctly sums token embeddings based on batch indices."""
    conditioner_cls = _load_conditioner()
    conditioner = conditioner_cls(c_token=2, c_s=3, condition_dropout=0.0)
    conditioner.pre_proj = nn.Identity()
    conditioner.post_pool_mlp = nn.Identity()

    token_embeddings = torch.tensor(
        [[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]],
    )
    batch_token_lvl = torch.tensor([0, 1, 0], dtype=torch.long)

    pooled = conditioner.embed(
        token_embeddings,
        batch_token_lvl=batch_token_lvl,
        batch_size=2,
    )

    expected = torch.tensor([[[6.0, 8.0], [3.0, 4.0]]])
    torch.testing.assert_close(pooled, expected)


def test_self_embedding_conditioner_sum_pools_atom_embeddings():
    """Test that the SelfEmbeddingConditioner correctly sums atom embeddings based on batch indices."""
    conditioner_cls = _load_conditioner()
    conditioner = conditioner_cls(c_token=2, c_s=3, c_atom=2, condition_dropout=0.0)
    conditioner.atom_pre_proj = nn.Identity()
    conditioner.post_pool_mlp = nn.Identity()

    atom_embeddings = torch.tensor(
        [[[1.0, 1.0], [2.0, 3.0], [4.0, 5.0], [7.0, 11.0]]],
    )
    batch_atom_lvl = torch.tensor([0, 1, 0, 1], dtype=torch.long)

    pooled = conditioner.embed(
        atom_embeddings,
        batch_idx=batch_atom_lvl,
        batch_size=2,
        input_level="atom",
    )

    expected = torch.tensor([[[5.0, 6.0], [9.0, 14.0]]])
    torch.testing.assert_close(pooled, expected)


def test_self_embedding_conditioner_can_choose_atom_or_token_pooling():
    """Test that the SelfEmbeddingConditioner can choose between atom and token pooling based on the atom_probability."""
    conditioner_cls = _load_conditioner()
    conditioner = conditioner_cls(c_token=2, c_s=3, c_atom=2, condition_dropout=0.0)
    conditioner.pre_proj = nn.Identity()
    conditioner.atom_pre_proj = nn.Identity()
    conditioner.post_pool_mlp = nn.Identity()

    token_embeddings = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    atom_embeddings = torch.tensor([[[5.0, 7.0], [11.0, 13.0], [17.0, 19.0]]])
    batch_token_lvl = torch.tensor([0, 1], dtype=torch.long)
    batch_atom_lvl = torch.tensor([0, 1, 1], dtype=torch.long)

    token_condition = conditioner.embed_random_level(
        token_embeddings=token_embeddings,
        atom_embeddings=atom_embeddings,
        batch_token_lvl=batch_token_lvl,
        batch_atom_lvl=batch_atom_lvl,
        batch_size=2,
        atom_probability=0.0,
    )
    atom_condition = conditioner.embed_random_level(
        token_embeddings=token_embeddings,
        atom_embeddings=atom_embeddings,
        batch_token_lvl=batch_token_lvl,
        batch_atom_lvl=batch_atom_lvl,
        batch_size=2,
        atom_probability=1.0,
    )

    torch.testing.assert_close(
        token_condition, torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    )
    torch.testing.assert_close(
        atom_condition,
        torch.tensor([[[5.0, 7.0], [28.0, 32.0]]]),
    )


def test_self_embedding_conditioner_uses_configured_atom_probability():
    """Test that the SelfEmbeddingConditioner uses the configured atom_probability."""
    conditioner_cls = _load_conditioner()
    conditioner = conditioner_cls(
        c_token=2,
        c_s=3,
        c_atom=2,
        atom_probability=1.0,
        condition_dropout=0.0,
    )
    conditioner.pre_proj = nn.Identity()
    conditioner.atom_pre_proj = nn.Identity()
    conditioner.post_pool_mlp = nn.Identity()

    condition = conditioner.embed_random_level(
        token_embeddings=torch.tensor([[[1.0, 2.0]]]),
        atom_embeddings=torch.tensor([[[5.0, 7.0], [11.0, 13.0]]]),
        batch_token_lvl=torch.tensor([0], dtype=torch.long),
        batch_atom_lvl=torch.tensor([0, 0], dtype=torch.long),
        batch_size=1,
    )

    torch.testing.assert_close(condition, torch.tensor([[[16.0, 20.0]]]))


def test_self_embedding_conditioner_uses_adaptive_layer_norm_conditioning():
    """Test that the SelfEmbeddingConditioner correctly applies adaptive layer normalization to the inputs."""
    conditioner_cls = _load_conditioner()
    conditioner = conditioner_cls(c_token=2, c_s=3, condition_dropout=0.0)
    conditioner.eval()

    with torch.no_grad():
        conditioner.to_token_adaln.weight.zero_()
        conditioner.to_token_adaln.bias.copy_(torch.tensor([0.5, -0.25, 1.0, -2.0]))
        conditioner.to_s_adaln.weight.zero_()
        conditioner.to_s_adaln.bias.copy_(
            torch.tensor([0.0, 0.25, -0.5, 2.0, -1.0, 0.5])
        )

    A_I = torch.tensor([[[1.0, 3.0], [2.0, 6.0], [5.0, 1.0]]])
    S_I = torch.tensor([[[1.0, 2.0, 4.0], [3.0, 3.0, 9.0], [2.0, 5.0, 8.0]]])
    condition_B = torch.zeros(1, 2, 2)
    batch_token_lvl = torch.tensor([0, 1, 0], dtype=torch.long)

    conditioned_A_I, conditioned_S_I = conditioner.condition_token_inputs(
        A_I=A_I,
        S_I=S_I,
        condition_B=condition_B,
        batch_token_lvl=batch_token_lvl,
    )

    token_scale = torch.tensor([0.5, -0.25]).view(1, 1, 2)
    token_shift = torch.tensor([1.0, -2.0]).view(1, 1, 2)
    s_scale = torch.tensor([0.0, 0.25, -0.5]).view(1, 1, 3)
    s_shift = torch.tensor([2.0, -1.0, 0.5]).view(1, 1, 3)
    expected_A_I = (
        torch.nn.functional.layer_norm(A_I, (2,)) * (1 + token_scale) + token_shift
    )
    expected_S_I = torch.nn.functional.layer_norm(S_I, (3,)) * (1 + s_scale) + s_shift

    torch.testing.assert_close(conditioned_A_I, expected_A_I)
    torch.testing.assert_close(conditioned_S_I, expected_S_I)


def test_self_embedding_conditioner_no_condition_adds_zero_parameter_dependency():
    """Test that when no condition is provided, the SelfEmbeddingConditioner still adds a zero parameter dependency for gradient flow."""
    conditioner_cls = _load_conditioner()
    conditioner = conditioner_cls(c_token=2, c_s=3, c_atom=2, condition_dropout=0.0)

    A_I = torch.randn(1, 4, 2)
    S_I = torch.randn(1, 4, 3)

    conditioned_A_I, conditioned_S_I = conditioner.condition_token_inputs(
        A_I=A_I,
        S_I=S_I,
        condition_B=None,
        batch_token_lvl=torch.tensor([0, 0, 0, 0], dtype=torch.long),
    )
    torch.testing.assert_close(conditioned_A_I, A_I)
    torch.testing.assert_close(conditioned_S_I, S_I)

    (conditioned_A_I.sum() + conditioned_S_I.sum()).backward()
    for parameter in conditioner.parameters():
        assert parameter.grad is not None
        torch.testing.assert_close(parameter.grad, torch.zeros_like(parameter.grad))


def test_zatom2_scd_override_uses_clean_features_and_two_passes(monkeypatch):
    """Test that when self-embedding guidance is enabled, the Zatom2 model uses clean features and performs two passes."""
    zatom2_cls = _load_zatom2(monkeypatch)

    class FakeTokenInitializer(nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = []

        def forward(self, f):
            self.calls.append(f)
            return {"initializer_call_count": torch.tensor(len(self.calls))}

    class FakeDiffusionModule(nn.Module):
        self_embedding_guidance_enabled = True
        self_embedding_guidance_probability = 0.0

        def __init__(self):
            super().__init__()
            self.calls = []

        def forward(self, **kwargs):
            self.calls.append(kwargs)
            return {"X_L": torch.zeros(1, 4, 3)}

    model = zatom2_cls.__new__(zatom2_cls)
    nn.Module.__init__(model)
    model.token_initializer = FakeTokenInitializer()
    model.diffusion_module = FakeDiffusionModule()
    model.training = True

    f = {
        "restype": torch.zeros(3, dtype=torch.long),
        "ref_element": torch.zeros(4, dtype=torch.long),
        "atom_to_token_map": torch.tensor([0, 1, 2, 1], dtype=torch.long),
    }
    model.forward(
        input={
            "X_noisy_L": torch.zeros(1, 4, 3),
            "X_clean_L": torch.ones(1, 4, 3),
            "t": torch.ones(1, 1),
            "f": f,
            "use_self_embedding_guidance": True,
            "self_embedding_ground_truth": {
                "sequence_gt_I": torch.tensor([5, 6, 7], dtype=torch.long),
                "atom_type_gt_I": torch.tensor([6, 8, 1], dtype=torch.long),
                "atom_type_valid_mask": torch.tensor([True, False, True]),
            },
        },
        n_cycle=1,
    )

    assert len(model.token_initializer.calls) == 2
    clean_f = model.token_initializer.calls[1]
    torch.testing.assert_close(clean_f["restype"], torch.tensor([5, 6, 7]))
    torch.testing.assert_close(clean_f["ref_element"], torch.tensor([6, 0, 1, 0]))

    diffusion_call = model.diffusion_module.calls[0]
    assert diffusion_call["n_recycle"] == 2
    assert diffusion_call["use_self_embedding_guidance"] is True
    assert diffusion_call["f_self_embedding"] is clean_f
    torch.testing.assert_close(
        diffusion_call["X_self_embedding_L"], torch.ones(1, 4, 3)
    )


def test_zatom2_scd_clean_feature_restore_matches_vector_ref_element(monkeypatch):
    """Test that the clean feature restoration correctly matches the vector reference element."""
    zatom2_cls = _load_zatom2(monkeypatch)
    model = zatom2_cls.__new__(zatom2_cls)
    sequence_gt_I = torch.zeros(3, 32)
    sequence_gt_I[torch.arange(3), torch.tensor([5, 6, 7])] = 1.0

    clean_f = model._build_self_embedding_features(
        {
            "f": {
                "restype": torch.zeros(3, 32),
                "ref_element": torch.zeros(4, 128),
                "atom_to_token_map": torch.tensor([0, 1, 2, 1], dtype=torch.long),
            },
            "self_embedding_ground_truth": {
                "sequence_gt_I": sequence_gt_I,
                "atom_type_gt_I": torch.tensor([6, 8, 1], dtype=torch.long),
                "atom_type_valid_mask": torch.tensor([True, False, True]),
            },
        }
    )

    expected_ref_element = torch.zeros(4, 128)
    expected_ref_element[0, 6] = 1.0
    expected_ref_element[2, 1] = 1.0
    torch.testing.assert_close(clean_f["restype"], sequence_gt_I)
    torch.testing.assert_close(clean_f["ref_element"], expected_ref_element)


def test_zatom2_scd_override_false_keeps_normal_recycling(monkeypatch):
    """Test that when self-embedding guidance is disabled, the Zatom2 model does not use clean features and performs normal recycling."""
    zatom2_cls = _load_zatom2(monkeypatch)

    class FakeTokenInitializer(nn.Module):
        """A fake token initializer that records calls and returns an empty dictionary."""

        def __init__(self):
            super().__init__()
            self.calls = []

        def forward(self, f):
            self.calls.append(f)
            return {}

    class FakeDiffusionModule(nn.Module):
        """A fake diffusion module that records calls and returns a zero tensor."""

        self_embedding_guidance_enabled = True
        self_embedding_guidance_probability = 1.0

        def __init__(self):
            super().__init__()
            self.calls = []

        def forward(self, **kwargs):
            self.calls.append(kwargs)
            return {"X_L": torch.zeros(1, 2, 3)}

    model = zatom2_cls.__new__(zatom2_cls)
    nn.Module.__init__(model)
    model.token_initializer = FakeTokenInitializer()
    model.diffusion_module = FakeDiffusionModule()
    model.training = True

    model.forward(
        input={
            "X_noisy_L": torch.zeros(1, 2, 3),
            "X_clean_L": torch.ones(1, 2, 3),
            "t": torch.ones(1, 1),
            "f": {
                "restype": torch.zeros(2, dtype=torch.long),
                "ref_element": torch.zeros(2, dtype=torch.long),
                "atom_to_token_map": torch.arange(2),
            },
            "use_self_embedding_guidance": False,
        },
        n_cycle=3,
    )

    assert len(model.token_initializer.calls) == 1
    diffusion_call = model.diffusion_module.calls[0]
    assert diffusion_call["n_recycle"] == 3
    assert "use_self_embedding_guidance" not in diffusion_call


class _PropertyTokenInitializer(nn.Module):
    """A fake token initializer that records calls and returns an empty dictionary."""

    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, f):
        """Forward method that records the input features and returns an empty dictionary."""
        self.calls.append(f)
        return {}


class _PropertyDiffusionModule(nn.Module):
    """A fake diffusion module that records calls and returns dummy outputs."""

    atomistic_property_prediction_enabled = True
    property_domain_names = ("omol25", "omat24")
    f_pred = "flow_matching"

    def __init__(self):
        super().__init__()
        self.calls = []
        self.reconstruction_atom_types = []

    def _property_domain_ids(self, f, *, batch_size, device):
        """Return the domain IDs for the given features, reshaped to match the batch size."""
        return torch.as_tensor(f["domain_id_B"], device=device).reshape(batch_size)

    def forward(self, **kwargs):
        """Forward method that records the input arguments and returns dummy outputs for testing."""
        self.calls.append(kwargs)
        D = kwargs["X_noisy_L"].shape[0]
        L = kwargs["X_noisy_L"].shape[1]
        I = len(kwargs["f"]["restype"])
        B = int(kwargs["f"]["batch_size"])
        return {
            "normalized_energy_B": torch.full((D, B), 0.25),
            "normalized_forces_L": torch.full((D, L, 3), -0.5),
            "atom_type_logits_I": torch.zeros(D, I, 128),
            "atom_type_indices_I": torch.ones(D, I, dtype=torch.long),
            "Q_sys": torch.ones(D, B, 4),
            "Q_I": torch.ones(D, I, 5),
            "Q_L": torch.ones(D, L, 6),
        }

    def reconstruct_atomistic_properties(self, outputs, f, *, atom_types):
        """Reconstruct atomistic properties from the outputs and record the atom types used for reconstruction."""
        self.reconstruction_atom_types.append(atom_types)
        D, B = outputs["normalized_energy_B"].shape
        L = outputs["normalized_forces_L"].shape[1]
        return {
            "energy_B": torch.full((D, B), 7.0),
            "forces_L": torch.full((D, L, 3), 2.0),
        }


class _ForbiddenSampler:
    """A sampler that raises an error if used, to ensure that generative sampling is not performed during property prediction."""

    def sample_diffusion_like_af3(self, *args, **kwargs):
        raise AssertionError("The generative sampler must not be used.")


def _property_model(zatom2_cls):
    """Create a Zatom2 model with mocked components for testing property prediction."""
    model = zatom2_cls.__new__(zatom2_cls)
    nn.Module.__init__(model)
    model.token_initializer = _PropertyTokenInitializer()
    model.diffusion_module = _PropertyDiffusionModule()
    model.inference_sampler = _ForbiddenSampler()
    model.use_classifier_free_guidance = False
    return model


def _property_input():
    """Create a dummy input dictionary for testing property prediction."""
    return {
        "coord_atom_lvl_to_be_noised": torch.tensor(
            [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]]
        ),
        "f": {
            "restype": torch.zeros(2, dtype=torch.long),
            "atom_to_token_map": torch.tensor([0, 1]),
            "batch_token_lvl": torch.tensor([0, 0]),
            "batch_atom_lvl": torch.tensor([0, 0]),
            "domain_id_B": torch.tensor([0]),
            "total_charge_B": torch.tensor([-1.0]),
            "spin_multiplicity_B": torch.tensor([2.0]),
            "charge_valid_mask_B": torch.tensor([True]),
            "spin_valid_mask_B": torch.tensor([True]),
            "ref_element": torch.tensor([6, 8]),
            "ref_atom_name_chars": torch.ones(2, 4, dtype=torch.long),
            "token_bonds": torch.ones(2, 2),
            "mean_force_norm": torch.ones(2, 1),
            "has_mean_force_norm": torch.ones(2, 1),
        },
    }


def test_predict_properties_is_one_clean_pass_without_force_norm_leakage(monkeypatch):
    """Test that the predict_properties method performs a single clean pass and does not leak force norm information."""
    model = _property_model(_load_zatom2(monkeypatch))
    input = _property_input()

    result = model.predict_properties(input, return_embeddings=True)

    assert len(model.diffusion_module.calls) == 1
    call = model.diffusion_module.calls[0]
    torch.testing.assert_close(call["t"], torch.ones(1, 1))
    assert call["n_recycle"] == 1
    assert call["return_property_embeddings"] is True
    initialized_f = model.token_initializer.calls[0]
    torch.testing.assert_close(initialized_f["mean_force_norm"], torch.zeros(2, 1))
    torch.testing.assert_close(initialized_f["has_mean_force_norm"], torch.zeros(2, 1))
    torch.testing.assert_close(initialized_f["total_charge"], torch.full((2, 1), -1.0))
    torch.testing.assert_close(
        initialized_f["spin_multiplicity"], torch.full((2, 1), 2.0)
    )
    torch.testing.assert_close(initialized_f["has_total_charge"], torch.ones(2, 1))
    torch.testing.assert_close(initialized_f["has_spin_multiplicity"], torch.ones(2, 1))
    torch.testing.assert_close(initialized_f["ref_element"], torch.tensor([6, 8]))
    torch.testing.assert_close(result["energy_B"], torch.tensor([7.0]))
    torch.testing.assert_close(result["forces_L"], torch.full((2, 3), 2.0))
    assert {"Q_sys", "Q_I", "Q_L"}.issubset(result)
    assert "atom_type_logits_I" not in result


def test_eval_forward_routes_property_prediction_without_sampling(monkeypatch):
    """The EMA-compatible forward route delegates MLIP inference to the property API."""
    model = _property_model(_load_zatom2(monkeypatch))
    model.eval()

    result = model.forward(
        _property_input(),
        property_prediction_atom_types="known",
    )

    assert len(model.diffusion_module.calls) == 1
    torch.testing.assert_close(model.diffusion_module.calls[0]["t"], torch.ones(1, 1))
    torch.testing.assert_close(result["energy_B"], torch.tensor([7.0]))


def test_predict_properties_masked_mode_removes_atom_type_leakage(monkeypatch):
    """Test that the predict_properties method removes atom type leakage in masked mode."""
    model = _property_model(_load_zatom2(monkeypatch))
    input = _property_input()
    original_elements = input["f"]["ref_element"].clone()

    result = model.predict_properties(input, atom_types="masked")

    initialized_f = model.token_initializer.calls[0]
    torch.testing.assert_close(
        initialized_f["ref_element"], torch.zeros_like(original_elements)
    )
    torch.testing.assert_close(
        initialized_f["ref_atom_name_chars"], torch.zeros(2, 4, dtype=torch.long)
    )
    torch.testing.assert_close(initialized_f["token_bonds"], torch.zeros(2, 2))
    assert initialized_f["atom_type_generation_mask"].all()
    torch.testing.assert_close(input["f"]["ref_element"], original_elements)
    assert result["atom_type_logits_I"].shape == (2, 128)


def test_predict_properties_requires_omol_charge_and_spin(monkeypatch):
    """Test that the predict_properties method raises an error if total charge and spin information is missing."""
    model = _property_model(_load_zatom2(monkeypatch))
    input = _property_input()
    del input["f"]["spin_multiplicity_B"]
    del input["f"]["spin_valid_mask_B"]

    with pytest.raises(ValueError, match="requires valid total charge and spin"):
        model.predict_properties(input)


def test_predict_properties_known_mode_rejects_masked_elements(monkeypatch):
    """Test that the predict_properties method raises an error if known mode is used with masked elements."""
    model = _property_model(_load_zatom2(monkeypatch))
    input = _property_input()
    input["f"]["ref_element"] = torch.zeros(2, dtype=torch.long)

    with pytest.raises(ValueError, match="missing or masked element"):
        model.predict_properties(input)


@pytest.mark.parametrize(
    ("metadata_key", "invalid_value"),
    [
        ("total_charge_B", 0.5),
        ("spin_multiplicity_B", 1.5),
    ],
)
def test_predict_properties_rejects_noninteger_omol_metadata(
    monkeypatch,
    metadata_key,
    invalid_value,
):
    """Test that the predict_properties method raises an error if total charge or spin metadata is non-integer."""
    model = _property_model(_load_zatom2(monkeypatch))
    input = _property_input()
    input["f"][metadata_key] = torch.tensor([invalid_value])

    with pytest.raises(ValueError, match="requires valid total charge and spin"):
        model.predict_properties(input)


class _CapturedPropertySampler:
    """A sampler that captures the sample returned by the diffusion module for testing property prediction."""

    def __init__(self, sample):
        self.sample = sample
        self.calls = 0

    def sample_diffusion_like_af3(self, **kwargs):
        """Sample method that increments the call count and returns the captured sample."""
        self.calls += 1
        return dict(self.sample)


@pytest.mark.parametrize(
    ("generation_mask", "expected_atom_type_mode"),
    [
        ([False, False], "known"),
        ([True, True], "masked"),
        ([False, True], None),
    ],
)
def test_generation_reconstructs_properties_only_for_unambiguous_type_masks(
    monkeypatch,
    generation_mask,
    expected_atom_type_mode,
):
    """Test that the generation method reconstructs properties only for unambiguous atom type masks."""
    model = _property_model(_load_zatom2(monkeypatch))
    model.eval()
    input = _property_input()
    input["f"]["has_mean_force_norm"].zero_()
    input["f"]["atom_type_generation_mask"] = torch.tensor(generation_mask)
    captured_sample = {
        "X_L": input["coord_atom_lvl_to_be_noised"],
        "normalized_energy_B": torch.full((1, 1), 0.25),
        "normalized_forces_L": torch.full((1, 2, 3), -0.5),
        "atom_type_logits_I": torch.zeros(1, 2, 128),
    }
    sampler = _CapturedPropertySampler(captured_sample)
    model.inference_sampler = sampler

    result = model(
        input,
        coord_atom_lvl_to_be_noised=input["coord_atom_lvl_to_be_noised"],
    )

    assert sampler.calls == 1
    torch.testing.assert_close(
        result["normalized_energy_B"], captured_sample["normalized_energy_B"]
    )
    torch.testing.assert_close(
        result["normalized_forces_L"], captured_sample["normalized_forces_L"]
    )
    if expected_atom_type_mode is None:
        assert model.diffusion_module.reconstruction_atom_types == []
        assert "energy_B" not in result
        assert "forces_L" not in result
    else:
        assert model.diffusion_module.reconstruction_atom_types == [
            expected_atom_type_mode
        ]
        torch.testing.assert_close(result["energy_B"], torch.full((1, 1), 7.0))
        torch.testing.assert_close(result["forces_L"], torch.full((1, 2, 3), 2.0))


def test_generation_with_disabled_property_heads_keeps_legacy_sample(monkeypatch):
    """Test that when property prediction heads are disabled, the generation method keeps the legacy sample without reconstruction."""
    model = _property_model(_load_zatom2(monkeypatch))
    model.eval()
    model.diffusion_module.atomistic_property_prediction_enabled = False
    input = _property_input()
    input["f"]["atom_type_generation_mask"] = torch.zeros(2, dtype=torch.bool)
    captured_sample = {"X_L": input["coord_atom_lvl_to_be_noised"]}
    model.inference_sampler = _CapturedPropertySampler(captured_sample)

    result = model(
        input,
        coord_atom_lvl_to_be_noised=input["coord_atom_lvl_to_be_noised"],
    )

    assert result == captured_sample
    assert model.diffusion_module.reconstruction_atom_types == []


def test_generation_does_not_reconstruct_omol_properties_without_spin(monkeypatch):
    """Test that the generation method does not reconstruct molecular properties if spin information is missing."""
    model = _property_model(_load_zatom2(monkeypatch))
    model.eval()
    input = _property_input()
    input["f"]["has_mean_force_norm"].zero_()
    input["f"]["atom_type_generation_mask"] = torch.zeros(2, dtype=torch.bool)
    del input["f"]["spin_multiplicity_B"]
    del input["f"]["spin_valid_mask_B"]
    captured_sample = {
        "X_L": input["coord_atom_lvl_to_be_noised"],
        "normalized_energy_B": torch.full((1, 1), 0.25),
        "normalized_forces_L": torch.full((1, 2, 3), -0.5),
    }
    model.inference_sampler = _CapturedPropertySampler(captured_sample)

    result = model(
        input,
        coord_atom_lvl_to_be_noised=input["coord_atom_lvl_to_be_noised"],
    )

    assert result == captured_sample
    assert model.diffusion_module.reconstruction_atom_types == []


def test_generation_does_not_publish_force_norm_conditioned_properties(monkeypatch):
    """Test that the generation method does not publish properties conditioned on mean force norm information."""
    model = _property_model(_load_zatom2(monkeypatch))
    model.eval()
    input = _property_input()
    input["f"]["atom_type_generation_mask"] = torch.zeros(2, dtype=torch.bool)
    captured_sample = {
        "X_L": input["coord_atom_lvl_to_be_noised"],
        "normalized_energy_B": torch.full((1, 1), 0.25),
        "normalized_forces_L": torch.full((1, 2, 3), -0.5),
    }
    model.inference_sampler = _CapturedPropertySampler(captured_sample)

    result = model(
        input,
        coord_atom_lvl_to_be_noised=input["coord_atom_lvl_to_be_noised"],
    )

    assert result == captured_sample
    assert model.diffusion_module.reconstruction_atom_types == []
