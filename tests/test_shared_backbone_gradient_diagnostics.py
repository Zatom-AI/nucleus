from types import SimpleNamespace

import pytest
import torch
from lightning import Fabric
from torch import nn

from nucleus.metrics.losses import (
    FLOW_GRADIENT_DIAGNOSTIC_NAMES,
    PROPERTY_GRADIENT_DIAGNOSTIC_NAME,
    collate_shared_backbone_gradient_metrics,
    compute_shared_backbone_gradient_cosines,
    pop_shared_backbone_gradient_diagnostics,
)
from nucleus.trainers.zatom.zatom import (
    _select_last_shared_backbone_block_parameters,
)


def test_shared_gradient_cosines_detect_alignment_conflict_and_orthogonality():
    parameter = nn.Parameter(torch.tensor([1.0, 2.0]))
    diagnostics = {
        PROPERTY_GRADIENT_DIAGNOSTIC_NAME: parameter[0] + parameter[1],
        FLOW_GRADIENT_DIAGNOSTIC_NAMES[0]: 2 * parameter[0] + 2 * parameter[1],
        FLOW_GRADIENT_DIAGNOSTIC_NAMES[1]: -parameter[0] - parameter[1],
        FLOW_GRADIENT_DIAGNOSTIC_NAMES[2]: parameter[0] - parameter[1],
    }

    metrics = compute_shared_backbone_gradient_cosines(diagnostics, [parameter])

    property_name = PROPERTY_GRADIENT_DIAGNOSTIC_NAME
    low_name, mid_name, high_name = FLOW_GRADIENT_DIAGNOSTIC_NAMES
    torch.testing.assert_close(
        metrics[f"l2_norm/{property_name}"], torch.sqrt(torch.tensor(2.0))
    )
    torch.testing.assert_close(
        metrics[f"cosine/{low_name}_vs_{property_name}"], torch.tensor(1.0)
    )
    torch.testing.assert_close(
        metrics[f"cosine/{mid_name}_vs_{property_name}"], torch.tensor(-1.0)
    )
    torch.testing.assert_close(
        metrics[f"cosine/{high_name}_vs_{property_name}"], torch.tensor(0.0)
    )
    assert metrics[f"conflict/{low_name}_vs_{property_name}"] == 0
    assert metrics[f"conflict/{mid_name}_vs_{property_name}"] == 1
    assert metrics[f"conflict/{high_name}_vs_{property_name}"] == 0

    # Diagnostic autograd calls must not populate .grad or consume the graph.
    assert parameter.grad is None
    sum(diagnostics.values()).backward()
    assert parameter.grad is not None


def test_shared_gradient_cosine_is_nan_for_an_inactive_task():
    parameter = nn.Parameter(torch.tensor([1.0, 2.0]))
    zero_flow_loss = (parameter * 0.0).sum()
    diagnostics = {
        PROPERTY_GRADIENT_DIAGNOSTIC_NAME: parameter.sum(),
        FLOW_GRADIENT_DIAGNOSTIC_NAMES[0]: zero_flow_loss,
    }

    metrics = compute_shared_backbone_gradient_cosines(diagnostics, [parameter])

    comparison = (
        f"{FLOW_GRADIENT_DIAGNOSTIC_NAMES[0]}_vs_"
        f"{PROPERTY_GRADIENT_DIAGNOSTIC_NAME}"
    )
    assert torch.isnan(metrics[f"cosine/{comparison}"])
    assert torch.isnan(metrics[f"conflict/{comparison}"])
    assert metrics[f"l2_norm/{FLOW_GRADIENT_DIAGNOSTIC_NAMES[0]}"] == 0


def test_shared_gradient_cosines_support_backward_callback_and_restore_gradients():
    parameter = nn.Parameter(torch.tensor([1.0, 2.0]))
    parameter.grad = torch.tensor([3.0, 4.0])
    original_gradient = parameter.grad
    diagnostics = {
        PROPERTY_GRADIENT_DIAGNOSTIC_NAME: parameter.sum(),
        FLOW_GRADIENT_DIAGNOSTIC_NAMES[0]: 2 * parameter.sum(),
    }
    backward_calls = 0

    def backward(loss, **kwargs):
        nonlocal backward_calls
        backward_calls += 1
        loss.backward(**kwargs)

    metrics = compute_shared_backbone_gradient_cosines(
        diagnostics,
        [parameter],
        backward=backward,
    )

    comparison = (
        f"{FLOW_GRADIENT_DIAGNOSTIC_NAMES[0]}_vs_"
        f"{PROPERTY_GRADIENT_DIAGNOSTIC_NAME}"
    )
    assert backward_calls == 2
    torch.testing.assert_close(metrics[f"cosine/{comparison}"], torch.tensor(1.0))
    assert parameter.grad is original_gradient
    torch.testing.assert_close(parameter.grad, torch.tensor([3.0, 4.0]))


def test_shared_gradient_cosines_use_fabric_backward_with_mixed_precision():
    fabric = Fabric(accelerator="cpu", devices=1, precision="bf16-mixed")
    model = fabric.setup(nn.Linear(2, 1))
    prediction = model(torch.ones(1, 2))
    diagnostics = {
        PROPERTY_GRADIENT_DIAGNOSTIC_NAME: prediction.sum(),
        FLOW_GRADIENT_DIAGNOSTIC_NAMES[0]: 2 * prediction.sum(),
    }

    metrics = compute_shared_backbone_gradient_cosines(
        diagnostics,
        tuple(model.parameters()),
        backward=fabric.backward,
    )
    metrics = collate_shared_backbone_gradient_metrics(
        metrics,
        lambda tensor: fabric.all_reduce(tensor, reduce_op="sum"),
        world_size=fabric.world_size,
    )

    comparison = (
        f"{FLOW_GRADIENT_DIAGNOSTIC_NAMES[0]}_vs_"
        f"{PROPERTY_GRADIENT_DIAGNOSTIC_NAME}"
    )
    torch.testing.assert_close(metrics[f"cosine/{comparison}"], torch.tensor(1.0))


def test_collates_shared_gradient_metrics_across_ranks():
    comparison = "flow_mse_low_noise_vs_atomistic_property"
    local_metrics = {
        f"cosine/{comparison}": torch.tensor(-0.2),
        f"conflict/{comparison}": torch.tensor(1.0),
        "l2_norm/atomistic_property": torch.tensor(2.0),
    }
    other_rank_metrics = {
        f"cosine/{comparison}": torch.tensor(0.4),
        f"conflict/{comparison}": torch.tensor(0.0),
        "l2_norm/atomistic_property": torch.tensor(4.0),
    }

    def all_reduce(local_statistics):
        names = tuple(sorted(other_rank_metrics))
        other_values = torch.stack([other_rank_metrics[name] for name in names])
        other_finite = torch.isfinite(other_values)
        other_values = torch.where(
            other_finite, other_values, torch.zeros_like(other_values)
        )
        other_statistics = torch.stack(
            (other_values, other_values.square(), other_finite.float())
        )
        return local_statistics + other_statistics

    metrics = collate_shared_backbone_gradient_metrics(
        local_metrics,
        all_reduce,
        world_size=2,
    )

    torch.testing.assert_close(metrics[f"cosine/{comparison}"], torch.tensor(0.1))
    torch.testing.assert_close(metrics[f"conflict/{comparison}"], torch.tensor(0.5))
    torch.testing.assert_close(
        metrics[f"cosine_rank_std/{comparison}"], torch.tensor(0.3)
    )
    torch.testing.assert_close(
        metrics[f"cosine_valid_rank_fraction/{comparison}"], torch.tensor(1.0)
    )
    torch.testing.assert_close(metrics["l2_norm/atomistic_property"], torch.tensor(3.0))
    torch.testing.assert_close(
        metrics["l2_norm_rank_std/atomistic_property"], torch.tensor(1.0)
    )


def test_collation_ignores_nonfinite_rank_metrics():
    comparison = "flow_mse_low_noise_vs_atomistic_property"
    local_metrics = {
        f"cosine/{comparison}": torch.tensor(float("nan")),
        f"conflict/{comparison}": torch.tensor(float("nan")),
    }

    def all_reduce(local_statistics):
        other_values = torch.tensor([0.0, 0.4])
        other_statistics = torch.stack(
            (other_values, other_values.square(), torch.ones_like(other_values))
        )
        return local_statistics + other_statistics

    metrics = collate_shared_backbone_gradient_metrics(
        local_metrics,
        all_reduce,
        world_size=2,
    )

    torch.testing.assert_close(metrics[f"cosine/{comparison}"], torch.tensor(0.4))
    torch.testing.assert_close(metrics[f"conflict/{comparison}"], torch.tensor(0.0))
    torch.testing.assert_close(
        metrics[f"cosine_rank_std/{comparison}"], torch.tensor(0.0)
    )
    torch.testing.assert_close(
        metrics[f"cosine_valid_rank_fraction/{comparison}"], torch.tensor(0.5)
    )


def test_pop_shared_gradient_diagnostics_collects_and_clears_modules():
    first = SimpleNamespace(
        shared_backbone_gradient_diagnostics={"first": torch.tensor(1.0)}
    )
    second = SimpleNamespace(
        shared_backbone_gradient_diagnostics={"second": torch.tensor(2.0)}
    )
    container = SimpleNamespace(to_compute=[first, second])

    diagnostics = pop_shared_backbone_gradient_diagnostics(container)

    assert diagnostics.keys() == {"first", "second"}
    assert first.shared_backbone_gradient_diagnostics == {}
    assert second.shared_backbone_gradient_diagnostics == {}


def test_selects_only_last_shared_backbone_blocks():
    class Stage(nn.Module):
        def __init__(self, n_blocks):
            super().__init__()
            self.blocks = nn.ModuleList(nn.Linear(2, 2) for _ in range(n_blocks))

    class Backbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.token_trunk = Stage(2)
            self.atom_decoder = Stage(3)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.diffusion_module = nn.Module()
            self.diffusion_module.backbone = Backbone()

    model = Model()
    names, parameters = _select_last_shared_backbone_block_parameters(model)

    assert names
    assert len(names) == len(parameters) == 4
    assert all(
        "token_trunk.blocks.1" in name or "atom_decoder.blocks.2" in name
        for name in names
    )


def test_select_shared_backbone_blocks_requires_expected_model_structure():
    with pytest.raises(RuntimeError, match="could not find trainable parameters"):
        _select_last_shared_backbone_block_parameters(nn.Linear(2, 2))
