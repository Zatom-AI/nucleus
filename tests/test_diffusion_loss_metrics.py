import pytest
import torch

from nucleus.metrics.losses import (
    DiffusionLoss,
    SequenceLoss,
    _split_losses_by_t,
    smoothed_lddt_loss,
)


def test_split_losses_by_t_flattens_diffusion_and_batch_axes():
    """Test that _split_losses_by_t flattens the diffusion and batch axes of the loss tensor."""
    t = torch.tensor(
        [
            [0.9, 0.1],
            [0.8, 0.2],
            [0.7, 0.3],
            [0.6, 0.4],
        ]
    )
    l_global = torch.arange(8).reshape(4, 2)

    low_t, high_t = _split_losses_by_t(l_global, t)

    assert torch.equal(low_t, torch.tensor([1, 3, 5, 7]))
    assert torch.equal(high_t, torch.tensor([6, 4, 2, 0]))


def test_split_losses_by_t_duplicates_singleton_for_finite_metrics():
    """Test that _split_losses_by_t duplicates the loss for a singleton batch for finite metrics."""
    l_global = torch.tensor([[2.5]])
    t = torch.tensor([[0.4]])

    low_t, high_t = _split_losses_by_t(l_global, t)

    assert torch.equal(low_t, torch.tensor([2.5]))
    assert torch.equal(high_t, torch.tensor([2.5]))


def test_smoothed_lddt_applies_per_example_batch_loss_weights():
    """Test that smoothed_lddt_loss applies per-example batch loss weights correctly."""
    ground_truth = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]] * 2])
    prediction = ground_truth.clone()
    prediction[:, 3, 0] = 11.0
    atom_to_token = torch.arange(4)
    batch_atom = torch.tensor([0, 0, 1, 1])
    false_tokens = torch.zeros(4, dtype=torch.bool)

    args = (
        prediction,
        ground_truth,
        torch.ones(1, 4, dtype=torch.bool),
        false_tokens,
        false_tokens,
        false_tokens,
        atom_to_token,
        batch_atom,
    )
    unweighted, _ = smoothed_lddt_loss(*args)
    unit_weighted, _ = smoothed_lddt_loss(*args, batch_loss_weight=torch.ones(1, 2))
    inaccurate_example_upweighted, _ = smoothed_lddt_loss(
        *args, batch_loss_weight=torch.tensor([[1.0, 3.0]])
    )
    accurate_example_upweighted, _ = smoothed_lddt_loss(
        *args, batch_loss_weight=torch.tensor([[3.0, 1.0]])
    )

    def single_example_loss(atom_slice):
        """Compute the loss for a single example given an atom slice."""
        loss, _ = smoothed_lddt_loss(
            prediction[:, atom_slice],
            ground_truth[:, atom_slice],
            torch.ones(1, 2, dtype=torch.bool),
            false_tokens[:2],
            false_tokens[:2],
            false_tokens[:2],
            torch.arange(2),
            torch.zeros(2, dtype=torch.long),
        )
        return loss

    accurate_loss = single_example_loss(slice(0, 2))
    inaccurate_loss = single_example_loss(slice(2, 4))

    torch.testing.assert_close(unweighted, unit_weighted)
    torch.testing.assert_close(unit_weighted, (accurate_loss + inaccurate_loss) / 2)
    torch.testing.assert_close(
        inaccurate_example_upweighted,
        (accurate_loss + 3 * inaccurate_loss) / 2,
    )
    torch.testing.assert_close(
        accurate_example_upweighted,
        (3 * accurate_loss + inaccurate_loss) / 2,
    )
    assert inaccurate_example_upweighted > accurate_example_upweighted > unweighted


def _assert_empty_pair_lddt_is_zero(device: torch.device, *, repeats: int = 1):
    """Check the all-fixed lDDT path without invoking an empty scatter."""
    n_diffusion, n_atoms, batch_size = 2, 4, 2
    ground_truth = torch.tensor(
        [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]] * batch_size],
        device=device,
    ).expand(n_diffusion, -1, -1)
    prediction = ground_truth.clone().requires_grad_()
    atom_to_token = torch.arange(n_atoms, device=device)
    batch_atom = torch.arange(batch_size, device=device).repeat_interleave(2)
    false_tokens = torch.zeros(n_atoms, dtype=torch.bool, device=device)
    coordinate_mask = torch.zeros(n_diffusion, n_atoms, dtype=torch.bool, device=device)
    batch_loss_weight = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=device)

    for _ in range(repeats):
        loss, _ = smoothed_lddt_loss(
            prediction,
            ground_truth,
            coordinate_mask,
            false_tokens,
            false_tokens,
            false_tokens,
            atom_to_token,
            batch_atom,
            batch_loss_weight=batch_loss_weight,
        )
        assert torch.equal(loss, torch.zeros_like(loss))

    loss.sum().backward()
    assert prediction.grad is not None
    assert torch.equal(prediction.grad, torch.zeros_like(prediction.grad))


def test_smoothed_lddt_empty_pair_batch_is_differentiable_zero():
    """All-fixed batches have no coordinate target and contribute zero lDDT."""
    _assert_empty_pair_lddt_is_zero(torch.device("cpu"))


def test_smoothed_lddt_zeroes_batch_elements_without_pairs():
    """An all-fixed system contributes zero beside a valid generative system."""
    ground_truth = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]] * 2])
    prediction = ground_truth.clone()
    prediction[:, 1, 0] = 11.0
    false_tokens = torch.zeros(4, dtype=torch.bool)

    loss, _ = smoothed_lddt_loss(
        prediction,
        ground_truth,
        torch.tensor([[True, True, False, False]]),
        false_tokens,
        false_tokens,
        false_tokens,
        torch.arange(4),
        torch.tensor([0, 0, 1, 1]),
        batch_loss_weight=torch.tensor([[1.0, 1e6]]),
    )
    valid_system_loss, _ = smoothed_lddt_loss(
        prediction[:, :2],
        ground_truth[:, :2],
        torch.ones(1, 2, dtype=torch.bool),
        false_tokens[:2],
        false_tokens[:2],
        false_tokens[:2],
        torch.arange(2),
        torch.zeros(2, dtype=torch.long),
    )

    torch.testing.assert_close(loss, valid_system_loss / 2)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is unavailable")
def test_smoothed_lddt_empty_pair_batch_is_stable_on_mps():
    """Empty MPS scatter inputs must not produce intermittent garbage losses."""
    _assert_empty_pair_lddt_is_zero(torch.device("mps"), repeats=100)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_smoothed_lddt_diagnostics_remain_on_the_input_device():
    """LDDT diagnostics must not synchronize each rank by moving to CPU early."""
    device = torch.device("cuda")
    ground_truth = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]], device=device)
    false_tokens = torch.zeros(2, dtype=torch.bool, device=device)

    _, diagnostics = smoothed_lddt_loss(
        ground_truth.clone(),
        ground_truth,
        torch.ones(1, 2, dtype=torch.bool, device=device),
        false_tokens,
        false_tokens,
        false_tokens,
        torch.arange(2, device=device),
        torch.zeros(2, dtype=torch.long, device=device),
        return_extras=True,
    )

    assert diagnostics
    assert all(value.device == ground_truth.device for value in diagnostics.values())


def _diffusion_loss_inputs(*, has_unindexed_tokens: bool):
    """Build a two-atom, one-token batch for DiffusionLoss.forward."""
    n_atoms, n_tokens = 2, 1
    atom_to_token = torch.zeros(n_atoms, dtype=torch.long)
    false_atoms = torch.zeros(n_atoms, dtype=torch.bool)
    network_input = {
        "t": torch.full((1, 1), 0.5),
        "t_L": torch.full((1, n_atoms), 0.5),
        "f": {
            "diffusion_batch_size": 1,
            "batch_size": 1,
            "batch_atom_lvl": torch.zeros(n_atoms, dtype=torch.long),
            "atom_to_token_map": atom_to_token,
            "is_polar": torch.zeros(n_tokens, dtype=torch.bool),
            "is_ligand": torch.zeros(n_tokens, dtype=torch.bool),
            "is_virtual": false_atoms,
            "is_sidechain": false_atoms,
        },
    }
    network_output = {"X_L": torch.zeros(1, n_atoms, 3)}
    loss_input = {
        "crd_mask_L": torch.ones(n_atoms, dtype=torch.bool),
        "X_gt_L_in_input_frame": torch.ones(1, n_atoms, 3),
        "atom_to_token_map_per_batch_element": atom_to_token,
        "is_original_unindexed_token_batch_offset": torch.zeros(
            n_atoms, dtype=torch.long
        ),
        "is_original_unindexed_token": torch.tensor([has_unindexed_tokens]),
    }
    return network_input, network_output, loss_input


def _unindexed_flow_loss_inputs(*, include_fixed_system: bool):
    """Build an active unindexed system, optionally packed with fixed MLIP."""
    batch_size = 2 if include_fixed_system else 1
    atoms_per_system = 2
    batch_atom = torch.arange(batch_size).repeat_interleave(atoms_per_system)
    n_atoms = batch_atom.numel()
    time = torch.tensor([[0.4, 0.7]])[:, :batch_size]
    velocity = torch.tensor([[[-0.5, -0.2, 0.1], [0.3, 0.6, 0.8]]]).repeat(
        1, batch_size, 1
    )
    velocity.requires_grad_()
    fixed_atom_mask = torch.zeros(n_atoms, dtype=torch.bool)
    if include_fixed_system:
        fixed_atom_mask[atoms_per_system:] = True

    network_input = {
        "t": time,
        "t_L": time[:, batch_atom],
        "X_noisy_L": torch.zeros(1, n_atoms, 3),
        "f": {
            "diffusion_batch_size": 1,
            "batch_size": batch_size,
            "batch_atom_lvl": batch_atom,
            "atom_to_token_map": batch_atom,
            "is_polar": torch.zeros(batch_size, dtype=torch.bool),
            "is_ligand": torch.zeros(batch_size, dtype=torch.bool),
            "is_virtual": torch.zeros(n_atoms, dtype=torch.bool),
            "is_sidechain": torch.zeros(n_atoms, dtype=torch.bool),
            "is_motif_atom_with_fixed_coord": fixed_atom_mask,
        },
    }
    network_output = {
        "X_L": torch.zeros(1, n_atoms, 3),
        "V_L": velocity,
    }
    loss_input = {
        "crd_mask_L": torch.ones(n_atoms, dtype=torch.bool),
        "X_gt_L_in_input_frame": torch.tensor(
            [[[0.2, 0.4, 0.6], [0.8, 1.1, 1.4]]]
        ).repeat(1, batch_size, 1),
        "atom_to_token_map_per_batch_element": torch.zeros(n_atoms, dtype=torch.long),
        "is_original_unindexed_token_batch_offset": batch_atom,
        "is_original_unindexed_token": torch.tensor(
            [True] + ([False] if include_fixed_system else [])
        ),
    }
    return network_input, network_output, loss_input


def test_unindexed_flow_loss_zeroes_fixed_system_in_packed_batch():
    """An inactive packed system must not create 0/0 loss or gradients."""
    loss_fn = DiffusionLoss(
        weight=1.0,
        sigma_data=16.0,
        lddt_weight=0.0,
        lp_weight=0.0,
        objective="flow_matching",
    )
    packed_input, packed_output, packed_loss_input = _unindexed_flow_loss_inputs(
        include_fixed_system=True
    )
    packed_loss, packed_metrics = loss_fn(
        packed_input, packed_output, packed_loss_input
    )
    packed_loss.backward()

    single_input, single_output, single_loss_input = _unindexed_flow_loss_inputs(
        include_fixed_system=False
    )
    single_loss, single_metrics = loss_fn(
        single_input, single_output, single_loss_input
    )
    single_loss.backward()

    assert torch.isfinite(packed_loss)
    assert torch.isfinite(packed_output["V_L"].grad).all()
    torch.testing.assert_close(packed_loss, single_loss / 2)
    torch.testing.assert_close(
        packed_metrics["mse_loss_mean"], single_metrics["mse_loss_mean"] / 2
    )
    torch.testing.assert_close(
        packed_output["V_L"].grad[:, :2], single_output["V_L"].grad / 2
    )
    torch.testing.assert_close(
        packed_output["V_L"].grad[:, 2:],
        torch.zeros_like(packed_output["V_L"].grad[:, 2:]),
    )


def test_loss_keys_do_not_depend_on_the_batch_contents():
    """Every DDP rank must report the same loss keys.

    Keys that appear only for batches with unindexed tokens desynchronize
    epoch-end metric logging, because ranks then enter different collectives.
    """
    loss = DiffusionLoss(weight=1.0, sigma_data=16.0, lddt_weight=0.0, lp_weight=1.0)

    _, with_motif = loss(*_diffusion_loss_inputs(has_unindexed_tokens=True))
    _, without_motif = loss(*_diffusion_loss_inputs(has_unindexed_tokens=False))

    assert set(with_motif) == set(without_motif)
    assert {"lp_norm", "lp_norm_unindexed_diffused"} <= set(without_motif)
    assert without_motif["lp_norm"] == 0.0
    assert without_motif["lp_norm_unindexed_diffused"] == 0.0


def _token_ce_with_cap_mode(cap_mode: str):
    """Return a two-token CE branch with one extreme outlier."""
    loss = SequenceLoss(token_ce_cap_mode=cap_mode)
    loss.loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    logits = torch.tensor(
        [[[20.0, 0.0], [0.0, 0.0]]],
        requires_grad=True,
    )
    effective, metrics = loss._token_ce_loss(
        logits=logits,
        indices=logits.detach().argmax(dim=-1),
        gt=torch.tensor([1, 0]),
        target_mask=torch.ones(2, dtype=torch.bool),
        valid_t_expanded=torch.ones(1, 2, dtype=torch.bool),
        t_I=torch.tensor([[0.1, 0.2]]),
        metric_prefix="sequence",
        loss_cap=5.0,
    )
    effective.backward()
    return effective.detach(), metrics, logits.grad


def test_mean_hard_token_ce_cap_zeros_the_whole_branch_gradient():
    """The historical post-mean clamp cannot recover after crossing its cap."""
    effective, metrics, gradient = _token_ce_with_cap_mode("mean_hard")

    torch.testing.assert_close(effective, torch.tensor(5.0))
    assert metrics["sequence_loss"] > effective
    assert metrics["sequence_loss_above_cap"] == 1
    torch.testing.assert_close(gradient, torch.zeros_like(gradient))


def test_per_token_ce_cap_keeps_non_outlier_gradients():
    """Per-token clipping suppresses the outlier without freezing other tokens."""
    effective, metrics, gradient = _token_ce_with_cap_mode("per_token")

    assert effective < metrics["sequence_loss"]
    assert metrics["sequence_token_loss_above_cap_fraction"] == 0.5
    torch.testing.assert_close(gradient[:, :1], torch.zeros_like(gradient[:, :1]))
    assert torch.count_nonzero(gradient[:, 1:]) > 0


def test_disabling_token_ce_cap_preserves_all_gradients():
    """The uncapped control optimizes the raw mean CE."""
    effective, metrics, gradient = _token_ce_with_cap_mode("none")

    torch.testing.assert_close(effective, metrics["sequence_loss"])
    assert torch.count_nonzero(gradient) > 0


def _packed_sequence_loss_inputs(device="cpu"):
    """Build three systems with unequal tokens, views, and active targets."""
    batch_token = torch.tensor([0, 1, 1, 1, 2, 2], device=device)
    time = torch.tensor(
        [
            [0.2, 0.3, 1.1],
            [1.2, 0.4, 0.5],
        ],
        device=device,
    )
    time_token = time[:, batch_token]
    n_diffusion, n_tokens = time_token.shape

    sequence_logits = torch.linspace(
        -2.0, 2.0, n_diffusion * n_tokens * 3, device=device
    ).reshape(n_diffusion, n_tokens, 3)
    sequence_logits[0, 0] = torch.tensor([12.0, -12.0, 0.0], device=device)
    sequence_logits.requires_grad_()
    atom_type_logits = torch.linspace(
        2.5, -2.5, n_diffusion * n_tokens * 4, device=device
    ).reshape(n_diffusion, n_tokens, 4)
    atom_type_logits[1, 4] = torch.tensor([-10.0, 10.0, -10.0, -10.0], device=device)
    atom_type_logits.requires_grad_()

    network_input = {
        "t": time,
        "t_I": time_token,
        "f": {
            "diffusion_batch_size": n_diffusion,
            "batch_size": 3,
            "batch_token_lvl": batch_token,
        },
    }
    network_output = {
        "sequence_logits_I": sequence_logits,
        "sequence_indices_I": sequence_logits.detach().argmax(dim=-1),
        "atom_type_logits_I": atom_type_logits,
        "atom_type_indices_I": atom_type_logits.detach().argmax(dim=-1),
    }
    loss_input = {
        "seq_token_lvl": torch.tensor([1, 0, 2, 1, 0, 2], device=device),
        # System 2 has no sequence target.
        "sequence_valid_mask": torch.tensor(
            [True, True, True, True, False, False], device=device
        ),
        "atom_type_gt_I": torch.tensor([0, 1, 2, 3, 0, 1], device=device),
        # System 1 has no atom-type target.
        "atom_type_valid_mask": torch.tensor(
            [True, False, False, False, True, True], device=device
        ),
    }
    return network_input, network_output, loss_input


def _slice_sequence_loss_system(network_input, network_output, loss_input, system):
    """Extract one system as a standalone sparse batch with leaf logits."""
    token_mask = network_input["f"]["batch_token_lvl"] == system
    sequence_logits = (
        network_output["sequence_logits_I"][:, token_mask]
        .detach()
        .clone()
        .requires_grad_()
    )
    atom_type_logits = (
        network_output["atom_type_logits_I"][:, token_mask]
        .detach()
        .clone()
        .requires_grad_()
    )
    single_input = {
        "t": network_input["t"][:, system : system + 1],
        "t_I": network_input["t_I"][:, token_mask],
        "f": {
            "diffusion_batch_size": network_input["f"]["diffusion_batch_size"],
            "batch_size": 1,
            "batch_token_lvl": torch.zeros_like(
                network_input["f"]["batch_token_lvl"][token_mask]
            ),
        },
    }
    single_output = {
        "sequence_logits_I": sequence_logits,
        "sequence_indices_I": sequence_logits.detach().argmax(dim=-1),
        "atom_type_logits_I": atom_type_logits,
        "atom_type_indices_I": atom_type_logits.detach().argmax(dim=-1),
    }
    single_loss_input = {key: value[token_mask] for key, value in loss_input.items()}
    return single_input, single_output, single_loss_input, token_mask


@pytest.mark.parametrize("cap_mode", ["mean_hard", "per_token", "none"])
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "mps",
            marks=pytest.mark.skipif(
                not torch.backends.mps.is_available(), reason="MPS is unavailable"
            ),
        ),
    ],
)
def test_sequence_loss_packed_batch_matches_mean_of_individual_systems(
    cap_mode, device
):
    """Packed CE losses and gradients are invariant to system partitioning."""
    network_input, network_output, loss_input = _packed_sequence_loss_inputs(device)
    loss_fn = SequenceLoss(
        max_t=1.0,
        sequence_weight=0.7,
        atom_type_weight=1.3,
        token_ce_cap_mode=cap_mode,
        sequence_loss_cap=1.0,
        atom_type_loss_cap=1.5,
    )

    packed_loss, packed_metrics = loss_fn(network_input, network_output, loss_input)
    packed_loss.backward()
    packed_sequence_gradient = network_output["sequence_logits_I"].grad.detach().clone()
    packed_atom_type_gradient = (
        network_output["atom_type_logits_I"].grad.detach().clone()
    )

    expected_sequence_gradient = torch.zeros_like(packed_sequence_gradient)
    expected_atom_type_gradient = torch.zeros_like(packed_atom_type_gradient)
    individual_losses = []
    individual_metrics = []
    batch_size = network_input["f"]["batch_size"]
    for system in range(batch_size):
        (
            single_input,
            single_output,
            single_loss_input,
            token_mask,
        ) = _slice_sequence_loss_system(
            network_input, network_output, loss_input, system
        )
        single_loss, single_metrics = loss_fn(
            single_input, single_output, single_loss_input
        )
        single_loss.backward()
        individual_losses.append(single_loss.detach())
        individual_metrics.append(single_metrics)
        expected_sequence_gradient[:, token_mask] = (
            single_output["sequence_logits_I"].grad / batch_size
        )
        expected_atom_type_gradient[:, token_mask] = (
            single_output["atom_type_logits_I"].grad / batch_size
        )

    torch.testing.assert_close(packed_loss, torch.stack(individual_losses).mean())
    torch.testing.assert_close(packed_sequence_gradient, expected_sequence_gradient)
    torch.testing.assert_close(packed_atom_type_gradient, expected_atom_type_gradient)
    for metric_name in (
        "sequence_loss",
        "effective_sequence_loss",
        "sequence_loss_above_cap",
        "sequence_token_loss_above_cap_fraction",
        "atom_type_loss",
        "effective_atom_type_loss",
        "atom_type_loss_above_cap",
        "atom_type_token_loss_above_cap_fraction",
    ):
        expected_metric = torch.stack(
            [metrics[metric_name] for metrics in individual_metrics]
        ).mean()
        torch.testing.assert_close(packed_metrics[metric_name], expected_metric)


def _sequence_loss_inputs(*, t: float):
    sequence_logits = torch.zeros(1, 2, 32, requires_grad=True)
    atom_type_logits = torch.zeros(1, 2, 128, requires_grad=True)
    network_input = {
        "t": torch.full((1, 1), t),
        "t_I": torch.full((1, 2), t),
        "f": {
            "diffusion_batch_size": 1,
            "batch_token_lvl": torch.zeros(2, dtype=torch.long),
        },
    }
    network_output = {
        "sequence_logits_I": sequence_logits,
        "sequence_indices_I": sequence_logits.detach().argmax(dim=-1),
        "atom_type_logits_I": atom_type_logits,
        "atom_type_indices_I": atom_type_logits.detach().argmax(dim=-1),
        "token_head_diagnostics": {
            "token_head_trunk_q_rms": torch.tensor(1.0),
        },
    }
    loss_input = {
        "seq_token_lvl": torch.zeros(2, dtype=torch.long),
        "sequence_valid_mask": torch.ones(2, dtype=torch.bool),
        "atom_type_gt_I": torch.zeros(2, dtype=torch.long),
        "atom_type_valid_mask": torch.zeros(2, dtype=torch.bool),
    }
    return network_input, network_output, loss_input


def test_sequence_loss_metric_keys_do_not_depend_on_valid_timesteps():
    """Ranks with no valid diffusion times must enter identical metric collectives."""
    loss = SequenceLoss(max_t=1.0)

    _, with_valid_time = loss(*_sequence_loss_inputs(t=0.5))
    _, without_valid_time = loss(*_sequence_loss_inputs(t=1.0))

    assert set(with_valid_time) == set(without_valid_time)
    assert "token_head_trunk_q_rms" in without_valid_time
    assert without_valid_time["effective_sequence_loss"] == 0
