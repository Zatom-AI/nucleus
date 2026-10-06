import torch
from beartype.typing import Any, Dict, List

ATOMISTIC_SYSTEM_LEVEL_KEYS = frozenset(
    {
        "energy_gt_B",
        "energy_valid_mask_B",
        "energy_present_mask_B",
        "force_present_mask_B",
        "domain_id_B",
        "domain_valid_mask_B",
        "source_atom_count_B",
        "retained_atom_count_B",
        "total_charge_B",
        "spin_multiplicity_B",
        "has_total_charge_B",
        "has_spin_multiplicity_B",
        "charge_valid_mask_B",
        "spin_valid_mask_B",
        "property_valid_mask_B",
        "property_task_id_B",
        "property_task_valid_mask_B",
    }
)

ATOMISTIC_REQUIRED_TARGET_KEYS = ATOMISTIC_SYSTEM_LEVEL_KEYS | {
    "force_gt_D_L",
    "force_valid_mask_L",
}


class SparseCollate:
    """
    Collate a list of examples (dictionaries) into a single sparse batch.

    Concatenates node/token level features along the sequence dimension (L or I),
    and creates 'batch_atom_lvl' and 'batch_token_lvl' index tensors.
    Assumes tensors with shapes [D, L, ...], [D, L], [L, ...], [I, ...], etc.
    """

    def __init__(self, max_num_tokens: int | None = None):
        if max_num_tokens is not None and max_num_tokens <= 0:
            raise ValueError(f"max_num_tokens must be positive, got {max_num_tokens}.")
        self.max_num_tokens = max_num_tokens

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Collate function for sparse input batching.

        Args:
            batch: List of examples (dictionaries) to collate.

        Returns:
            Dict with collated features and batch indices.
        """
        if len(batch) == 0:
            return {}

        # Atomistic-property batches use a fixed target contract.  If any part
        # is present, require every example to provide every part rather than
        # silently dropping labels through the optional-key intersection below.
        has_atomistic_targets = any(
            bool(
                ATOMISTIC_REQUIRED_TARGET_KEYS
                & set(example.get("ground_truth", {}).keys())
            )
            for example in batch
        )
        if has_atomistic_targets:
            for example_index, example in enumerate(batch):
                missing = ATOMISTIC_REQUIRED_TARGET_KEYS - set(
                    example.get("ground_truth", {}).keys()
                )
                if missing:
                    raise KeyError(
                        "Incomplete atomistic-property target contract for batch "
                        f"example {example_index}; missing {sorted(missing)}."
                    )

        collated = {}

        # We need to know which keys are atom level (L) and which are token level (I).
        # We can infer this by looking at lengths of known tensors.
        # ground_truth['coord_atom_lvl'] is [L, 3]
        # ground_truth['sequence_gt_I'] is [I, 32]

        batch_atom_lvl = []
        batch_token_lvl = []

        # We must shift indices like atom_to_token_map
        device = batch[0]["t"].device

        atom_to_token_shifts = []
        current_token_offset = 0

        for b_idx, example in enumerate(batch):
            L = example["ground_truth"]["coord_atom_lvl"].shape[0]
            I = example["ground_truth"]["sequence_gt_I"].shape[0]

            batch_atom_lvl.append(
                torch.full((L,), b_idx, dtype=torch.long, device=device)
            )
            batch_token_lvl.append(
                torch.full((I,), b_idx, dtype=torch.long, device=device)
            )

            atom_to_token_shifts.append(current_token_offset)
            current_token_offset += I

        batch_atom_lvl = torch.cat(batch_atom_lvl, dim=0)
        batch_token_lvl = torch.cat(batch_token_lvl, dim=0)

        # Function to recursively collate
        def _collate_dict(dict_list, token_offsets) -> Dict[str, Any]:
            """
            Recursively collate a list of dictionaries.

            Args:
                dict_list: List of dictionaries to collate.
                token_offsets: List of token offsets for each example.

            Returns:
                Dict with collated features and batch indices.
            """
            out = {}
            # Some dataset backends add optional annotations only for a subset of
            # examples (for example, OMol25 may include extra quantum properties
            # that are not present on every record). Collate only the keys that
            # exist in every example so a mixed batch does not fail on a missing
            # optional field.
            common_keys = [
                k
                for k in dict_list[0].keys()
                if k != "_token_budget_is_last_batch" and all(k in d for d in dict_list)
            ]
            num_dupes = dict_list[0]["t"].shape[0] if "t" in dict_list[0] else 1
            for k in common_keys:
                if isinstance(dict_list[0][k], dict):
                    out[k] = _collate_dict([d[k] for d in dict_list], token_offsets)
                elif isinstance(dict_list[0][k], torch.Tensor):
                    if k == "t":
                        out[k] = torch.stack([d[k] for d in dict_list], dim=1)
                        continue
                    elif k == "augmentation_rotation_D":
                        tensors = [d[k] for d in dict_list]
                        expected_shape = tensors[0].shape
                        if (
                            len(expected_shape) != 3
                            or expected_shape[-2:] != (3, 3)
                            or any(t.shape != expected_shape for t in tensors)
                        ):
                            raise ValueError(
                                "augmentation_rotation_D must have shape "
                                "[D, 3, 3] with the same D in every uncollated "
                                "example."
                            )
                        out[k] = torch.stack(tensors, dim=1)
                        continue
                    elif k in ATOMISTIC_SYSTEM_LEVEL_KEYS:
                        tensors = [d[k] for d in dict_list]
                        if any(t.numel() != 1 for t in tensors):
                            raise ValueError(
                                f"System-level field {k!r} must contain exactly "
                                "one value per uncollated example."
                            )
                        out[k] = torch.cat([t.reshape(1) for t in tensors], dim=0)
                        continue
                    elif k == "force_gt_D_L":
                        tensors = [d[k] for d in dict_list]
                        if any(t.ndim != 3 or t.shape[-1] != 3 for t in tensors):
                            raise ValueError(
                                "force_gt_D_L must have shape [D, L, 3] in "
                                "every uncollated example."
                            )
                        if any(t.shape[0] != tensors[0].shape[0] for t in tensors):
                            raise ValueError(
                                "All force_gt_D_L tensors in a sparse batch must "
                                "have the same diffusion dimension D."
                            )
                        out[k] = torch.cat(tensors, dim=1)
                        continue
                    elif k == "fractional_coord_gt_L":
                        diffusion_batch_size = batch[0]["t"].shape[0]
                        # A randomly translated material has one target per
                        # diffusion view. Broadcast static molecule/material
                        # targets only when such a per-view target is present.
                        has_diffusion_axis = any(d[k].ndim == 3 for d in dict_list)
                        tensors = []
                        for example_index, example in enumerate(batch):
                            tensor = dict_list[example_index][k]
                            num_atoms = example["ground_truth"]["coord_atom_lvl"].shape[
                                0
                            ]
                            if tensor.ndim not in (2, 3) or tensor.shape[-1] != 3:
                                raise ValueError(
                                    "fractional_coord_gt_L must have shape [L, 3] "
                                    "or [D, L, 3] in every uncollated example; "
                                    f"example {example_index} has shape "
                                    f"{tuple(tensor.shape)}."
                                )
                            if tensor.shape[-2] != num_atoms:
                                raise ValueError(
                                    "fractional_coord_gt_L must align with the atom "
                                    f"dimension L={num_atoms}; example {example_index} "
                                    f"has shape {tuple(tensor.shape)}."
                                )
                            if tensor.ndim == 2 and has_diffusion_axis:
                                tensor = tensor.unsqueeze(0).expand(
                                    diffusion_batch_size, -1, -1
                                )
                            elif (
                                tensor.ndim == 3
                                and tensor.shape[0] != diffusion_batch_size
                            ):
                                raise ValueError(
                                    "All [D, L, 3] fractional_coord_gt_L tensors in "
                                    "a sparse batch must have diffusion dimension "
                                    f"D={diffusion_batch_size}; example "
                                    f"{example_index} has shape {tuple(tensor.shape)}."
                                )
                            tensors.append(tensor)
                        concat_dim = 1 if has_diffusion_axis else 0
                        out[k] = torch.cat(tensors, dim=concat_dim)
                        continue
                    elif k in ("unindexing_pair_mask", "token_bonds"):
                        # Skip pairwise features - they are recomputed later.
                        out[k] = dict_list[0][k]
                        continue
                    elif k == "is_original_unindexed_token":
                        # Concatenate along the token dimension (I) and keep
                        # the per-example offsets for later indexing.
                        out[k] = torch.cat([d[k] for d in dict_list], dim=0)
                        continue
                    elif k in ("coord_atom_lvl_to_be_noised", "noise"):
                        tensors = [d[k] for d in dict_list]
                        # These are usually [L, 3] in sparse batches, but some
                        # diffusion paths already carry a leading diffusion axis
                        # and store them as [D, L, 3].
                        concat_dim = 0 if tensors[0].ndim == 2 else 1
                        out[k] = torch.cat(tensors, dim=concat_dim)
                        continue

                    # We need to concatenate along the sequence dimension.
                    # Typically, tensors are [L, ...], [I, ...], [D, L, ...], [D, I, ...]
                    # We will check if the dimension size matches L or I
                    dim_to_concat = None

                    # Heuristics to find concatenation dimension
                    shape = dict_list[0][k].shape
                    L_0 = batch[0]["ground_truth"]["coord_atom_lvl"].shape[0]
                    I_0 = batch[0]["ground_truth"]["sequence_gt_I"].shape[0]

                    if len(shape) > 0 and shape[0] == L_0:
                        dim_to_concat = 0
                    elif len(shape) > 0 and shape[0] == I_0:
                        dim_to_concat = 0
                    elif len(shape) > 1 and shape[1] == L_0:
                        dim_to_concat = 1
                    elif len(shape) > 1 and shape[1] == I_0:
                        dim_to_concat = 1

                    if dim_to_concat is not None:
                        tensors = [d[k] for d in dict_list]

                        # Special shifting for indices
                        if k == "atom_to_token_map":
                            tensors = [
                                t + offset for t, offset in zip(tensors, token_offsets)
                            ]

                        # Concatenate
                        out[k] = torch.cat(tensors, dim=dim_to_concat)
                    else:
                        # For scalars or unhandled shapes, just keep as list or stack if scalar(s)
                        if len(shape) == 0 or (
                            len(shape) == 1 and shape[0] == num_dupes
                        ):  # e.g. 't'
                            out[k] = torch.stack([d[k] for d in dict_list], dim=1)
                        else:
                            out[k] = [d[k] for d in dict_list]
                else:
                    out[k] = [d[k] for d in dict_list]
            return out

        assert (
            "t" in batch[0]
        ), f"Timestep 't' must be in the batch. Found keys: {list(batch[0].keys())}"
        collated = _collate_dict(batch, atom_to_token_shifts)

        # Add batch indices
        collated["feats"]["batch_atom_lvl"] = batch_atom_lvl
        collated["feats"]["batch_token_lvl"] = batch_token_lvl
        collated["feats"]["batch_num_tokens"] = torch.bincount(batch_token_lvl).tolist()
        diffusion_batch_size = collated["t"].shape[0]
        collated["feats"]["diffusion_batch_size"] = diffusion_batch_size
        collated["feats"]["batch_size"] = len(batch)
        expanded_num_tokens = diffusion_batch_size * batch_token_lvl.shape[0]
        collated["feats"]["batch_num_tokens_expanded"] = expanded_num_tokens
        if self.max_num_tokens is not None:
            if expanded_num_tokens > self.max_num_tokens:
                raise ValueError(
                    "Sparse batch exceeds max_num_tokens: "
                    f"expanded_num_tokens={expanded_num_tokens}, "
                    f"max_num_tokens={self.max_num_tokens}."
                )
            collated["feats"]["batch_util_rate"] = torch.tensor(
                expanded_num_tokens / self.max_num_tokens,
                dtype=torch.float32,
                device=device,
            )

        # Add atom-wise token indices specific to each batch element
        collated["ground_truth"]["atom_to_token_map_per_batch_element"] = torch.cat(
            [b["feats"]["atom_to_token_map"] for b in batch], dim=0
        )

        original_unindexed_token_lengths = torch.tensor(
            [len(b["ground_truth"]["is_original_unindexed_token"]) for b in batch],
            device=device,
        )
        original_unindexed_token_offset = (
            original_unindexed_token_lengths.cumsum(dim=0)
            - original_unindexed_token_lengths
        )
        collated["ground_truth"]["is_original_unindexed_token_batch_offset"] = (
            original_unindexed_token_offset[batch_atom_lvl]
        )

        # Create unindexing_pair_mask if it doesn't exist to prevent cross-batch attention
        N_atoms = batch_atom_lvl.shape[0]
        N_tokens = batch_token_lvl.shape[0]

        if "unindexing_pair_mask" in collated["feats"]:
            # unindexing_pair_mask is [N_tokens, N_tokens]
            # We must make it True for tokens from different batches
            # Since it was [I, I] for each example, we can block diagonalize it
            # where off-diagonal blocks are True (masked out)
            masks = [d["feats"]["unindexing_pair_mask"] for d in batch]
            block_diag = torch.ones(
                (N_tokens, N_tokens), dtype=torch.bool, device=masks[0].device
            )
            offset = 0
            for m in masks:
                s = m.shape[0]
                block_diag[offset : offset + s, offset : offset + s] = m
                offset += s
            collated["feats"]["unindexing_pair_mask"] = block_diag

        token_bond_masks = [d["feats"].get("token_bonds") for d in batch]
        if any(mask is not None for mask in token_bond_masks):
            # Bond conditioning is optional per example. Missing blocks remain
            # false so mixed-task batches expose bonds only to examples that
            # sampled structure prediction.
            first_mask = next(mask for mask in token_bond_masks if mask is not None)
            block_diag = torch.zeros(
                (N_tokens, N_tokens), dtype=torch.bool, device=first_mask.device
            )
            offset = 0
            for example, mask in zip(batch, token_bond_masks):
                num_tokens = example["ground_truth"]["sequence_gt_I"].shape[0]
                if mask is not None:
                    block_diag[
                        offset : offset + num_tokens,
                        offset : offset + num_tokens,
                    ] = mask
                offset += num_tokens
            collated["feats"]["token_bonds"] = block_diag

        # Build t_L and t_I which expand t to atom and token dimensions respectively.
        # This allows each example to have a different t.
        # t is originally [D] in each example.
        if "t" in collated:
            t_stack = collated["t"]  # [D, B]
            if t_stack.ndim != 2:
                raise ValueError(
                    "Expected collated timestep tensor to have shape [D, B], "
                    f"got {tuple(t_stack.shape)}."
                )
            D = t_stack.shape[0]
            t_L = torch.zeros((D, N_atoms), dtype=t_stack.dtype, device=t_stack.device)
            t_I = torch.zeros((D, N_tokens), dtype=t_stack.dtype, device=t_stack.device)
            for b_idx in range(len(batch)):
                mask_atom = batch_atom_lvl == b_idx
                mask_token = batch_token_lvl == b_idx
                t_L[:, mask_atom] = t_stack[:, b_idx].unsqueeze(1)
                t_I[:, mask_token] = t_stack[:, b_idx].unsqueeze(1)
            collated["t_L"] = t_L
            collated["t_I"] = t_I

        return collated
