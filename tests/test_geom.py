"""GEOM split loading, conditioning, and real Zatom-2 optimization."""

import json
import pickle
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from atomworks.io.tools.rdkit import atom_array_from_rdkit
from atomworks.ml.datasets import PandasDataset
from foundry.metrics.metric import MetricManager
from foundry.utils.datasets import assemble_val_loader_dict
from hydra import compose, initialize_config_dir
from hydra.errors import InstantiationException
from hydra.utils import instantiate
from omegaconf import OmegaConf
from rdkit import Chem
from torch.utils._pytree import tree_leaves
from torch.utils.data import SequentialSampler

from nucleus.trainers.zatom.zatom import ZatomDesignTrainer
from nucleus.utils.batching import SparseCollate
from nucleus.utils.datasets import (
    assemble_distributed_loader,
    get_diffusion_expanded_num_tokens,
)
from nucleus.utils.geom import (
    build_geom_dataset,
    build_geom_inference_dataset,
    build_geom_raw_dataset,
    load_geom_conformer,
)
from nucleus.utils.training_data import build_train_dataset_and_sampler


def _molecule(smiles="C[NH2+]c1ccccc1", offset=0.0):
    molecule = Chem.AddHs(Chem.MolFromSmiles(smiles))
    conformer = Chem.Conformer(molecule.GetNumAtoms())
    for index in range(molecule.GetNumAtoms()):
        conformer.SetAtomPosition(
            index, [0.4 * index + offset, float(index % 3), float(index % 2)]
        )
    molecule.AddConformer(conformer)
    molecule.SetProp("source_release", "synthetic")
    return molecule


def _write_split(raw, split, entries):
    directory = raw / split
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / f"{split}_data.pickle").open("wb") as handle:
        pickle.dump(entries, handle)


def _write_processed_split(directory, split, entries):
    from torch_geometric.data import Data, InMemoryDataset

    directory.mkdir(parents=True, exist_ok=True)
    records = [
        Data(
            id=f"geom_{split}_{molecule_index}_{conformer_index}",
            pos=torch.tensor(mol.GetConformer().GetPositions(), dtype=torch.float32),
            atom_types=torch.tensor([atom.GetAtomicNum() for atom in mol.GetAtoms()]),
        )
        for molecule_index, (_, conformers) in enumerate(entries)
        for conformer_index, mol in enumerate(conformers)
    ]
    # PyG uses no slices for a singleton; materialize the equivalent packed
    # offsets so tiny fixtures exercise the format of the full release.
    if len(records) == 1:
        record = records[0]
        torch.save(
            (
                {"id": [record.id], "pos": record.pos, "atom_types": record.atom_types},
                {
                    key: torch.tensor([0, len(record.pos)])
                    for key in ("pos", "atom_types")
                },
                Data,
            ),
            directory / f"{split}.pt",
        )
    else:
        InMemoryDataset.save(records, directory / f"{split}.pt")


def _write_inference_json(path, count=1):
    entries = {f"generate_{i}": {"length": "9m"} for i in range(count)}
    path.write_text(json.dumps(entries))
    return path


def _compose(monkeypatch, tmp_path, *overrides):
    root = Path(__file__).parents[1]
    monkeypatch.setenv("PROJECT_ROOT", str(root))
    monkeypatch.delenv("GEOM_DATA_DIR", raising=False)
    with initialize_config_dir(
        version_base="1.3", config_dir=str(root / "models/zatom2/configs")
    ):
        config = compose(
            config_name="train",
            overrides=[
                "experiment=train-geom-drugs",
                "trainer_runtime=cpu",
                f"paths.output_dir={tmp_path / 'output'}",
                f"paths.work_dir={tmp_path}",
                *overrides,
            ],
        )
    OmegaConf.resolve(config)
    return config


@pytest.mark.parametrize("remove_hydrogens", [False, True])
def test_conformer_preserves_coordinates_chemistry_and_source(remove_hydrogens):
    molecule = _molecule("C[NH2+][C@@H](C)c1ccccc1")
    original = atom_array_from_rdkit(molecule, remove_hydrogens=False)
    keep = original.element != "H" if remove_hydrogens else np.ones(len(original), bool)
    expected = original[keep]
    data = load_geom_conformer(
        molecule, example_id="selected", remove_hydrogens=remove_hydrogens
    )
    atoms = data["atom_array"]
    np.testing.assert_array_equal(atoms.source_atom_index, np.flatnonzero(keep))
    for name in ("coord", "element", "charge", "chiral_tag", "is_aromatic"):
        np.testing.assert_array_equal(getattr(atoms, name), getattr(expected, name))
    np.testing.assert_array_equal(atoms.bonds.as_array(), expected.bonds.as_array())
    assert atoms.charge.sum() == 1
    assert (atoms.res_name == "LIG").all()
    assert not atoms.is_polymer.any()
    assert data["extra_info"]["source_properties"]["source_release"] == "synthetic"
    assert data["specification"] == {}
    assert molecule.GetNumAtoms() == len(original)
    np.testing.assert_allclose(
        molecule.GetConformer().GetPositions(), original.coord, rtol=0, atol=5e-7
    )


def test_dataset_preserves_split_order_duplicate_smiles_and_conformer_limit(tmp_path):
    conformers = [_molecule(offset=float(i)) for i in range(6)]
    _write_split(tmp_path, "train", [("same", conformers), ("same", [conformers[0]])])
    _write_split(tmp_path, "val", [("held_out", [_molecule(offset=20.0)])])
    dataset = build_geom_raw_dataset(tmp_path, "train")
    assert isinstance(dataset, PandasDataset)
    assert len(dataset) == 6
    assert dataset.idx_to_id(4) == "geom/train/0/4"
    assert dataset.idx_to_id(5) == "geom/train/1/0"
    assert dataset.id_to_idx("geom/train/1/0") == 5
    assert (
        len(build_geom_raw_dataset(tmp_path, "train", max_conformers_per_molecule=None))
        == 7
    )
    assert (
        len(build_geom_raw_dataset(tmp_path, "train", max_conformers_per_molecule=1))
        == 2
    )
    validation = build_geom_raw_dataset(tmp_path, "val")
    assert len(validation) == 1
    assert validation[0]["extra_info"]["split"] == "val"
    assert validation[0]["extra_info"]["smiles"] == "held_out"
    assert validation[0]["atom_array"].coord[0, 0] == 20.0
    example = dataset[0]
    example["atom_array"].coord[:] = 999.0
    assert dataset[0]["atom_array"].coord[0, 0] == 0.0
    restored = pickle.loads(pickle.dumps(dataset))
    np.testing.assert_array_equal(
        restored[1]["atom_array"].coord, dataset[1]["atom_array"].coord
    )


@pytest.mark.parametrize("cap", [0, -1, 1.5, True])
def test_dataset_rejects_invalid_conformer_cap(tmp_path, cap):
    with pytest.raises(ValueError, match="positive int"):
        build_geom_raw_dataset(tmp_path, "train", max_conformers_per_molecule=cap)


@pytest.mark.parametrize("cap", [0, -1, 1.5, True])
def test_dataset_rejects_invalid_sample_cap(tmp_path, cap):
    with pytest.raises(ValueError, match="max_samples must be a positive int"):
        build_geom_dataset(tmp_path, "val", max_samples=cap)


@pytest.mark.parametrize("remove_hydrogens", [False, True])
def test_processed_conformers_are_separate_atom_arrays(tmp_path, remove_hydrogens):
    conformers = [_molecule("CCO", offset=float(i)) for i in range(5)]
    _write_processed_split(
        tmp_path, "train", [("CCO", conformers), ("CCO", [conformers[0]])]
    )
    dataset = build_geom_dataset(tmp_path, "train", remove_hydrogens=remove_hydrogens)
    assert len(dataset) == 6
    assert dataset.idx_to_id(4) == "geom/train/0/4"
    assert dataset.id_to_idx("geom/train/1/0") == 5
    for i in range(5):
        example = dataset[i]
        atoms = example["atom_array"]
        original = atom_array_from_rdkit(conformers[i], remove_hydrogens=False)
        keep = (
            original.element != "H"
            if remove_hydrogens
            else np.ones(len(original), bool)
        )
        np.testing.assert_array_equal(atoms.coord, original.coord[keep])
        np.testing.assert_array_equal(atoms.element, original.element[keep])
        np.testing.assert_array_equal(atoms.source_atom_index, np.flatnonzero(keep))
        assert example["extra_info"]["source_id"] == f"geom_train_0_{i}"
        assert example["extra_info"]["conformer_index"] == i
        assert len(np.unique(atoms.pn_unit_iid)) == 1
    # Arrays are independent and tensor storage is reopened in spawned workers.
    example["atom_array"].coord[:] = 999.0
    assert dataset[4]["atom_array"].coord[0, 0] == 4.0
    restored = pickle.loads(pickle.dumps(dataset))
    assert restored.loader._tensors is None
    np.testing.assert_array_equal(
        restored[4]["atom_array"].coord, dataset[4]["atom_array"].coord
    )
    assert len(build_geom_dataset(tmp_path, "train", max_samples=3)) == 3


def test_processed_loader_rejects_corrupt_slices_and_wrong_split(tmp_path):
    _write_processed_split(tmp_path, "train", [("CCO", [_molecule("CCO")] * 2)])
    path = tmp_path / "train.pt"
    payload = torch.load(path, weights_only=False)
    payload[0]["id"][0] = "geom_test_0_0"
    torch.save(payload, path)
    with pytest.raises(ValueError, match="invalid source conformer ID"):
        build_geom_dataset(tmp_path, "train")
    payload[0]["id"][0] = "geom_train_0_0"
    payload[1]["atom_types"][1] += 1
    torch.save(payload, path)
    with pytest.raises(ValueError, match="inconsistent GEOM"):
        build_geom_dataset(tmp_path, "train")


def test_download_helper_keeps_conformers_and_checks_existing_files(
    monkeypatch, tmp_path
):
    import hashlib
    import importlib.util
    import tarfile

    script = Path(__file__).parents[1] / "scripts/download_geom.py"
    spec = importlib.util.spec_from_file_location("download_geom", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = tmp_path / "source"
    for split in ("train", "val", "test"):
        _write_processed_split(source, split, [("CCO", [_molecule("CCO")] * 5)])
    archive_path = tmp_path / "fixture.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        for path in source.iterdir():
            archive.add(path, arcname=path.name)
    archive_bytes = archive_path.read_bytes()
    monkeypatch.setattr(
        module, "ARCHIVE_SHA256", hashlib.sha256(archive_bytes).hexdigest()
    )
    downloads = []

    def download(**kwargs):
        assert kwargs["repo_id"] == "Zatom-AI/geom"
        assert kwargs["repo_type"] == "dataset"
        assert kwargs["revision"] == module.REVISION
        downloads.append(kwargs)
        (kwargs["local_dir"] / kwargs["filename"]).write_bytes(archive_bytes)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", download)
    destination = tmp_path / "download"
    report = module.prepare_geom(destination)
    assert len(downloads) == 1
    for stats in report["splits"].values():
        assert stats["molecules"] == 1
        assert stats["conformers"] == 5
    assert module.prepare_geom(destination) == report
    assert len(downloads) == 1
    path = destination / "val.pt"
    damaged = bytearray(path.read_bytes())
    damaged[0] ^= 1
    path.write_bytes(damaged)
    with pytest.raises(FileExistsError, match="differs from archive"):
        module.prepare_geom(destination)
    assert path.read_bytes() == damaged


def test_validation_caps_json_requests_before_distributed_sampling(
    monkeypatch, tmp_path
):
    path = _write_inference_json(tmp_path / "inference.json", count=102)
    config = _compose(
        monkeypatch,
        tmp_path,
        f"datasets.inference_json={path}",
        "datasets.val.geom.dataset.max_samples=100",
    )
    rank_indices = []
    for rank in range(4):
        loaders = assemble_val_loader_dict(
            config.datasets.val,
            rank=rank,
            world_size=4,
            loader_cfg=OmegaConf.create(
                {"dataloader_params": {"batch_size": 1, "num_workers": 0}}
            ),
        )
        loader = loaders["geom"]
        assert len(loader.dataset) == 100
        assert len(loader) == 25
        assert loader.dataset.idx_to_id(99) == "generate_99"
        rank_indices.extend(list(loader.sampler))
    assert sorted(rank_indices) == list(range(100))
    assert (
        len(build_geom_inference_dataset(path, transform=None, max_samples=None)) == 102
    )
    assert (
        len(build_geom_inference_dataset(path, transform=None, max_samples=200)) == 102
    )


def test_missing_or_invalid_splits_never_fall_back_to_training(tmp_path):
    _write_processed_split(tmp_path, "train", [("CCO", [_molecule("CCO")])])
    with pytest.raises(FileNotFoundError, match="val.pt"):
        build_geom_dataset(tmp_path, "val")
    with pytest.raises(ValueError, match="Unknown GEOM split"):
        build_geom_dataset(tmp_path, "validation")


@pytest.mark.parametrize("entries", [[], ["invalid"], [("CCO", [])], [("CCO", [None])]])
def test_dataset_rejects_malformed_entries(tmp_path, entries):
    _write_split(tmp_path, "train", entries)
    with pytest.raises(ValueError):
        build_geom_raw_dataset(tmp_path, "train")


def test_coordinates_are_validated_on_access(tmp_path):
    molecule = _molecule()
    molecule.GetConformer().SetAtomPosition(0, [np.nan, 0.0, 0.0])
    _write_split(tmp_path, "train", [("bad", [molecule])])
    dataset = build_geom_raw_dataset(tmp_path, "train")
    assert len(dataset) == 1
    with pytest.raises(ValueError, match="finite"):
        dataset[0]
    molecule.RemoveAllConformers()
    with pytest.raises(ValueError, match="exactly one"):
        load_geom_conformer(molecule, example_id="bad")


@pytest.mark.parametrize(
    "experiment", ["train-geom-drugs", "train-geom-drugs-unconditional"]
)
def test_config_and_real_dataset_factories(monkeypatch, tmp_path, experiment):
    raw = tmp_path / "raw"
    _write_processed_split(raw, "train", [("train", [_molecule("CCO")])])
    path = _write_inference_json(tmp_path / "inference.json")
    config = _compose(
        monkeypatch,
        tmp_path,
        f"experiment={experiment}",
        f"datasets.data_dir={raw}",
        f"datasets.inference_json={path}",
        "datasets.val.geom.dataset.max_samples=100",
        "dataloader.train.n_fallback_retries=0",
    )
    assert set(config.datasets.train) == {"geom"}
    assert set(config.datasets.val) == {"geom"}
    assert set(config.logger) == {"csv"}
    assert "use_structurecloud" not in config.datasets
    assert "max_conformers_per_molecule" not in config.datasets.train.geom.dataset
    assert config.datasets.val.geom.dataset.max_samples == 100
    assert config.datasets.val.geom.dataset.transform.is_inference
    assert "split" not in config.datasets.val.geom.dataset
    assert "max_samples" not in config.datasets.train.geom.dataset
    assert config.dataloader.train.n_fallback_retries == 0
    assert config.trainer.n_examples_per_epoch == 1172577
    conditions = config.datasets.global_transform_args.train_conditions
    assert set(conditions) == {"unconditional"}
    assert conditions.unconditional.frequency == 1.0
    assert set(config.trainer.metrics) == {"molecule_generation_metrics"}
    assert not config.callbacks.log_grouped_zevals_structure_prediction_metrics_callback.enabled
    train, sampler = build_train_dataset_and_sampler(
        config,
        fabric=SimpleNamespace(global_rank=0, world_size=1),
        n_examples_per_epoch=1,
    )
    assert len(train) == 1
    assert sampler is not None
    assert "atom_array" not in train[0]
    assert "specification" not in train[0]
    loaders = assemble_val_loader_dict(
        config.datasets.val,
        loader_cfg=OmegaConf.create(
            {"dataloader_params": {"batch_size": 1, "num_workers": 0}}
        ),
    )
    assert set(loaders) == {"geom"}
    validation = loaders["geom"].dataset[0]
    assert validation["example_id"] == "generate_0"
    assert "atom_array" in validation
    assert len(validation["atom_array"]) == 9
    assert not validation["coord_atom_lvl_to_be_noised"].any()
    assert not validation["ground_truth"]["mask_atom_lvl"].any()
    assert validation["feats"]["atom_type_generation_mask"].all()
    assert not validation["atom_array"].is_motif_atom_with_fixed_seq.any()
    assert not validation["atom_array"].is_motif_atom_with_fixed_coord.any()


def test_geom_worker_transfer_bounds_shared_storages(monkeypatch, tmp_path):
    """Real GEOM features cross worker IPC with bounded mappings and no sample loss."""
    # Two complete worker chunks and one partial chunk, inside one packing window.
    _write_processed_split(
        tmp_path,
        "train",
        [("CCO", [_molecule("CCO")] * 5) for _ in range(13)],
    )
    config = _compose(
        monkeypatch,
        tmp_path,
        f"datasets.data_dir={tmp_path}",
        "dataloader.train.dataloader_params.num_workers=2",
        # Exercise dataset serialization into fresh workers as well as tensor IPC.
        "++dataloader.train.dataloader_params.multiprocessing_context=spawn",
        "++dataloader.train.dataloader_params.timeout=120",
        "dataloader.train.n_fallback_retries=0",
    )
    dataset = instantiate(config.datasets.train.geom.dataset)
    loader = assemble_distributed_loader(
        dataset,
        SequentialSampler(dataset),
        rank=0,
        world_size=1,
        loader_cfg=config.dataloader.train,
        max_num_tokens=config.datasets.max_num_tokens,
        stateful=True,
        shuffle=False,
    )
    assert loader.candidate_batch_size == 256
    assert loader.worker_chunk_size == 32
    batches = list(loader)
    examples = [example for batch in batches for example in batch]
    assert sorted(example["example_id"] for example in examples) == sorted(
        dataset.idx_to_id(index) for index in range(len(dataset))
    )
    assert all(
        sum(get_diffusion_expanded_num_tokens(example) for example in batch)
        <= config.datasets.max_num_tokens
        for batch in batches
    )
    tensors = [
        leaf
        for leaf in tree_leaves(examples)
        if torch.is_tensor(leaf) and leaf.numel() > 0
    ]
    assert tensors and all(tensor.is_shared() for tensor in tensors)
    storages = {tensor.untyped_storage().data_ptr() for tensor in tensors}
    num_dtypes = len({tensor.dtype for tensor in tensors})
    num_chunks = (
        len(dataset) + loader.worker_chunk_size - 1
    ) // loader.worker_chunk_size
    assert len(storages) <= num_chunks * num_dtypes
    assert len(storages) < len(tensors)


@pytest.mark.parametrize(
    "spec",
    [
        {"length": "9m", "input": "target.cif"},
        {"length": "9m", "total_charge": 1},
        {"length": "9P"},
        {"length": "0m"},
    ],
)
def test_geom_validation_rejects_conditioned_requests(tmp_path, spec):
    path = tmp_path / "inference.json"
    path.write_text(json.dumps({"request": spec}))
    with pytest.raises(ValueError, match="requires only a fixed 'Nm' length"):
        build_geom_inference_dataset(path, transform=None)






def _check_optimizer(config, molecule):
    dataset, sampler = build_train_dataset_and_sampler(
        config,
        fabric=SimpleNamespace(global_rank=0, world_size=1),
        n_examples_per_epoch=1,
    )
    loader = assemble_distributed_loader(
        dataset,
        sampler,
        rank=0,
        world_size=1,
        loader_cfg=config.dataloader.train,
        max_num_tokens=config.datasets.max_num_tokens,
        stateful=True,
    )
    example = next(iter(loader))[0]
    atoms = example["atom_array"]
    assert len(atoms) == molecule.GetNumAtoms()
    assert not atoms.is_motif_atom_with_fixed_coord.any()
    assert not atoms.is_motif_atom_with_fixed_seq.any()
    assert example["feats"]["atom_type_generation_mask"].all()
    assert "token_bonds" not in example["feats"]
    assert not example["feats"]["ref_pos"].any()
    assert not example["feats"]["ref_pos_is_ground_truth"].any()
    assert not example["feats"]["motif_pos"].any()
    expected = molecule.GetConformer().GetPositions()
    # The preserved AtomArray is the unaugmented reference for validation.
    np.testing.assert_allclose(atoms.coord, expected, rtol=0, atol=1e-6)

    batch = SparseCollate()([example])
    model = instantiate(config.model.net, _recursive_=False).train()
    trainer = object.__new__(ZatomDesignTrainer)
    trainer.state = {"model": model, "global_step": 0}
    trainer.tensor_validation_every_n_steps = 1
    network_input = trainer._assemble_network_inputs(batch)
    loss_input = trainer._assemble_loss_extra_info(batch)
    losses = {name: instantiate(value) for name, value in config.trainer.loss.items()}
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    initial = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if name.startswith("diffusion_module.euclidean_coord_head")
    }
    output = model(network_input, n_cycle=1)
    values = {
        name: loss(network_input, output, loss_input)[0]
        for name, loss in losses.items()
    }
    assert values["material_geometry_loss"].item() == 0.0
    assert values["sequence_loss"].item() > 0
    total = sum(values.values())
    assert torch.isfinite(total)
    total.backward()
    identity_gradients = []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name
            if name.startswith("diffusion_module.atom_type_head"):
                identity_gradients.append(parameter.grad)
    assert identity_gradients
    assert any(bool(gradient.any()) for gradient in identity_gradients)
    optimizer.step()
    assert any(
        not torch.equal(parameter.detach(), initial[name])
        for name, parameter in model.named_parameters()
        if name in initial
    )


@pytest.mark.parametrize("capacity", ["crop_size", "max_atoms_in_crop"])
def test_pipeline_rejects_partial_molecule_budgets(monkeypatch, tmp_path, capacity):
    config = _compose(monkeypatch, tmp_path, f"datasets.{capacity}=2")
    pipeline = instantiate(config.datasets.train.geom.dataset.transform)
    with pytest.raises(ValueError, match="complete GEOM molecule needs"):
        pipeline(load_geom_conformer(_molecule("CCO"), example_id="oversize"))


def test_pipeline_can_train_without_hydrogens(monkeypatch, tmp_path):
    config = _compose(monkeypatch, tmp_path, "datasets.remove_hydrogens=true")
    pipeline = instantiate(
        config.datasets.train.geom.dataset.transform, return_atom_array=True
    )
    result = pipeline(load_geom_conformer(_molecule("CCO"), example_id="heavy"))
    assert len(result["atom_array"]) == 3
    assert not (result["atom_array"].element == "H").any()


def test_pipeline_rejects_reference_coordinate_leakage(monkeypatch, tmp_path):
    config = _compose(monkeypatch, tmp_path)
    config.datasets.train.geom.dataset.transform.provide_reference_conformer_when_unmasked = True
    with pytest.raises(InstantiationException, match="no reference conformers"):
        instantiate(config.datasets.train.geom.dataset.transform)
