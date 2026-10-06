from __future__ import annotations


import argparse


import importlib.util


import json


import os


import random


import re


import signal


import subprocess


import sys


import threading


import time


from pathlib import Path


from types import SimpleNamespace


import numpy as np


import pytest


import torch


REPO = Path(__file__).resolve().parents[1]


sys.path.insert(0, str(REPO / 'scripts'))


import generate_scope_seeded as generation  # noqa: E402


class FakeEngine:
    def __init__(self, fail_sample=None):
        # Emulate initialization consuming different RNG state after a restart.
        random.random()
        torch.rand(5)
        self.fail_sample = fail_sample
        self.calls = []

    def run(self, *, inputs, n_batches, out_dir):
        sample = next(iter(json.loads(Path(inputs).read_text())))
        self.calls.append(sample)
        values = [random.random(), float(np.random.random()), torch.rand(()).item()]
        cif = Path(out_dir) / f"inputs_{sample}_0_model_0.cif"
        cif.write_text(json.dumps(values))
        if sample == self.fail_sample:
            raise RuntimeError("interrupted after CIF write, before commit")


def test_resumed_generation_matches_uninterrupted_and_resharded(tmp_path):
    specs = {f"scope_len050_{i:03d}": {"length": 50} for i in range(4)}
    reference = tmp_path / "reference/generated"
    generation.generate_samples(FakeEngine(), specs, reference, 43)
    resumed = tmp_path / "resumed/generated"
    with pytest.raises(RuntimeError, match="interrupted"):
        generation.generate_samples(FakeEngine("scope_len050_001"), specs, resumed, 43)
    engine = FakeEngine()
    generation.generate_samples(engine, specs, resumed, 43)
    assert engine.calls == list(specs)[1:]
    for index, (sample, spec) in enumerate(reversed(list(specs.items()))):
        shard = tmp_path / f"shard_{index}/generated"
        generation.generate_samples(FakeEngine(), {sample: spec}, shard, 43)
        name = f"inputs_{sample}_0_model_0.cif"
        assert (shard / name).read_bytes() == (reference / name).read_bytes()
        assert (resumed / name).read_bytes() == (reference / name).read_bytes()


@pytest.mark.parametrize("change", ["seed", "input", "cif", "missing"])
def test_resume_rejects_incompatible_or_damaged_sample(tmp_path, change):
    specs = {"scope_len050_000": {"length": 50}}
    generation.generate_samples(FakeEngine(), specs, tmp_path, 43)
    cif = next(tmp_path.glob("*.cif"))
    if change == "cif":
        cif.write_text("damaged")
    elif change == "missing":
        cif.unlink()
    elif change == "input":
        specs["scope_len050_000"]["length"] = 51
    with pytest.raises(ValueError, match="Completed sample changed"):
        generation.generate_samples(
            FakeEngine(), specs, tmp_path, 44 if change == "seed" else 43
        )


def test_mpnn_seed_is_applied_after_initialization_and_after_skips(
    tmp_path, monkeypatch
):
    import run_scope_self_consistency_shard as oracle

    backbones = tmp_path / "backbones"
    backbones.mkdir()
    for sample in ("a", "b", "c"):
        (backbones / f"{sample}.pdb").write_text("END\n")
    args = argparse.Namespace(
        backbone_dir=backbones,
        zevals_output_dir=tmp_path / "zevals",
        progress_json=tmp_path / "progress.json",
        shard_index=0,
        zevals_dir=REPO / "models/zevals",
        mpnn_num_sequences=8,
        mpnn_temperature=0.1,
        mpnn_backbone_noise=0.02,
        mpnn_seed=0,
        structure_predictor="esmfold2",
        esmfold_model_name="unused",
        esmfold_chunk_size=64,
        esmfold2_num_loops=3,
        esmfold2_num_sampling_steps=50,
        esmfold2_num_samples=1,
        esmfold2_seed=0,
        sc_rmsd_threshold=1.5,
        sc_atom="CA",
        tmalign_bin="unused",
        allow_incomplete_refolds=False,
    )
    # Both lazy oracle initialization and refolding perturb the RNG.
    folder = SimpleNamespace(_get_model=lambda **kw: torch.rand(17))
    monkeypatch.setattr(oracle, "_build_oracles", lambda *a, **kw: (None, folder))
    monkeypatch.setattr(oracle, "_collect_current", lambda *a: None)
    completed = set()
    values = {}

    def design(paths, **kwargs):
        name = paths[0].stem
        values[name] = torch.rand(8).tolist()
        completed.add(name)
        generation.seed_rng(999)

    monkeypatch.setattr(oracle, "compute_designability", design)
    monkeypatch.setattr(
        oracle, "_complete_sample", lambda out, p, *a: p.stem in completed
    )
    monkeypatch.setattr(
        oracle, "_complete_seeded_sample", lambda out, p, *a: p.stem in completed
    )
    oracle.run_shard(args)
    reference = values.copy()
    completed = {"a"}
    values = {}
    oracle.run_shard(args)
    assert values == {key: reference[key] for key in ("b", "c")}


def test_truncated_refold_is_not_reused(tmp_path, monkeypatch):
    import run_scope_self_consistency_shard as oracle
    from zevals.proteins import designability

    backbone = tmp_path / "backbone.pdb"
    structures = tmp_path / "designability/backbone/structures"
    structures.mkdir(parents=True)
    (structures.parent / "designed.fasta").write_text(">seq_0\nAA\n")
    for index in range(8):
        (structures / f"seq_{index}.cif").write_text("placeholder")
    monkeypatch.setattr(designability, "_atom_coords", lambda p: np.zeros((2, 3)))
    assert oracle._complete_seeded_sample(tmp_path, backbone, 8)
    monkeypatch.setattr(
        designability,
        "_atom_coords",
        lambda p: np.zeros((1 if p.name == "seq_7.cif" else 2, 3)),
    )
    assert not oracle._complete_seeded_sample(tmp_path, backbone, 8)


def test_generation_rejects_missing_ccd_before_engine_import(tmp_path, monkeypatch):
    monkeypatch.setenv("CCD_MIRROR_PATH", str(tmp_path / "missing"))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate",
            "--package",
            "zatom2",
            "--checkpoint",
            "unused.ckpt",
            "--inputs",
            "unused.json",
            "--output",
            str(tmp_path / "generated"),
            "--seed",
            "43",
        ],
    )
    with pytest.raises(ValueError, match="Missing readable CCD_MIRROR_PATH"):
        generation.main()
    assert not (tmp_path / "generated").exists()

