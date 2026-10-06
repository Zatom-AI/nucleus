"""Management of checkpoints"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import dotenv

PAPER_CHECKPOINT_REVISION = "74b6811f6356d701b6fa5ad627d529a2ef51540d"

DEFAULT_CHECKPOINT_DIR = Path.home() / ".nucleus" / "checkpoints"


def _normalize_paths(paths: Iterable[Path]) -> list[Path]:
    """Return absolute, deduplicated paths in order."""
    seen = set()
    normalized: List[Path] = []
    for path in paths:
        resolved = path.expanduser().absolute()
        if resolved not in seen:
            normalized.append(resolved)
            seen.add(resolved)
    return normalized


def get_default_checkpoint_dirs() -> list[Path]:
    """Return checkpoint search paths.

    Always starts with the default ~/.nucleus/checkpoints directory and then
    appends any additional directories from the colon-separated
    NUCLEUS_CHECKPOINT_DIRS environment variable.
    """
    env_dirs = os.environ.get("NUCLEUS_CHECKPOINT_DIRS", "")

    # For backward compatibility, also check NUCLEUS_CHECKPOINTS_DIR
    if not env_dirs:
        env_dirs = os.environ.get("NUCLEUS_CHECKPOINTS_DIR", "")

    extra_dirs: list[Path] = []
    if env_dirs:
        extra_dirs = [Path(p.strip()) for p in env_dirs.split(":") if p.strip()]
    return _normalize_paths([*extra_dirs, DEFAULT_CHECKPOINT_DIR])


def get_default_checkpoint_dir() -> Path:
    """Backward-compatible helper returning the primary checkpoint directory."""
    return get_default_checkpoint_dirs()[0]


def append_checkpoint_to_env(checkpoint_dirs: list[Path]) -> bool:
    dotenv_path = dotenv.find_dotenv()
    if dotenv_path:
        checkpoint_dirs = _normalize_paths(checkpoint_dirs)
        dotenv.set_key(
            dotenv_path=dotenv_path,
            key_to_set="NUCLEUS_CHECKPOINT_DIRS",
            value_to_set=":".join(str(path) for path in checkpoint_dirs),
            export=False,
        )
        return True
    else:
        return False


@dataclass
class RegisteredCheckpoint:
    filename: str
    description: str
    url: str | None = None
    repo_id: str | None = None
    repo_filename: str | None = None
    sha256: str | None = None
    revision: str | None = None

    def get_default_path(self):
        checkpoint_dirs = get_default_checkpoint_dirs()
        for checkpoint_dir in checkpoint_dirs:
            candidate = checkpoint_dir / self.filename
            if candidate.exists():
                return candidate
        return checkpoint_dirs[0] / self.filename


REGISTERED_CHECKPOINTS = {
    "zatom2-table1-ablation": RegisteredCheckpoint(
        filename=(
            "zatom2-table1-ablation-zatom2-s-omol25-joint-force-conditioned-"
            "epoch80.ckpt"
        ),
        description=(
            "Paper Table 1 ablation: Zatom-2-S, OMol25 joint force-conditioned, 80 "
            "epochs"
        ),
        repo_id="Zatom-AI/Zatom-2",
        revision=PAPER_CHECKPOINT_REVISION,
        repo_filename=(
            "paper-ready-checkpoints/paper-table1-zatom2-s-omol25-joint-force-"
            "conditioned/step-00104250.ckpt"
        ),
        sha256="7a08a2df01b1d1a2176586a231aa0e0a19c707ee1c6be673ea35d2dda4a08ab5",
    ),
    "zatom2-table1-joint-ablation": RegisteredCheckpoint(
        filename=(
            "zatom2-table1-ablation-omol25-omat24-joint-force-conditioned-epoch80.ckpt"
        ),
        description=(
            "Paper Table 1 ablation: Zatom-2, OMol25 + OMat24 joint force-conditioned, "
            "80 epochs"
        ),
        repo_id="Zatom-AI/Zatom-2",
        revision=PAPER_CHECKPOINT_REVISION,
        repo_filename=(
            "paper-ready-checkpoints/paper-table1-omol25-omat24-joint-force-"
            "conditioned/step-00097500.ckpt"
        ),
        sha256="a47edf3608c30bcb4c9273b7a71bc81b0ef7a282d6c0aa4a741463176c8590c8",
    ),
    "zatom2-table1-joint-no-force-ablation": RegisteredCheckpoint(
        filename="zatom2-table1-joint-no-force-ablation.ckpt",
        description=(
            "Paper Table 1: Zatom-2 base, OMol25 + OMat24 generation + structure "
            "without force conditioning, step 103,500"
        ),
        repo_id="Zatom-AI/Zatom-2",
        revision=PAPER_CHECKPOINT_REVISION,
        repo_filename=(
            "paper-ready-checkpoints/paper-table1-omol25-omat24-joint-without-force-"
            "conditioning/step-00103500.ckpt"
        ),
        sha256="7f9ec158b6427398d28d12a21afa9dd3a280eaf50dc2a89e3b3003b105354572",
    ),
    "zatom2-table1-joint-no-force-ablation-raw-epoch80": RegisteredCheckpoint(
        filename="zatom2-table1-joint-no-force-ablation-raw-epoch80.ckpt",
        description=(
            "Paper Table 1: Zatom-2 base, OMol25 + OMat24 generation + structure "
            "without force conditioning, raw epoch 80"
        ),
        repo_id="Zatom-AI/Zatom-2",
        revision=PAPER_CHECKPOINT_REVISION,
        repo_filename=(
            "paper-ready-checkpoints/paper-table1-omol25-omat24-joint-without-force-"
            "conditioning/epoch-0080.ckpt"
        ),
        sha256="a66516768170525dafe08ab5cb119b2d082a6b12c1ca66f9544817e2eee3a17b",
    ),
    "zatom2-table1-joint-properties-ablation": RegisteredCheckpoint(
        filename=(
            "zatom2-table1-ablation-omol25-omat24-joint-properties-force-conditioned-"
            "epoch80.ckpt"
        ),
        description=(
            "Paper Table 1 ablation: Zatom-2, OMol25 + OMat24 joint-properties force-"
            "conditioned, 80 epochs"
        ),
        repo_id="Zatom-AI/Zatom-2",
        revision=PAPER_CHECKPOINT_REVISION,
        repo_filename=(
            "paper-ready-checkpoints/paper-table1-omol25-omat24-joint-properties-force-"
            "conditioned/step-00097500.ckpt"
        ),
        sha256="7e265ebdc8190af066f2165ea2d28a8aea92525438ddcd61a7b313d9a7056963",
    ),
    "zatom2-table1-joint-properties-ablation-raw-epoch79": RegisteredCheckpoint(
        filename="zatom2-table1-joint-properties-ablation-raw-epoch79.ckpt",
        description=(
            "Paper Table 1: Zatom-2 base, OMol25 + OMat24 joint-properties force-"
            "conditioned, raw epoch 79"
        ),
        repo_id="Zatom-AI/Zatom-2",
        revision=PAPER_CHECKPOINT_REVISION,
        repo_filename=(
            "paper-ready-checkpoints/paper-table1-omol25-omat24-joint-properties-force-"
            "conditioned/epoch-0079.ckpt"
        ),
        sha256="955590d303208c24ce18e37d14f8feaf9c1c5c272d7c7ad336ab209b9e100751",
    ),
    "zatom2-table1-unconditional-ablation": RegisteredCheckpoint(
        filename=(
            "zatom2-table1-ablation-zatom2-s-omol25-unconditional-force-conditioned-"
            "epoch80.ckpt"
        ),
        description=(
            "Paper Table 1 ablation: Zatom-2-S, OMol25 unconditional force-"
            "conditioned, 80 epochs"
        ),
        repo_id="Zatom-AI/Zatom-2",
        revision=PAPER_CHECKPOINT_REVISION,
        repo_filename=(
            "paper-ready-checkpoints/paper-table1-zatom2-s-omol25-unconditional-force-"
            "conditioned/step-00097500.ckpt"
        ),
        sha256="895d67fcc784fd14aadb00c7ad507ef4c37ce4b25ea1927c11521acbfacdb0d4",
    ),
    "zatom2-table1-unconditional-no-force-ablation": RegisteredCheckpoint(
        filename=(
            "zatom2-table1-ablation-zatom2-s-omol25-unconditional-without-force-"
            "conditioning-epoch80.ckpt"
        ),
        description=(
            "Paper Table 1 ablation: Zatom-2-S, OMol25 unconditional without force "
            "conditioning, 80 epochs"
        ),
        repo_id="Zatom-AI/Zatom-2",
        revision=PAPER_CHECKPOINT_REVISION,
        repo_filename=(
            "paper-ready-checkpoints/paper-table1-zatom2-s-omol25-unconditional-"
            "without-force-conditioning/step-00097500.ckpt"
        ),
        sha256="6f99f4cbec8a38439ca71d162c806e90950859f14130972a4ed18d05d526c3e3",
    ),
    "zatom2-table1-zatom2-l-joint-properties-ablation": RegisteredCheckpoint(
        filename=(
            "zatom2-table1-ablation-zatom2-l-omol25-omat24-joint-properties-force-"
            "conditioned-epoch80.ckpt"
        ),
        description=(
            "Paper Table 1 final model VI: Zatom-2-L, OMol25 + OMat24 joint-properties "
            "force-conditioned, 80 epochs"
        ),
        repo_id="Zatom-AI/Zatom-2",
        revision=PAPER_CHECKPOINT_REVISION,
        repo_filename=(
            "paper-ready-checkpoints/paper-table1-zatom2-l-omol25-omat24-joint-"
            "properties-force-conditioned/step-00096750.ckpt"
        ),
        sha256="ebaa4afa55b7ab4d51df819ad45a95182cf03435a05460a37d1e64242c79b386",
    ),
    "zatom2-table2-qm9-mp20-unconditional-no-force-ablation": RegisteredCheckpoint(
        filename=(
            "zatom2-table2-ablation-qm9-mp20-unconditional-without-force-conditioning-"
            "epoch9925.ckpt"
        ),
        description=(
            "Paper Table 2 ablation: QM9 + MP20 unconditional without force "
            "conditioning, epoch 9,925 of 10,000"
        ),
        repo_id="Zatom-AI/Zatom-2",
        revision=PAPER_CHECKPOINT_REVISION,
        repo_filename=(
            "paper-ready-checkpoints/paper-table2-qm9-mp20-unconditional-without-force-"
            "conditioning/epoch-9925.ckpt"
        ),
        sha256="412e14210ffe4bb7d50d87c64bdea658a767238ea87f56888dd6c06734d02e55",
    ),
    "zatom2-table3-scope-from-joint-force-conditioned": RegisteredCheckpoint(
        filename="zatom2-table3-scope-from-joint-force-conditioned-step00120000.ckpt",
        description=(
            "Paper Table 3 final: Zatom-2 SCOPe-2k, model IV with force conditioning, "
            "20,000 finetuning steps, global step 120,000"
        ),
        repo_id="Zatom-AI/Zatom-2",
        revision=PAPER_CHECKPOINT_REVISION,
        repo_filename=(
            "paper-ready-checkpoints/paper-table3-scope-from-joint-force-conditioned/"
            "step-00120000.ckpt"
        ),
        sha256="d146c9aa1c3270c6c976cbcdff0f77734c0bddfcdf1be8c53b2988699b63af00",
    ),
    "zatom2-table3-scope-from-joint-properties-force-conditioned": RegisteredCheckpoint(
        filename=(
            "zatom2-table3-scope-from-joint-properties-force-conditioned-"
            "step00130000.ckpt"
        ),
        description=(
            "Paper Table 3 final: Zatom-2 SCOPe-2k, model V with force conditioning, "
            "30,000 finetuning steps, global step 130,000"
        ),
        repo_id="Zatom-AI/Zatom-2",
        revision=PAPER_CHECKPOINT_REVISION,
        repo_filename=(
            "paper-ready-checkpoints/paper-table3-scope-from-joint-properties-force-"
            "conditioned/step-00130000.ckpt"
        ),
        sha256="803faeab28541bdae974d8f86cc9265bd74728c949e6b2e435504583d58e5563",
    ),
    "zatom2-table3-scope-from-joint-without-force-conditioning": RegisteredCheckpoint(
        filename=(
            "zatom2-table3-scope-from-joint-without-force-conditioning-"
            "step00125000.ckpt"
        ),
        description=(
            "Paper Table 3 final: Zatom-2 SCOPe-2k, model IV without force "
            "conditioning, 25,000 finetuning steps, global step 125,000"
        ),
        repo_id="Zatom-AI/Zatom-2",
        revision=PAPER_CHECKPOINT_REVISION,
        repo_filename=(
            "paper-ready-checkpoints/paper-table3-scope-from-joint-without-force-"
            "conditioning/step-00125000.ckpt"
        ),
        sha256="4dc8df979d7ef6aa205adf77a6d742cb304e1e22d8b7d808056f2d5296103ed6",
    ),
    "zatom2-table3-scope-no-pretraining": RegisteredCheckpoint(
        filename="zatom2-table3-scope-no-pretraining-step00025000.ckpt",
        description=(
            "Paper Table 3 final: Zatom-2 SCOPe-2k, scratch training, global step "
            "25,000"
        ),
        repo_id="Zatom-AI/Zatom-2",
        revision=PAPER_CHECKPOINT_REVISION,
        repo_filename=(
            "paper-ready-checkpoints/paper-table3-scope-no-pretraining/step-"
            "00025000.ckpt"
        ),
        sha256="a166d5a5917921c353164c8c69e3baf77f5af6a0b8b48b63e4750e100abb387d",
    ),
    "zatom2-table10-properties-finetune-from-joint": RegisteredCheckpoint(
        filename=(
            "zatom2-table10-finetune-properties-from-joint-force-conditioned-"
            "epoch120.ckpt"
        ),
        description=(
            "Paper Table 10 fine-tune: Zatom-2 properties from joint force-conditioned "
            "pretraining, 80 + 40 epochs"
        ),
        repo_id="Zatom-AI/Zatom-2",
        revision=PAPER_CHECKPOINT_REVISION,
        repo_filename=(
            "paper-ready-checkpoints/paper-table10-finetune-properties-from-joint-"
            "force-conditioned/epoch-0119.ckpt"
        ),
        sha256="5c12d4f631a4017ea5dacc3fe9704a34ef26c37ad46ebfa3b3a6c50f50e10bbf",
    ),
    "zatom2-table10-properties-finetune-from-joint-properties": RegisteredCheckpoint(
        filename=(
            "zatom2-table10-finetune-properties-from-joint-properties-force-"
            "conditioned-epoch120.ckpt"
        ),
        description=(
            "Paper Table 10 fine-tune: Zatom-2 properties from joint-properties force-"
            "conditioned pretraining, 80 + 40 epochs"
        ),
        repo_id="Zatom-AI/Zatom-2",
        revision=PAPER_CHECKPOINT_REVISION,
        repo_filename=(
            "paper-ready-checkpoints/paper-table10-finetune-properties-from-joint-"
            "properties-force-conditioned/epoch-0119.ckpt"
        ),
        sha256="f2b79ed6673b54dbedc55ab7f93db6fb353c4060e9d5d800e5fdf212e50b2752",
    ),
    "zatom2-table10-properties-finetune-from-joint-properties-epoch240": (
        RegisteredCheckpoint(
            filename=(
                "zatom2-table10-finetune-properties-from-joint-properties-force-"
                "conditioned-epoch240.ckpt"
            ),
            description=(
                "Paper Table 10 MLIP fine-tune: Zatom-2 properties from joint-"
                "properties force-conditioned pretraining, nominal epoch 240 (raw "
                "epoch 236)"
            ),
            repo_id="Zatom-AI/Zatom-2",
        revision=PAPER_CHECKPOINT_REVISION,
            repo_filename=(
                "paper-ready-checkpoints/paper-table10-finetune-properties-from-joint-"
                "properties-force-conditioned/epoch-0236.ckpt"
            ),
            sha256="f6bf4fb1ef5e1d88c2dc90de43216163db068a48b8304406ed2bf0b15edcac44",
        )
    ),
    "zatom2-figure3-500k-joint-properties-ablation": RegisteredCheckpoint(
        filename=(
            "zatom2-figure3-ablation-omol25-500k-omat24-500k-joint-properties-force-"
            "conditioned-epoch20.ckpt"
        ),
        description=(
            "Paper Figure 3 ablation: Zatom-2, OMol25 500k + OMat24 500k joint-"
            "properties force-conditioned, 20 epochs"
        ),
        repo_id="Zatom-AI/Zatom-2",
        revision=PAPER_CHECKPOINT_REVISION,
        repo_filename=(
            "paper-ready-checkpoints/paper-figure3-zatom2-omol25-500k-omat24-500k-"
            "joint-properties-force-conditioned/epoch-0019.ckpt"
        ),
        sha256="e78da9d1a90d67e893740cc9af8609cda16e88aa3e635da8bbe0b2a9fbd3a4de",
    ),
    "zatom2-figure3-500k-joint-properties-ablation-epoch16": RegisteredCheckpoint(
        filename="zatom2-figure3-500k-joint-properties-ablation-epoch16.ckpt",
        description=(
            "Paper Figure 3 ablation: Zatom-2 base, OMol25 500k + OMat24 500k joint-"
            "properties force-conditioned, 16 epochs (raw epoch 15)"
        ),
        repo_id="Zatom-AI/Zatom-2",
        revision=PAPER_CHECKPOINT_REVISION,
        repo_filename=(
            "paper-ready-checkpoints/paper-figure3-zatom2-omol25-500k-omat24-500k-"
            "joint-properties-force-conditioned/epoch-0015.ckpt"
        ),
        sha256="b2aa388028d607847a580f4f89a06b10a9d386dd5109f09c0d13f0c6cee14ff3",
    ),
    "zatom2-figure3-500k-joint-properties-ablation-epoch40": RegisteredCheckpoint(
        filename=(
            "zatom2-figure3-ablation-omol25-500k-omat24-500k-joint-properties-force-"
            "conditioned-epoch40.ckpt"
        ),
        description=(
            "Paper Figure 3 ablation: Zatom-2, OMol25 500k + OMat24 500k joint-"
            "properties force-conditioned, 40 epochs"
        ),
        repo_id="Zatom-AI/Zatom-2",
        revision=PAPER_CHECKPOINT_REVISION,
        repo_filename=(
            "paper-ready-checkpoints/paper-figure3-zatom2-omol25-500k-omat24-500k-"
            "joint-properties-force-conditioned/epoch-0039.ckpt"
        ),
        sha256="6653853c6c8e8d33c69837bfde8e7e60dd16c412c07e1cf5540c8d6815d2f0ed",
    ),
    "zatom2-figure3-500k-joint-properties-ablation-epoch8": RegisteredCheckpoint(
        filename="zatom2-figure3-500k-joint-properties-ablation-epoch8.ckpt",
        description=(
            "Paper Figure 3 ablation: Zatom-2 base, OMol25 500k + OMat24 500k joint-"
            "properties force-conditioned, 8 epochs (raw epoch 7)"
        ),
        repo_id="Zatom-AI/Zatom-2",
        revision=PAPER_CHECKPOINT_REVISION,
        repo_filename=(
            "paper-ready-checkpoints/paper-figure3-zatom2-omol25-500k-omat24-500k-"
            "joint-properties-force-conditioned/epoch-0007.ckpt"
        ),
        sha256="c0bc631a7e4a54d7a0be658f078bd366c12610115556ec82afbe776171e98ef7",
    ),
    "zatom2-figure3-500k-joint-properties-ablation-step3000": RegisteredCheckpoint(
        filename="zatom2-figure3-500k-joint-properties-ablation-step3000.ckpt",
        description=(
            "Paper Figure 3 ablation: Zatom-2 base, OMol25 500k + OMat24 500k joint-"
            "properties force-conditioned, step 3,000"
        ),
        repo_id="Zatom-AI/Zatom-2",
        revision=PAPER_CHECKPOINT_REVISION,
        repo_filename=(
            "paper-ready-checkpoints/paper-figure3-zatom2-omol25-500k-omat24-500k-"
            "joint-properties-force-conditioned/step-00003000.ckpt"
        ),
        sha256="9ea35ce2b69bf5ebcbb179b5ccf8a6a28412a87db5784758498ab35daddc80e9",
    ),
    "zatom2-figure3-zatom2-l-500k-joint-properties-ablation": RegisteredCheckpoint(
        filename=(
            "zatom2-figure3-ablation-zatom2-l-omol25-500k-omat24-500k-joint-properties-"
            "force-conditioned-epoch20.ckpt"
        ),
        description=(
            "Paper Figure 3 ablation: Zatom-2-L, OMol25 500k + OMat24 500k joint-"
            "properties force-conditioned, 20 epochs"
        ),
        repo_id="Zatom-AI/Zatom-2",
        revision=PAPER_CHECKPOINT_REVISION,
        repo_filename=(
            "paper-ready-checkpoints/paper-figure3-zatom2-l-omol25-500k-omat24-500k-"
            "joint-properties-force-conditioned/epoch-0019.ckpt"
        ),
        sha256="7375b21f3765904ef0938859b00c2ca960ed4f8a9de4ddc451d5ed073dbab282",
    ),
    "zatom2-figure3-zatom2-l-500k-joint-properties-ablation-epoch16": (
        RegisteredCheckpoint(
            filename=(
                "zatom2-figure3-zatom2-l-500k-joint-properties-ablation-epoch16.ckpt"
            ),
            description=(
                "Paper Figure 3 ablation: Zatom-2-L, OMol25 500k + OMat24 500k joint-"
                "properties force-conditioned, 16 epochs (raw epoch 15)"
            ),
            repo_id="Zatom-AI/Zatom-2",
        revision=PAPER_CHECKPOINT_REVISION,
            repo_filename=(
                "paper-ready-checkpoints/paper-figure3-zatom2-l-omol25-500k-omat24-"
                "500k-joint-properties-force-conditioned/epoch-0015.ckpt"
            ),
            sha256="96c0ae142abe1bc7812d4e1597ec515a7871723b350bc0d4cd4b8193f257ac2f",
        )
    ),
    "zatom2-figure3-zatom2-l-500k-joint-properties-ablation-epoch40": (
        RegisteredCheckpoint(
            filename=(
                "zatom2-figure3-ablation-zatom2-l-omol25-500k-omat24-500k-joint-"
                "properties-force-conditioned-epoch40.ckpt"
            ),
            description=(
                "Paper Figure 3 ablation: Zatom-2-L, OMol25 500k + OMat24 500k joint-"
                "properties force-conditioned, 40 epochs"
            ),
            repo_id="Zatom-AI/Zatom-2",
        revision=PAPER_CHECKPOINT_REVISION,
            repo_filename=(
                "paper-ready-checkpoints/paper-figure3-zatom2-l-omol25-500k-omat24-"
                "500k-joint-properties-force-conditioned/epoch-0039.ckpt"
            ),
            sha256="244389d755dce61bdcd268c0fbb35be99412ac56c885721ecfd628567025f576",
        )
    ),
    "zatom2-figure3-zatom2-l-500k-joint-properties-ablation-epoch8": (
        RegisteredCheckpoint(
            filename=(
                "zatom2-figure3-zatom2-l-500k-joint-properties-ablation-epoch8.ckpt"
            ),
            description=(
                "Paper Figure 3 ablation: Zatom-2-L, OMol25 500k + OMat24 500k joint-"
                "properties force-conditioned, 8 epochs (raw epoch 7)"
            ),
            repo_id="Zatom-AI/Zatom-2",
        revision=PAPER_CHECKPOINT_REVISION,
            repo_filename=(
                "paper-ready-checkpoints/paper-figure3-zatom2-l-omol25-500k-omat24-"
                "500k-joint-properties-force-conditioned/epoch-0007.ckpt"
            ),
            sha256="6324be77023fa2df4f64c62405ed94122ec90c318ae077305051686583bc2e84",
        )
    ),
    "zatom2-figure3-zatom2-l-500k-joint-properties-ablation-step3000": (
        RegisteredCheckpoint(
            filename=(
                "zatom2-figure3-zatom2-l-500k-joint-properties-ablation-step3000.ckpt"
            ),
            description=(
                "Paper Figure 3 ablation: Zatom-2-L, OMol25 500k + OMat24 500k joint-"
                "properties force-conditioned, step 3,000"
            ),
            repo_id="Zatom-AI/Zatom-2",
        revision=PAPER_CHECKPOINT_REVISION,
            repo_filename=(
                "paper-ready-checkpoints/paper-figure3-zatom2-l-omol25-500k-omat24-"
                "500k-joint-properties-force-conditioned/step-00003000.ckpt"
            ),
            sha256="63589ad85bec996b6bf75f2a61f1b5db39f999016c89b95f86409b2109c113ec",
        )
    ),
}

# The short public name selects configuration VI from the manuscript.
REGISTERED_CHECKPOINTS["zatom2"] = REGISTERED_CHECKPOINTS[
    "zatom2-table1-zatom2-l-joint-properties-ablation"
]
