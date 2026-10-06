"""Backs the `zevals setup` console-script subcommand.

FoldSeek, MMseqs2, and USalign (TMalign/qTMclust) are compiled binaries, not
pip packages — `pip install zevals[proteins]` can't install them for you, so
this module runs their installer scripts (`scripts/install_foldseek.sh`,
`scripts/install_mmseqs.sh`, `scripts/install_usalign.sh`) on request
instead. FoldSeek/MMseqs2 ship pre-built static binaries (downloaded
directly); USalign ships no binaries at all, so its script compiles
`TMalign`/`qTMclust` from source via `g++` instead — still a single `zevals
setup` call either way. All three scripts are idempotent (skip re-download/
rebuild if the binary is already on PATH), so calling this repeatedly is safe.

Separately, `build_database()` builds a FoldSeek reference database (e.g.
PDB) via `scripts/build_foldseek_db.sh` — needed for the *novelty* metric,
not for structural/sequence diversity. This is a deliberately distinct,
explicit step from binary installation: reference databases are multi-GB+
and location-sensitive (you want them on persistent cluster storage, not
wherever a binary happens to land), so `install()` never triggers it
implicitly.

Third, `download_checkpoint()` fetches the small (~6-30MB), publicly-hosted,
no-license-needed inverse-folding checkpoints (ProteinMPNN/SolubleMPNN/
LigandMPNN/NA-MPNN variants) — unlike the FoldSeek database, these are cheap
enough to have a genuine "just works" default: `default_checkpoint_dir()`
gives a predictable per-user cache location, and every
`oracle/inverse_folder/*.yaml` config's `weights_dir`/`checkpoint_path`
default now points there, so `zevals setup --checkpoints ...` (with no
`--checkpoint-dir` override) is enough to make the *default* proteins/rna
configs work with zero further overrides.
"""

from __future__ import annotations

import os
import subprocess
import urllib.request
from pathlib import Path

_INSTALLERS = {
    "foldseek": "install_foldseek.sh",
    "mmseqs": "install_mmseqs.sh",
    "usalign": "install_usalign.sh",
}

# name -> (files {filename: url}, subdir under the checkpoint cache root,
# path_style). Sourced from the upstream repos' own checkpoint hosting
# (verified live):
#   - ProteinMPNN's vanilla/ca/soluble _model_weights/ dirs are bundled
#     directly in github.com/dauparas/ProteinMPNN (raw.githubusercontent.com).
#   - SolubleMPNN (LigandMPNN-repo naming) and LigandMPNN itself are hosted
#     at files.ipd.uw.edu, per LigandMPNN's own get_model_params.sh.
#   - NA-MPNN's design/specificity checkpoints are bundled directly in
#     github.com/baker-laboratory/NA-MPNN (raw.githubusercontent.com).
# Only the default model_name/checkpoint each respective oracle config
# expects is fetched (v_48_020 / solublempnn_v_48_020 / ligandmpnn_v_32_010_25
# / s_19137 / s_70114) — not every noise-level variant — to keep this lean;
# the resulting directory still accepts additional hand-downloaded variants
# alongside it.
#
# path_style controls what download_checkpoint() returns: "dir" for oracle
# config fields that take a directory (ProteinMPNNFolder's `weights_dir`),
# "file" for fields that take the checkpoint file itself (LigandMPNNFolder /
# NAMPNNFolder's `checkpoint_path`).
_CHECKPOINTS: dict[str, dict] = {
    "protein_mpnn_vanilla": {
        "files": {
            "v_48_020.pt": "https://raw.githubusercontent.com/dauparas/ProteinMPNN/main/vanilla_model_weights/v_48_020.pt",
        },
        "subdir": "protein_mpnn/vanilla_model_weights",
        "path_style": "dir",
    },
    "protein_mpnn_ca": {
        "files": {
            "v_48_020.pt": "https://raw.githubusercontent.com/dauparas/ProteinMPNN/main/ca_model_weights/v_48_020.pt",
        },
        "subdir": "protein_mpnn/ca_model_weights",
        "path_style": "dir",
    },
    "protein_mpnn_soluble": {
        "files": {
            "v_48_020.pt": "https://raw.githubusercontent.com/dauparas/ProteinMPNN/main/soluble_model_weights/v_48_020.pt",
        },
        "subdir": "protein_mpnn/soluble_model_weights",
        "path_style": "dir",
    },
    "soluble_mpnn": {
        "files": {
            "solublempnn_v_48_020.pt": "https://files.ipd.uw.edu/pub/ligandmpnn/solublempnn_v_48_020.pt",
        },
        "subdir": "soluble_mpnn",
        "path_style": "dir",
    },
    "ligand_mpnn": {
        "files": {
            "ligandmpnn_v_32_010_25.pt": "https://files.ipd.uw.edu/pub/ligandmpnn/ligandmpnn_v_32_010_25.pt",
        },
        "subdir": "ligand_mpnn",
        "path_style": "file",
    },
    "na_mpnn_design": {
        "files": {
            "s_19137.pt": "https://raw.githubusercontent.com/baker-laboratory/NA-MPNN/main/models/design_model/s_19137.pt",
        },
        "subdir": "na_mpnn",
        "path_style": "file",
    },
    "na_mpnn_specificity": {
        "files": {
            "s_70114.pt": "https://raw.githubusercontent.com/baker-laboratory/NA-MPNN/main/models/specificity_model/s_70114.pt",
        },
        "subdir": "na_mpnn",
        "path_style": "file",
    },
    "rhofold": {
        "files": {
            "RhoFold_pretrained.pt": "https://huggingface.co/cuhkaih/rhofold/resolve/main/rhofold_pretrained_params.pt",
        },
        "subdir": "rhofold",
        "path_style": "file",
    },
    # gRNAde itself stays a subprocess-only oracle (see oracles/inverse_folders/
    # grnade.py) — its design.py entrypoint hard-requires two more checkpoints
    # (RibonanzaNet + RibonanzaNet-SS, for scoring/filtering designs) plus a
    # torch-geometric-based conda env zevals doesn't install. These three
    # entries only pre-stage the weights (all public, MIT/HF-hosted, no auth)
    # to the spot gRNAde's own `checkpoints/` layout expects, for whoever sets
    # that env up separately.
    "grnade": {
        "files": {
            "gRNAde_drop3d@0.75_maxlen@500.h5": "https://huggingface.co/chaitjo/gRNAde/resolve/main/gRNAde_drop3d%400.75_maxlen%40500.h5",
        },
        "subdir": "grnade",
        "path_style": "file",
    },
    "grnade_ribonanzanet": {
        "files": {
            "ribonanzanet.pt": "https://huggingface.co/chaitjo/gRNAde/resolve/main/ribonanzanet/ribonanzanet.pt",
        },
        "subdir": "grnade/ribonanzanet",
        "path_style": "file",
    },
    "grnade_ribonanzanet_ss": {
        "files": {
            "ribonanzanet_ss.pt": "https://huggingface.co/chaitjo/gRNAde/resolve/main/ribonanzanet_sec_struct/ribonanzanet_ss.pt",
        },
        "subdir": "grnade/ribonanzanet_sec_struct",
        "path_style": "file",
    },
}


def _resolve_scripts_dir() -> Path:
    """Locate the `scripts/` directory in either a source checkout or an
    installed wheel — same installed-wheel-vs-checkout logic as
    `pipeline.suite._resolve_config_dir()`, just for a different
    force-included directory (see `[tool.hatch.build.targets.wheel.force-include]`
    in pyproject.toml)."""
    here = Path(__file__).resolve()
    candidates = [
        here.parent / "scripts",  # installed: <pkg>/zevals/scripts
        here.parents[2] / "scripts",  # source checkout: zevals/scripts
    ]
    for candidate in candidates:
        if (candidate / "install_foldseek.sh").is_file():
            return candidate
    raise FileNotFoundError(
        "Could not locate zevals's scripts/ directory. Checked: "
        + ", ".join(str(c) for c in candidates)
    )


def install(
    tools: list[str] | None = None,
    install_dir: str | None = None,
    force: bool = False,
) -> None:
    """Run the static-binary installer script(s) for `tools`.

    Args:
        tools: Which tools to install — any of `"foldseek"`, `"mmseqs"`,
            `"usalign"` (builds `TMalign`/`qTMclust` from source via `g++` —
            see `scripts/install_usalign.sh`; needs a C++ compiler on PATH).
            `None` (default) installs all three.
        install_dir: Passed through to each script (defaults to `~/.local/bin`
            if not given — see the scripts themselves).
        force: Reinstall even if the binary is already on PATH.

    Raises:
        subprocess.CalledProcessError: if an installer script fails (e.g. a
            network error downloading the tarball).
    """
    tools = tools or list(_INSTALLERS)
    unknown = set(tools) - set(_INSTALLERS)
    if unknown:
        raise ValueError(
            f"Unknown tool(s) {sorted(unknown)}. Available: {sorted(_INSTALLERS)}"
        )

    scripts_dir = _resolve_scripts_dir()
    for tool in tools:
        script_path = scripts_dir / _INSTALLERS[tool]
        cmd = ["bash", str(script_path)]
        if install_dir is not None:
            cmd.append(install_dir)
            if force:
                cmd.append("--force")
        elif force:
            cmd += [str(Path.home() / ".local" / "bin"), "--force"]
        subprocess.run(cmd, check=True)


def build_database(db_name: str, output_dir: str, force: bool = False) -> str:
    """Download + build a FoldSeek reference database (needed for novelty).

    Requires `foldseek` on `PATH` — call `install(tools=["foldseek"])` (or
    run `zevals setup`) first.

    Args:
        db_name:    One of FoldSeek's built-in presets — e.g. `"PDB"`,
            `"Alphafold/UniProt50-minimal"`. Run `foldseek databases` with no
            arguments (after installing the binary) to list every preset.
        output_dir: Persistent, large-storage directory to build the
            database under — NOT `/tmp` or purged scratch, these are
            multi-GB+ and meant to be reused across every future training
            run, not re-downloaded per session.
        force:      Rebuild even if the database prefix already exists.

    Returns:
        The built database's prefix path — pass this straight to a
        pipeline's `target_db` (e.g. `configs/proteins.yaml`'s `target_db`,
        or via a `ZevalSuite` override:
        `"pipeline.target_db=<returned path>"`).

    Raises:
        subprocess.CalledProcessError: if the build fails (e.g. a network
            error, or `foldseek` not installed yet).
    """
    scripts_dir = _resolve_scripts_dir()
    script_path = scripts_dir / "build_foldseek_db.sh"
    cmd = ["bash", str(script_path), db_name, output_dir]
    if force:
        cmd.append("--force")
    # Not captured — a database build can run for a long time (large
    # downloads), so progress output streams straight to the terminal
    # rather than being silently buffered until the process exits.
    subprocess.run(cmd, check=True)
    slug = db_name.replace("/", "_").replace(" ", "_")
    return str(Path(output_dir) / slug / "db")


def default_checkpoint_dir() -> Path:
    """The standard per-user checkpoint cache location every
    `oracle/inverse_folder/*.yaml` config's `weights_dir`/`checkpoint_path`
    default points at — `$XDG_CACHE_HOME/zevals/checkpoints` if set (so this
    naturally follows a redirected cache location, e.g. onto shared cluster
    storage), else `~/.cache/zevals/checkpoints`.
    """
    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg_cache) if xdg_cache else Path.home() / ".cache"
    return base / "zevals" / "checkpoints"


def download_checkpoint(
    name: str, output_dir: str | None = None, force: bool = False
) -> str:
    """Download a small, publicly-hosted inverse-folding checkpoint.

    Unlike `build_database()`, these are cheap (~6-30MB) and need no license,
    so this is safe to include in a routine `zevals setup` call — see
    `default_checkpoint_dir()`.

    Args:
        name:       One of `_CHECKPOINTS` — `"protein_mpnn_vanilla"`,
            `"protein_mpnn_ca"`, `"protein_mpnn_soluble"`, `"soluble_mpnn"`,
            `"ligand_mpnn"`, `"na_mpnn_design"`, `"na_mpnn_specificity"`.
        output_dir: Checkpoint cache root. Defaults to
            `default_checkpoint_dir()` — the same location every shipped
            `oracle/inverse_folder/*.yaml` config's `weights_dir`/
            `checkpoint_path` default already points at, so leaving this
            unset is what makes the default configs work with zero
            overrides once this has been run.
        force:      Re-download even if the file already exists.

    Returns:
        The resulting `weights_dir` (for ProteinMPNN-family checkpoints) or
        `checkpoint_path` (for LigandMPNN, a direct file) — pass straight to
        the corresponding oracle config field. Matches
        `default_checkpoint_dir()` unless `output_dir` was overridden.

    Raises:
        ValueError: if `name` isn't a known checkpoint.
        urllib.error.URLError: if the download fails (e.g. a network error).
    """
    if name not in _CHECKPOINTS:
        raise ValueError(
            f"Unknown checkpoint {name!r}. Available: {sorted(_CHECKPOINTS)}"
        )

    spec = _CHECKPOINTS[name]
    root = Path(output_dir) if output_dir is not None else default_checkpoint_dir()
    dest_dir = root / spec["subdir"]
    dest_dir.mkdir(parents=True, exist_ok=True)

    last_path = dest_dir
    for filename, url in spec["files"].items():
        dest_path = dest_dir / filename
        last_path = dest_path
        if dest_path.exists() and not force:
            print(
                f"{dest_path} already exists — skipping (pass force=True to re-download)."
            )
            continue
        print(f"Downloading {url} -> {dest_path} ...")
        # Download to a temp sibling and rename into place only on success —
        # `urlretrieve`-ing straight to `dest_path` would leave a truncated
        # file there on a network drop / interrupt, and the exists()-check
        # above would then treat that corrupt file as "already downloaded"
        # on every future call, silently skipping re-download forever short
        # of `--force`.
        tmp_path = dest_path.with_suffix(dest_path.suffix + ".part")
        try:
            urllib.request.urlretrieve(url, tmp_path)
            tmp_path.replace(dest_path)
        finally:
            tmp_path.unlink(missing_ok=True)

    return str(last_path) if spec["path_style"] == "file" else str(dest_dir)
