import os
import subprocess
from pathlib import Path

import typer
from dotenv import load_dotenv
from hydra import compose, initialize_config_dir

app = typer.Typer()
load_dotenv(override=True)


def _get_config_path() -> str:
    """Locate packaged configs and initialize the development project root."""
    dev_config_path = Path(__file__).parent.parent.parent / "configs"
    if dev_config_path.exists():
        os.environ.setdefault("PROJECT_ROOT", str(dev_config_path.parents[2]))
        return str(dev_config_path)
    os.environ.setdefault("PROJECT_ROOT", str(Path.cwd()))
    return str(Path(__file__).parent / "configs")


@app.command(
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True}
)
def design(ctx: typer.Context):
    """Run design using hydra config overrides and input files."""
    # Find the Zatom-2 configs directory relative to this file
    # Development: models/zatom2/src/zatom2/cli.py -> models/zatom2/configs/
    # Installed: site-packages/zatom2/cli.py -> site-packages/zatom2/configs/

    # Try development location first
    config_path = _get_config_path()

    # Get all arguments
    args = ctx.params.get("args", []) + ctx.args
    args = [a for a in args if a not in ["design", "fold"]]

    # Ensure we have at least a default inference_engine if not specified
    has_inference_engine = any(arg.startswith("inference_engine=") for arg in args)
    if not has_inference_engine:
        args.append("inference_engine=zatom2")

    with initialize_config_dir(config_dir=config_path, version_base="1.3"):
        cfg = compose(config_name="inference", overrides=args)

        # Lazy import to avoid loading heavy dependencies at CLI startup
        from foundry.utils.logging import suppress_warnings

        from zatom2.run_inference import run_inference

        with suppress_warnings(is_inference=True):
            run_inference(cfg)


@app.command(
    "evaluate-mlip",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def evaluate_mlip(ctx: typer.Context):
    """Evaluate a checkpoint on labeled OMol25/OMat24 validation data."""
    config_path = _get_config_path()
    args = ctx.params.get("args", []) + ctx.args
    args = [arg for arg in args if arg not in ["evaluate-mlip", "evaluate_mlip"]]

    with initialize_config_dir(config_dir=config_path, version_base="1.3"):
        cfg = compose(config_name="mlip_evaluation", overrides=args)

        from foundry.utils.logging import suppress_warnings

        from zatom2.mlip_evaluation import run_mlip_evaluation

        with suppress_warnings(is_inference=True):
            run_mlip_evaluation(cfg)


@app.command(
    "evaluate-generation",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def evaluate_generation(ctx: typer.Context):
    """Sample and evaluate 1,000 OMol25 or OMat24 structures."""
    config_path = _get_config_path()
    args = ctx.params.get("args", []) + ctx.args
    args = [
        arg for arg in args if arg not in ["evaluate-generation", "evaluate_generation"]
    ]

    with initialize_config_dir(config_dir=config_path, version_base="1.3"):
        cfg = compose(config_name="generation_evaluation", overrides=args)

        from foundry.utils.logging import suppress_warnings

        from zatom2.generation_evaluation import run_generation_evaluation

        with suppress_warnings(is_inference=True):
            run_generation_evaluation(cfg)


@app.command(
    "evaluate-table2",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def evaluate_table2(ctx: typer.Context):
    """Evaluate a Table-2 QM9+MP20 checkpoint with frozen benchmark inputs."""
    config_path = _get_config_path()
    args = ctx.params.get("args", []) + ctx.args
    args = [arg for arg in args if arg not in ["evaluate-table2", "evaluate_table2"]]

    with initialize_config_dir(config_dir=config_path, version_base="1.3"):
        cfg = compose(config_name="table2_evaluation", overrides=args)

        from foundry.utils.logging import suppress_warnings

        from zatom2.table2_evaluation import run_table2_evaluation

        with suppress_warnings(is_inference=True):
            try:
                run_table2_evaluation(cfg)
            except (TimeoutError, subprocess.TimeoutExpired) as error:
                typer.echo(f"Evaluation paused; resume the same out_dir: {error}")
                raise typer.Exit(code=75) from error


if __name__ == "__main__":
    app()
