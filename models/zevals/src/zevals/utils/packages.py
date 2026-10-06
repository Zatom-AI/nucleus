def load_rdkit_chem():
    """Dynamically import the RDKit Chem module, or raise an informative error if it's not available."""
    try:
        from rdkit import Chem  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "Molecule-generation metrics require rdkit. "
            "Install the `zevals` molecule extras or add `rdkit`."
        ) from exc

    return Chem


def load_posebusters():
    """Dynamically import the PoseBusters class, or raise an informative error if it's not available."""
    try:
        from posebusters import PoseBusters  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "PoseBusters is not installed. Run: pip install posebusters"
        ) from exc

    return PoseBusters


def load_fpsim2_engine():
    """Dynamically import FPSim2, or raise an informative error."""
    try:
        from FPSim2 import FPSim2Engine  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "FPSim2 molecule novelty requires fpsim2. "
            "Install the `zevals` molecule extras or add `fpsim2`."
        ) from exc

    return FPSim2Engine
