"""PoseBusters pose validator oracle.

Install via:
  pip install posebusters

See https://github.com/maabuu/posebusters for documentation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from zevals.oracles.base import BasePoseValidator, PoseValidationResult


class PoseBustersValidator(BasePoseValidator):
    """Wraps PoseBusters for geometry, clash, and stereochemistry checks.

    Hydra config: oracle/pose_validator/posebusters.yaml
      _target_:        zevals.oracles.pose_validators.posebusters.PoseBustersValidator
      table:           dock     # "dock" | "gen" | "redock"
      add_hydrogens:   true
      full_report:     true     # return per-check breakdown in `details`
    """

    def __init__(
        self,
        table: Literal["dock", "gen", "redock"] = "dock",
        add_hydrogens: bool = True,
        full_report: bool = True,
        **kwargs: Any,
    ) -> None:
        self.table = table
        self.add_hydrogens = add_hydrogens
        self.full_report = full_report

    def validate(
        self,
        ligand_path: Path,
        protein_path: Path | None = None,
    ) -> PoseValidationResult:
        try:
            from posebusters import PoseBusters  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "PoseBusters is not installed. Run: pip install posebusters"
            ) from exc

        buster = PoseBusters(table=self.table, top_n=None)

        if protein_path is not None:
            df = buster.bust(
                str(ligand_path), str(protein_path), full_report=self.full_report
            )
        else:
            df = buster.bust(str(ligand_path), full_report=self.full_report)

        passes = bool(df["passes_all"].all())
        details: dict[str, Any] = df.to_dict(orient="list") if self.full_report else {}

        return PoseValidationResult(passes=passes, details=details)
