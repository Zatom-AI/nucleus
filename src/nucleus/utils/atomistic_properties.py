"""Versioned normalization statistics for atomistic property prediction."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

ATOMISTIC_PROPERTY_STATISTICS_SCHEMA_VERSION = 1
ENERGY_UNIT = "eV"
FORCE_UNIT = "eV/angstrom"


@dataclass(frozen=True)
class DomainPropertyStatistics:
    """Normalization statistics for one electronic-structure domain."""

    name: str
    domain_id: int
    source_split_ids: tuple[str, ...]
    elemental_references_eV: Mapping[int, float]
    residual_energy_per_atom_mean_eV: float
    residual_energy_per_atom_scale_eV: float
    force_rms_eV_per_angstrom: float
    n_systems: int | None = None
    n_atoms: int | None = None

    @classmethod
    def from_dict(cls, name: str, value: Mapping[str, Any]):
        """Validate and construct statistics from a JSON domain mapping."""
        energy_unit = value.get("energy_unit")
        force_unit = value.get("force_unit")
        if energy_unit != ENERGY_UNIT:
            raise ValueError(
                f"Domain {name!r} must use energy_unit={ENERGY_UNIT!r}; "
                f"got {energy_unit!r}."
            )
        if force_unit != FORCE_UNIT:
            raise ValueError(
                f"Domain {name!r} must use force_unit={FORCE_UNIT!r}; "
                f"got {force_unit!r}."
            )

        source_split_ids = tuple(str(v) for v in value.get("source_split_ids", ()))
        if not source_split_ids:
            raise ValueError(
                f"Domain {name!r} must identify at least one source split."
            )

        raw_references = value.get("elemental_references_eV")
        if not isinstance(raw_references, Mapping) or not raw_references:
            raise ValueError(
                f"Domain {name!r} must contain non-empty elemental_references_eV."
            )
        references = {
            int(z): float(reference) for z, reference in raw_references.items()
        }
        if any(z <= 0 for z in references):
            raise ValueError(f"Domain {name!r} contains a non-positive atomic number.")
        if not all(torch.isfinite(torch.tensor(v)) for v in references.values()):
            raise ValueError(
                f"Domain {name!r} contains non-finite elemental references."
            )

        energy_mean = float(value["residual_energy_per_atom_mean_eV"])
        energy_scale = float(value["residual_energy_per_atom_scale_eV"])
        force_scale = float(value["force_rms_eV_per_angstrom"])
        scalars = torch.tensor([energy_mean, energy_scale, force_scale])
        if not torch.isfinite(scalars).all():
            raise ValueError(
                f"Domain {name!r} contains non-finite normalization values."
            )
        if energy_scale <= 0 or force_scale <= 0:
            raise ValueError(f"Domain {name!r} normalization scales must be positive.")

        return cls(
            name=name,
            domain_id=int(value["domain_id"]),
            source_split_ids=source_split_ids,
            elemental_references_eV=references,
            residual_energy_per_atom_mean_eV=energy_mean,
            residual_energy_per_atom_scale_eV=energy_scale,
            force_rms_eV_per_angstrom=force_scale,
            n_systems=(
                None if value.get("n_systems") is None else int(value["n_systems"])
            ),
            n_atoms=None if value.get("n_atoms") is None else int(value["n_atoms"]),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical JSON representation."""
        value: dict[str, Any] = {
            "domain_id": self.domain_id,
            "source_split_ids": list(self.source_split_ids),
            "energy_unit": ENERGY_UNIT,
            "force_unit": FORCE_UNIT,
            "elemental_references_eV": {
                str(z): reference
                for z, reference in sorted(self.elemental_references_eV.items())
            },
            "residual_energy_per_atom_mean_eV": (self.residual_energy_per_atom_mean_eV),
            "residual_energy_per_atom_scale_eV": (
                self.residual_energy_per_atom_scale_eV
            ),
            "force_rms_eV_per_angstrom": self.force_rms_eV_per_angstrom,
        }
        if self.n_systems is not None:
            value["n_systems"] = self.n_systems
        if self.n_atoms is not None:
            value["n_atoms"] = self.n_atoms
        return value


@dataclass(frozen=True)
class AtomisticPropertyStatistics:
    """Validated, versioned collection of per-domain property statistics."""

    domains: tuple[DomainPropertyStatistics, ...]
    schema_version: int = ATOMISTIC_PROPERTY_STATISTICS_SCHEMA_VERSION

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]):
        """Construct statistics from their serialized mapping."""
        schema_version = int(value.get("schema_version", -1))
        if schema_version != ATOMISTIC_PROPERTY_STATISTICS_SCHEMA_VERSION:
            raise ValueError(
                "Unsupported atomistic property statistics schema_version "
                f"{schema_version}; expected "
                f"{ATOMISTIC_PROPERTY_STATISTICS_SCHEMA_VERSION}."
            )
        raw_domains = value.get("domains")
        if not isinstance(raw_domains, Mapping) or not raw_domains:
            raise ValueError(
                "Statistics artifact must contain a non-empty domains map."
            )

        domains = tuple(
            sorted(
                (
                    DomainPropertyStatistics.from_dict(str(name), domain)
                    for name, domain in raw_domains.items()
                ),
                key=lambda domain: domain.domain_id,
            )
        )
        ids = tuple(domain.domain_id for domain in domains)
        expected_ids = tuple(range(len(domains)))
        if ids != expected_ids:
            raise ValueError(
                "Domain IDs must be unique and contiguous starting at zero; "
                f"got {ids}."
            )
        return cls(domains=domains, schema_version=schema_version)

    @classmethod
    def load(cls, path: str | Path):
        """Load and validate a JSON artifact."""
        artifact_path = Path(path)
        with artifact_path.open(encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    @classmethod
    def from_values(
        cls,
        *,
        domain_names: Sequence[str],
        energy_mean: Sequence[float],
        energy_scale: Sequence[float],
        force_scale: Sequence[float],
        elemental_references: Sequence[Sequence[float]],
    ):
        """Build in-memory statistics from dense configuration values.

        This constructor is intended for tests and programmatic model construction.
        Production training should use a versioned JSON artifact tied to explicit
        training split identifiers.
        """
        n_domains = len(domain_names)
        if not (
            len(energy_mean)
            == len(energy_scale)
            == len(force_scale)
            == len(elemental_references)
            == n_domains
        ):
            raise ValueError("All per-domain statistics must have the same length.")

        domains = []
        for domain_id, name in enumerate(domain_names):
            references = {
                atomic_number: float(reference)
                for atomic_number, reference in enumerate(
                    elemental_references[domain_id]
                )
                if atomic_number > 0
            }
            domains.append(
                DomainPropertyStatistics.from_dict(
                    str(name),
                    {
                        "domain_id": domain_id,
                        "source_split_ids": ["in-memory"],
                        "energy_unit": ENERGY_UNIT,
                        "force_unit": FORCE_UNIT,
                        "elemental_references_eV": references,
                        "residual_energy_per_atom_mean_eV": energy_mean[domain_id],
                        "residual_energy_per_atom_scale_eV": energy_scale[domain_id],
                        "force_rms_eV_per_angstrom": force_scale[domain_id],
                    },
                )
            )
        return cls(domains=tuple(domains))

    @property
    def domain_names(self) -> tuple[str, ...]:
        """Domain names ordered by integer domain identifier."""
        return tuple(domain.name for domain in self.domains)

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical JSON representation."""
        return {
            "schema_version": self.schema_version,
            "domains": {domain.name: domain.to_dict() for domain in self.domains},
        }

    def save(self, path: str | Path) -> None:
        """Write the canonical JSON artifact."""
        artifact_path = Path(path)
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        with artifact_path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, sort_keys=True)
            handle.write("\n")

    def tensors(
        self,
        *,
        max_atomic_number: int = 127,
        dtype: torch.dtype = torch.float32,
    ) -> dict[str, torch.Tensor]:
        """Convert statistics to dense tensors suitable for registered buffers."""
        n_domains = len(self.domains)
        references = torch.zeros((n_domains, max_atomic_number + 1), dtype=dtype)
        reference_valid = torch.zeros_like(references, dtype=torch.bool)
        energy_mean = torch.empty(n_domains, dtype=dtype)
        energy_scale = torch.empty(n_domains, dtype=dtype)
        force_scale = torch.empty(n_domains, dtype=dtype)

        for domain in self.domains:
            energy_mean[domain.domain_id] = domain.residual_energy_per_atom_mean_eV
            energy_scale[domain.domain_id] = domain.residual_energy_per_atom_scale_eV
            force_scale[domain.domain_id] = domain.force_rms_eV_per_angstrom
            for atomic_number, reference in domain.elemental_references_eV.items():
                if atomic_number > max_atomic_number:
                    raise ValueError(
                        f"Atomic number {atomic_number} exceeds configured maximum "
                        f"{max_atomic_number}."
                    )
                references[domain.domain_id, atomic_number] = reference
                reference_valid[domain.domain_id, atomic_number] = True

        return {
            "elemental_references": references,
            "elemental_reference_valid": reference_valid,
            "energy_mean": energy_mean,
            "energy_scale": energy_scale,
            "force_scale": force_scale,
        }
