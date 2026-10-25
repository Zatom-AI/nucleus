"""Utilities for Open Babel bond sidecar records."""

from __future__ import annotations

import hashlib
import os
import pickle
import threading
from collections.abc import Mapping
from contextlib import suppress
from pathlib import Path
from typing import Any, Iterable

import biotite.structure as struc
import lmdb
import numpy as np
from atomworks.enums import ChainType

try:
    from atomworks.ml.datasets.loaders import create_ase_atoms_loader
except ImportError:
    from nucleus.utils.ase_loader import create_ase_atoms_loader

from nucleus.utils.atom_array import (
    _atomic_numbers_from_atom_array,
    _openbabel_obmol_from_atom_array,
    openbabel,
)

ATOMIC_NUMBERS_HASH_ALGORITHM = "sha256:int32-le"
BOND_SIDECAR_METADATA_KEY = b"__metadata__"

_ENVIRONMENT_LOCK = threading.Lock()
_ENVIRONMENTS: dict[tuple[int, str], tuple[lmdb.Environment, int]] = {}


def read_bond_sidecar_metadata(
    bond_sidecar_path: str | Path,
) -> dict[str, Any]:
    """Read and validate the metadata stored in a bond sidecar.

    Args:
        bond_sidecar_path: Single-file or directory-backed bond-sidecar LMDB.

    Returns:
        The decoded metadata dictionary.
    """
    path = Path(bond_sidecar_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Bond sidecar does not exist: {path}")

    environment = lmdb.open(
        str(path),
        subdir=path.is_dir(),
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        create=False,
        max_readers=512,
    )
    try:
        with environment.begin(write=False) as transaction:
            encoded_metadata = transaction.get(BOND_SIDECAR_METADATA_KEY)
        if encoded_metadata is None:
            raise ValueError(f"Bond sidecar has no metadata record: {path}")
        metadata = pickle.loads(  # noqa: S301 - trusted local sidecar artifact
            encoded_metadata
        )
        if not isinstance(metadata, dict):
            raise ValueError("Bond sidecar metadata must be a dictionary.")
        return metadata
    finally:
        environment.close()


def write_fragment_status_index_cache(
    *,
    example_ids: Iterable[str],
    dataset_length: int,
    bond_sidecar_path: str | Path,
    output_path: str | Path,
    data_id: str,
    has_disconnected_fragments: bool,
    expected_count: int | None = None,
    overwrite: bool = False,
) -> int:
    """Write zero-based dataset indices selected by bond-fragment status.

    The output format is the validated index-cache format consumed by
    ``subset_to_data_id_cache_path``. The iterable order must match dataset
    indexing exactly: the first example ID corresponds to index 0, and so on.

    Args:
        example_ids: Example IDs in dataset-index order.
        dataset_length: Expected number of IDs and cache-header dataset length.
        bond_sidecar_path: LMDB containing one bond record per example ID.
        output_path: Text cache to create atomically.
        data_id: Logical subset name written to the cache header.
        has_disconnected_fragments: Fragment-status value to select.
        expected_count: Optional exact selected-record count.
        overwrite: Whether to replace an existing output cache.

    Returns:
        The number of selected indices written.
    """
    if dataset_length <= 0:
        raise ValueError("dataset_length must be positive.")
    if not data_id:
        raise ValueError("data_id must be non-empty.")
    if expected_count is not None and expected_count < 0:
        raise ValueError("expected_count must be non-negative when provided.")

    sidecar_path = Path(bond_sidecar_path).expanduser()
    if not sidecar_path.exists():
        raise FileNotFoundError(f"Bond sidecar does not exist: {sidecar_path}")
    output_path = Path(output_path).expanduser()
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Index cache already exists: {output_path}. Pass overwrite=True to replace it."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    if temporary_path.exists():
        temporary_path.unlink()

    environment = lmdb.open(
        str(sidecar_path),
        subdir=sidecar_path.is_dir(),
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        create=False,
        max_readers=512,
    )
    seen_count = 0
    selected_count = 0
    try:
        with environment.begin(write=False) as transaction:
            encoded_metadata = transaction.get(BOND_SIDECAR_METADATA_KEY)
            if encoded_metadata is None:
                raise ValueError(f"Bond sidecar has no metadata record: {sidecar_path}")
            metadata = pickle.loads(  # noqa: S301 - trusted local artifact
                encoded_metadata
            )
            if not isinstance(metadata, dict) or metadata.get("complete") is not True:
                raise ValueError(f"Bond sidecar is not marked complete: {sidecar_path}")

            with temporary_path.open("w", encoding="utf-8") as handle:
                handle.write(f"# data_id={data_id} dataset_len={dataset_length}\n")
                for index, example_id in enumerate(example_ids):
                    if index >= dataset_length:
                        raise ValueError(
                            "example_ids contains more entries than dataset_length."
                        )
                    encoded_record = transaction.get(str(example_id).encode("utf-8"))
                    if encoded_record is None:
                        raise KeyError(
                            f"No bond sidecar record for example_id={example_id!r}."
                        )
                    record = pickle.loads(  # noqa: S301 - trusted local artifact
                        encoded_record
                    )
                    fragment_status = record.get("has_disconnected_fragments")
                    if not isinstance(fragment_status, (bool, np.bool_)):
                        raise ValueError(
                            "Invalid has_disconnected_fragments value for "
                            f"example_id={example_id!r}: {fragment_status!r}."
                        )
                    if bool(fragment_status) == has_disconnected_fragments:
                        handle.write(f"{index}\n")
                        selected_count += 1
                    seen_count += 1

        if seen_count != dataset_length:
            raise ValueError(
                f"example_ids contains {seen_count} entries, expected {dataset_length}."
            )
        if expected_count is not None and selected_count != expected_count:
            raise ValueError(
                f"Selected {selected_count} indices, expected {expected_count}."
            )
        temporary_path.replace(output_path)
    except Exception:
        with suppress(FileNotFoundError):
            temporary_path.unlink()
        raise
    finally:
        environment.close()

    return selected_count


def hash_atomic_numbers(atomic_numbers: Any) -> str:
    """Hash an ordered atomic-number sequence using a platform-stable encoding."""
    values = np.asarray(atomic_numbers, dtype="<i4")
    if values.ndim != 1:
        raise ValueError(
            f"Atomic numbers must be one-dimensional, got shape {values.shape}."
        )
    if np.any(values <= 0):
        raise ValueError("Atomic numbers must contain only positive values.")
    return hashlib.sha256(values.tobytes(order="C")).hexdigest()


def _normalized_bond_edges(ob_mol: Any) -> np.ndarray:
    """Extract sorted, zero-based, undirected ``[E, 2]`` Open Babel edges."""
    edges = np.asarray(
        [
            sorted((bond.GetBeginAtomIdx() - 1, bond.GetEndAtomIdx() - 1))
            for bond in openbabel.OBMolBondIter(ob_mol)
        ],
        dtype=np.int32,
    )
    if edges.size == 0:
        return np.empty((0, 2), dtype=np.int32)

    edges = edges.reshape(-1, 2)
    order = np.lexsort((edges[:, 1], edges[:, 0]))
    edges = edges[order]
    if len(edges) > 1 and np.any(np.all(edges[1:] == edges[:-1], axis=1)):
        raise ValueError("Open Babel produced duplicate bond edges.")
    return edges


def bond_graph_has_disconnected_fragments(num_atoms: int, bond_edges: Any) -> bool:
    """Return whether an undirected atom-bond graph has multiple components."""
    if num_atoms <= 0:
        raise ValueError(f"num_atoms must be positive, got {num_atoms}.")

    edges = np.asarray(bond_edges, dtype=np.int64)
    if edges.size == 0:
        return num_atoms > 1
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError(f"bond_edges must have shape [E, 2], got {edges.shape}.")
    if np.any(edges < 0) or np.any(edges >= num_atoms):
        raise ValueError("bond_edges contains an atom index outside num_atoms.")

    parents = np.arange(num_atoms, dtype=np.int64)

    def find(atom_index: int) -> int:
        while parents[atom_index] != atom_index:
            parents[atom_index] = parents[parents[atom_index]]
            atom_index = int(parents[atom_index])
        return atom_index

    for atom_a, atom_b in edges:
        root_a = find(int(atom_a))
        root_b = find(int(atom_b))
        if root_a != root_b:
            parents[root_b] = root_a

    return len({find(atom_index) for atom_index in range(num_atoms)}) > 1


def openbabel_bond_record(
    atom_array: Any,
    *,
    example_id: str,
) -> dict[str, Any]:
    """Create a bond-only sidecar record from an AtomArray-like object.

    Atom order is verified directly on the Open Babel molecule before any edges
    are returned. No RDKit sanitization is required for bond-only records.
    """
    if openbabel is None:
        raise ImportError(
            "Open Babel is required to construct bond sidecar records. "
            "Install the `openbabel` Python bindings."
        )

    num_atoms = len(np.asarray(atom_array.coord))
    atomic_numbers = _atomic_numbers_from_atom_array(atom_array, num_atoms)
    if atomic_numbers is None:
        raise ValueError(f"{example_id}: could not determine all atomic numbers.")

    ob_mol = _openbabel_obmol_from_atom_array(atom_array, example_id=example_id)
    if ob_mol is None:
        raise ValueError(f"{example_id}: Open Babel did not produce a valid molecule.")

    ob_mol_atomic_numbers = np.asarray(
        [atom.GetAtomicNum() for atom in openbabel.OBMolAtomIter(ob_mol)],
        dtype=np.int64,
    )
    if not np.array_equal(ob_mol_atomic_numbers, atomic_numbers):
        raise ValueError(
            f"{example_id}: atom order changed during Open Babel conversion."
        )

    bond_edges = _normalized_bond_edges(ob_mol)
    return {
        "bond_edges": bond_edges,
        "num_atoms": int(num_atoms),
        "atomic_numbers_hash": hash_atomic_numbers(atomic_numbers),
        "has_disconnected_fragments": bond_graph_has_disconnected_fragments(
            num_atoms, bond_edges
        ),
        "method": "openbabel",
        "method_version": str(openbabel.OBReleaseVersion()),
    }


def perceive_and_attach_bonds(atom_array: Any, *, example_id: str) -> bool:
    """Perceive bonds with the sidecar-construction logic and attach them.

    Args:
        atom_array: Fully resolved atom array whose coordinates define the graph.
        example_id: Identifier included in validation errors.

    Returns:
        Whether the perceived graph contains disconnected fragments.
    """
    record = openbabel_bond_record(atom_array, example_id=example_id)
    atom_array.bonds = struc.BondList(
        atom_array.array_length(),
        record["bond_edges"],
    )
    return bool(record["has_disconnected_fragments"])


def _can_perceive_ligand_bonds(atom_array, native_coordinates) -> bool:
    """Return whether native coordinates define a complete non-material ligand."""
    if native_coordinates is None:
        return False

    native_coordinates = np.asarray(native_coordinates)
    if native_coordinates.shape != atom_array.coord.shape:
        return False
    if not np.isfinite(native_coordinates).all():
        return False

    categories = set(atom_array.get_annotation_categories())
    if "is_ligand" in categories:
        if not np.asarray(atom_array.is_ligand, dtype=bool).all():
            return False
    elif not np.asarray(atom_array.res_name == "LIG", dtype=bool).all():
        # The generic inference parser assigns ``is_ligand`` later in the
        # transform pipeline. At this boundary, molecule-only inputs are the
        # non-polymer LIG residue produced by the contig parser.
        return False
    if (
        "is_polymer" in categories
        and np.asarray(atom_array.is_polymer, dtype=bool).any()
    ):
        return False
    if (
        "is_material" in categories
        and np.asarray(atom_array.is_material, dtype=bool).any()
    ):
        return False
    return True


def perceive_ligand_bonds_from_native_coordinates(
    atom_array: Any, native_coordinates: Any, *, example_id: str
) -> bool | None:
    """Attach bonds perceived from native coordinates to a molecule-only input.

    Inference zeroes generated coordinates before the pipeline runs, so bonds are
    perceived on a copy carrying the preserved native coordinates.

    Returns:
        Whether the perceived graph contains disconnected fragments, or ``None``
        when the input is not a fully resolved non-material ligand.
    """
    if not _can_perceive_ligand_bonds(atom_array, native_coordinates):
        return None
    perception_atom_array = atom_array.copy()
    perception_atom_array.coord = np.asarray(native_coordinates).copy()
    has_disconnected_fragments = perceive_and_attach_bonds(
        perception_atom_array,
        example_id=example_id,
    )
    atom_array.bonds = perception_atom_array.bonds
    return has_disconnected_fragments


def encode_bond_record(record: dict[str, Any]) -> bytes:
    """Serialize one sidecar record."""
    return pickle.dumps(record, protocol=pickle.HIGHEST_PROTOCOL)


def decode_bond_record(value: bytes) -> dict[str, Any]:
    """Deserialize and minimally validate one sidecar record."""
    record = pickle.loads(value)  # noqa: S301 - sidecars are trusted local artifacts
    required_keys = {
        "bond_edges",
        "num_atoms",
        "atomic_numbers_hash",
        "has_disconnected_fragments",
        "method",
        "method_version",
    }
    if not isinstance(record, dict) or not required_keys.issubset(record):
        raise ValueError("Invalid bond sidecar record.")

    bond_edges = np.asarray(record["bond_edges"])
    if bond_edges.dtype != np.dtype(np.int32) or bond_edges.ndim != 2:
        raise ValueError("bond_edges must be an int32 [E, 2] array.")
    if bond_edges.shape[1] != 2:
        raise ValueError("bond_edges must have shape [E, 2].")
    if not isinstance(record["has_disconnected_fragments"], (bool, np.bool_)):
        raise ValueError("has_disconnected_fragments must be a boolean.")
    return record


class BondSidecarASEAtomsLoader:
    """Load ASE atoms and attach bonds stored in a sidecar LMDB.

    The sidecar is opened lazily in each data-loader worker. Records are looked
    up using the ``example_id`` emitted by ``ASELMDBDataset`` and validated
    against the loaded atom count and atom ordering before a Biotite
    :class:`BondList` is constructed.
    """

    def __init__(
        self,
        *,
        bond_sidecar_path: str | Path,
        require_complete_sidecar: bool = True,
        chain_id: str = "A",
        res_name: str = "LIG",
        res_id: int = 1,
        chain_type: ChainType | str | int = ChainType.NON_POLYMER,
        is_polymer: bool = False,
        include_chain_info: bool = True,
        keep_ase_atoms: bool = True,
    ):
        self.bond_sidecar_path = Path(bond_sidecar_path).expanduser()
        self.require_complete_sidecar = require_complete_sidecar
        self._ase_atoms_loader = create_ase_atoms_loader(
            chain_id=chain_id,
            res_name=res_name,
            res_id=res_id,
            chain_type=chain_type,
            is_polymer=is_polymer,
            include_chain_info=include_chain_info,
            keep_ase_atoms=keep_ase_atoms,
        )
        self._environment: lmdb.Environment | None = None
        self._environment_pid: int | None = None
        self._environment_key: tuple[int, str] | None = None

    def _validate_sidecar_metadata(self, environment: lmdb.Environment) -> None:
        with environment.begin(write=False) as transaction:
            encoded_metadata = transaction.get(BOND_SIDECAR_METADATA_KEY)
        if encoded_metadata is None:
            raise ValueError(
                f"Bond sidecar has no metadata record: {self.bond_sidecar_path}"
            )

        metadata = pickle.loads(  # noqa: S301 - trusted local sidecar artifact
            encoded_metadata
        )
        if not isinstance(metadata, dict):
            raise ValueError("Bond sidecar metadata must be a dictionary.")
        if (
            metadata.get("atomic_numbers_hash_algorithm")
            != ATOMIC_NUMBERS_HASH_ALGORITHM
        ):
            raise ValueError(
                "Bond sidecar uses an unexpected atomic-number hash algorithm: "
                f"{metadata.get('atomic_numbers_hash_algorithm')!r}."
            )
        if self.require_complete_sidecar and metadata.get("complete") is not True:
            raise ValueError(
                f"Bond sidecar is not marked complete: {self.bond_sidecar_path}"
            )

    def _open_environment(self) -> lmdb.Environment:
        path = self.bond_sidecar_path
        if not path.exists():
            raise FileNotFoundError(f"Bond sidecar does not exist: {path}")

        environment = lmdb.open(
            str(path),
            subdir=path.is_dir(),
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            create=False,
            max_readers=512,
        )
        try:
            self._validate_sidecar_metadata(environment)
        except Exception:
            environment.close()
            raise
        return environment

    def _get_environment(self) -> lmdb.Environment:
        process_id = os.getpid()
        if self._environment is not None and self._environment_pid == process_id:
            return self._environment

        # python-lmdb requires all users of one file in a process to share the
        # same Environment object. This also permits train and validation
        # datasets to read the same sidecar in one process.
        environment_key = (process_id, str(self.bond_sidecar_path.resolve()))
        with _ENVIRONMENT_LOCK:
            inherited_keys = [key for key in _ENVIRONMENTS if key[0] != process_id]
            for inherited_key in inherited_keys:
                inherited_environment, _ = _ENVIRONMENTS.pop(inherited_key)
                inherited_environment.close()

            cached = _ENVIRONMENTS.get(environment_key)
            if cached is None:
                environment = self._open_environment()
                _ENVIRONMENTS[environment_key] = (environment, 1)
            else:
                environment, reference_count = cached
                _ENVIRONMENTS[environment_key] = (
                    environment,
                    reference_count + 1,
                )

        self._environment = environment
        self._environment_pid = process_id
        self._environment_key = environment_key
        return self._environment

    def _load_bond_record(self, example_id: str) -> dict[str, Any]:
        environment = self._get_environment()
        with environment.begin(write=False) as transaction:
            encoded_record = transaction.get(example_id.encode("utf-8"))
        if encoded_record is None:
            raise KeyError(
                f"No bond sidecar record for example_id={example_id!r} in "
                f"{self.bond_sidecar_path}."
            )
        return decode_bond_record(encoded_record)

    def fragment_status(
        self,
        *,
        example_id: str,
        atomic_numbers: Any | None = None,
    ) -> bool:
        """Return fragment status, optionally validating the sample identity.

        Args:
            example_id: Sidecar LMDB key to load.
            atomic_numbers: Optional ordered atomic numbers for validating that
                the caller and sidecar refer to the same structure.

        Returns:
            Whether the sidecar record contains disconnected fragments.
        """
        record = self._load_bond_record(example_id)
        if atomic_numbers is not None:
            values = np.asarray(atomic_numbers)
            if values.ndim != 1:
                raise ValueError(
                    f"{example_id}: atomic numbers must be one-dimensional, "
                    f"got shape {values.shape}."
                )
            if record["num_atoms"] != len(values):
                raise ValueError(
                    f"{example_id}: sidecar contains {record['num_atoms']} atoms, "
                    f"but the sample contains {len(values)}."
                )
            if record["atomic_numbers_hash"] != hash_atomic_numbers(values):
                raise ValueError(
                    f"{example_id}: sidecar atomic-number hash does not match "
                    "the sample ordering."
                )
        return bool(record["has_disconnected_fragments"])

    @staticmethod
    def _validated_bond_edges(
        atom_array: Any,
        record: dict[str, Any],
        *,
        example_id: str,
    ) -> np.ndarray:
        num_atoms = atom_array.array_length()
        if record["num_atoms"] != num_atoms:
            raise ValueError(
                f"{example_id}: sidecar contains {record['num_atoms']} atoms, "
                f"but the loaded atom array contains {num_atoms}."
            )

        atomic_numbers = _atomic_numbers_from_atom_array(atom_array, num_atoms)
        if atomic_numbers is None:
            raise ValueError(f"{example_id}: could not determine all atomic numbers.")
        if record["atomic_numbers_hash"] != hash_atomic_numbers(atomic_numbers):
            raise ValueError(
                f"{example_id}: sidecar atomic-number hash does not match the "
                "loaded atom ordering."
            )

        bond_edges = np.asarray(record["bond_edges"], dtype=np.int32)
        if bond_edges.size:
            if np.any(bond_edges < 0) or np.any(bond_edges >= num_atoms):
                raise ValueError(f"{example_id}: sidecar bond index is out of range.")
            if np.any(bond_edges[:, 0] == bond_edges[:, 1]):
                raise ValueError(f"{example_id}: sidecar contains a self-bond.")
        return bond_edges

    def attach_bonds(self, atom_array: Any, *, example_id: str) -> bool:
        """Attach one validated sidecar graph to an existing atom array.

        Args:
            atom_array: Atom array whose atom count and element ordering must match
                the sidecar record.
            example_id: Sidecar LMDB key to load.

        Returns:
            Whether the attached graph contains disconnected fragments.
        """
        record = self._load_bond_record(example_id)
        bond_edges = self._validated_bond_edges(
            atom_array,
            record,
            example_id=example_id,
        )
        atom_array.bonds = struc.BondList(atom_array.array_length(), bond_edges)
        return bool(record["has_disconnected_fragments"])

    def __call__(self, raw_data: Any) -> dict[str, Any]:
        """Load one ASE record and attach its validated sidecar bonds."""
        if not isinstance(raw_data, Mapping) or "example_id" not in raw_data:
            raise TypeError(
                "Bond-aware ASE loading requires an ASELMDBDataset record with "
                "an `example_id`; configure return_type='record'."
            )

        example_id = str(raw_data["example_id"])
        data = self._ase_atoms_loader(raw_data)
        data["has_disconnected_fragments"] = self.attach_bonds(
            data["atom_array"],
            example_id=example_id,
        )
        return data

    def close(self) -> None:
        """Close this process's sidecar LMDB handle, if one is open."""
        if self._environment is None:
            return

        with _ENVIRONMENT_LOCK:
            cached = _ENVIRONMENTS.get(self._environment_key)
            if cached is not None and cached[0] is self._environment:
                environment, reference_count = cached
                if reference_count == 1:
                    environment.close()
                    del _ENVIRONMENTS[self._environment_key]
                else:
                    _ENVIRONMENTS[self._environment_key] = (
                        environment,
                        reference_count - 1,
                    )
            else:
                self._environment.close()

        self._environment = None
        self._environment_pid = None
        self._environment_key = None

    def __getstate__(self) -> dict[str, Any]:
        """Drop live LMDB handles when pickled for data-loader workers."""
        state = self.__dict__.copy()
        state["_environment"] = None
        state["_environment_pid"] = None
        state["_environment_key"] = None
        return state

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def create_bond_sidecar_ase_atoms_loader(
    **kwargs: Any,
) -> BondSidecarASEAtomsLoader:
    """Create a picklable ASE loader backed by a bond sidecar LMDB."""
    return BondSidecarASEAtomsLoader(**kwargs)


class BondSidecarFragmentStatusLoader:
    """Read per-example fragment status without constructing an atom array."""

    def __init__(
        self,
        *,
        bond_sidecar_path: str | Path,
        require_complete_sidecar: bool = True,
    ) -> None:
        self._record_loader = BondSidecarASEAtomsLoader(
            bond_sidecar_path=bond_sidecar_path,
            require_complete_sidecar=require_complete_sidecar,
        )

    def __call__(self, *, example_id: str, atomic_numbers: Any) -> bool:
        """Return validated fragment status for one ordered atomic-number list."""
        return self._record_loader.fragment_status(
            example_id=example_id,
            atomic_numbers=atomic_numbers,
        )

    def close(self) -> None:
        """Close this process's sidecar handle, if one is open."""
        self._record_loader.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
