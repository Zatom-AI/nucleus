"""Private restart hooks copied into the isolated, pinned LeMat environment.

This module deliberately imports no zevals code. It is installed beside the
reference-pin startup shim and runs in upstream's Python 3.11 environment.
Cache payloads are local, generated Python objects, not imported user data.
"""

from __future__ import annotations

import fcntl
import functools
import hashlib
import json
import math
import os
import pickle
import tempfile
from pathlib import Path
from typing import Any

CACHE_VERSION = 1


class CacheIntegrityError(BaseException):
    """Stop the runner; upstream must not reinterpret corrupt cache as invalidity."""


def _encoded(value: Any) -> bytes:
    from monty.json import MontyEncoder

    return json.dumps(
        value, cls=MontyEncoder, sort_keys=True, separators=(",", ":")
    ).encode()


def _identity_bytes(value: Any) -> bytes:
    # The pinned validity preprocessor adds the current wall-clock time to every
    # Structure. It has no scientific meaning and otherwise invalidates every
    # cache key on each allocation. Remove only this audited metadata field,
    # including copies carried by nested relaxed Structures.
    def strip_timestamp(item):
        if isinstance(item, dict):
            properties = item.get("properties")
            if isinstance(properties, dict):
                properties.pop("validity_timestamp", None)
            return {key: strip_timestamp(value) for key, value in item.items()}
        if isinstance(item, list):
            return [strip_timestamp(value) for value in item]
        return item

    return _encoded(strip_timestamp(json.loads(_encoded(value))))


def _atomic_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".writing-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def cached_call(namespace: str, identity: Any, compute: Any, *, valid=None) -> Any:
    """Memoize one complete calculation with an atomic, hash-verified payload."""
    cache_dir = Path(os.environ["ZEVALS_LEMAT_RESUME_CACHE"])
    contract = os.environ["ZEVALS_LEMAT_RESUME_CONTRACT"]
    key = hashlib.sha256(
        _identity_bytes([CACHE_VERSION, contract, identity])
    ).hexdigest()
    record_path = cache_dir / namespace / f"{key}.json"
    record_path.parent.mkdir(parents=True, exist_ok=True)
    # Duplicate generated structures can be evaluated by different upstream
    # workers. Serialize matching entries so payload/metadata writes cannot mix.
    with record_path.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _cached_locked(record_path, key, contract, namespace, compute, valid)


def _cached_locked(record_path, key, contract, namespace, compute, valid):
    if record_path.exists():
        try:
            record = json.loads(record_path.read_text())
            if record.get("key") != key or record.get("contract") != contract:
                raise CacheIntegrityError(
                    f"LeMat resume cache identity mismatch: {record_path}"
                )
            digest = record.get("sha256")
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise CacheIntegrityError(f"Invalid LeMat cache digest: {record_path}")
            payload = record_path.with_name(f"{key}.{digest}.pickle").read_bytes()
            if hashlib.sha256(payload).hexdigest() != digest:
                raise CacheIntegrityError(
                    f"LeMat resume cache checksum mismatch: {record_path}"
                )
            # Only locally generated and digest-verified cache entries are accepted.
            result = pickle.loads(payload)
        except Exception as exc:
            raise CacheIntegrityError(
                f"Unreadable LeMat resume cache: {record_path}"
            ) from exc
        print(f"ZEVALS_LEMAT_RESUME hit {namespace} {key}", flush=True)
        return result
    result = compute()
    # Exceptions and explicit upstream failure dictionaries are never cached.
    if isinstance(result, dict) and "error" in result:
        return result
    if valid is not None and not valid(result):
        return result
    _publish(record_path, key, contract, result)
    print(f"ZEVALS_LEMAT_RESUME saved {namespace} {key}", flush=True)
    return result


def _publish(record_path, key, contract, result):
    payload = pickle.dumps(result, protocol=5)
    digest = hashlib.sha256(payload).hexdigest()
    # Immutable, content-addressed payloads keep the previous commit readable
    # if a later comparison-row update is killed before metadata replacement.
    _atomic_bytes(record_path.with_name(f"{key}.{digest}.pickle"), payload)
    _atomic_bytes(
        record_path,
        _encoded(
            {
                "key": key,
                "contract": contract,
                "sha256": digest,
            }
        ),
    )


class _ComparisonRows:
    """Private matcher proxy retaining upstream's exact comparison ordering."""

    def __init__(self, matcher, structures, namespace, configuration):
        self.matcher = matcher
        self.namespace = namespace
        self.contract = os.environ["ZEVALS_LEMAT_RESUME_CONTRACT"]
        self.dataset = hashlib.sha256(
            _identity_bytes([configuration, [item.as_dict() for item in structures]])
        ).hexdigest()
        self.indices = {id(item): index for index, item in enumerate(structures)}
        self.row_index = None
        self.row = {}
        self.dirty = False
        self.record_path = None
        self.key = None

    def __getattr__(self, name):
        return getattr(self.matcher, name)

    def flush(self):
        if self.dirty:
            _publish(self.record_path, self.key, self.contract, self.row)
            self.dirty = False

    def is_equivalent(self, first, second):
        first_index, second_index = self.indices[id(first)], self.indices[id(second)]
        if first_index != self.row_index:
            self.flush()
            self.row_index = first_index
            self.key = hashlib.sha256(
                _identity_bytes(
                    [CACHE_VERSION, self.contract, self.dataset, first_index]
                )
            ).hexdigest()
            self.record_path = (
                Path(os.environ["ZEVALS_LEMAT_RESUME_CACHE"])
                / self.namespace
                / f"{self.key}.json"
            )
            self.row = _cached_locked(
                self.record_path,
                self.key,
                self.contract,
                self.namespace,
                lambda: {},
                lambda value: False,
            )
        if second_index not in self.row:
            # Exceptions preserve upstream's error handling and are never cached.
            self.row[second_index] = self.matcher.is_equivalent(first, second)
            self.dirty = True
        return self.row[second_index]


def install() -> None:
    """Patch only pure oracle calls and evaluations of the entire input set."""
    if os.environ.get("ZEVALS_LEMAT_INTERPRETER_ROLE") != "benchmark":
        return
    if not os.environ.get("ZEVALS_LEMAT_RESUME_CACHE"):
        return
    import lemat_genbench.preprocess.multi_mlip_preprocess as multi_mlip
    from lemat_genbench.benchmarks.distribution_benchmark import (
        DistributionBenchmark,
    )
    from lemat_genbench.benchmarks.diversity_benchmark import DiversityBenchmark
    from lemat_genbench.benchmarks.hhi_benchmark import HHIBenchmark
    from lemat_genbench.benchmarks.multi_mlip_stability_benchmark import (
        StabilityBenchmark,
    )
    from lemat_genbench.benchmarks.novelty_benchmark import NoveltyBenchmark
    from lemat_genbench.benchmarks.sun_benchmark import SUNBenchmark
    from lemat_genbench.benchmarks.uniqueness_benchmark import UniquenessBenchmark
    from lemat_genbench.metrics.novelty_metric import NoveltyMetric
    from lemat_genbench.metrics.sun_metric import SUNMetric
    from lemat_genbench.metrics.uniqueness_metric import UniquenessMetric

    original_single = multi_mlip._process_single_mlip

    @functools.wraps(original_single)
    def single(structure, calculator, mlip_name, *args, **kwargs):
        # Upstream catches some oracle failures and returns partial dictionaries.
        # Retry these on the next allocation instead of freezing missing coverage.
        import inspect

        bound = inspect.signature(original_single).bind(
            structure, calculator, mlip_name, *args, **kwargs
        )
        flags = bound.arguments
        required = ["energy", "forces"]
        if flags["calculate_formation_energy"]:
            required.append("formation_energy")
        if flags["calculate_energy_above_hull"]:
            required.append("e_above_hull")
        if flags["extract_embeddings"]:
            required.extend(["node_embeddings", "graph_embedding"])
        if flags["relax_structures"]:
            required.extend(["relaxed_structure", "relaxation_energy"])
            if flags["calculate_formation_energy"]:
                required.append("relaxed_formation_energy")
            if flags["calculate_energy_above_hull"]:
                required.append("relaxed_e_above_hull")

        def complete(result):
            import numpy as np

            for key in required:
                if result.get(key) is None:
                    return False
                if key != "relaxed_structure" and not np.isfinite(result[key]).all():
                    return False
            return True

        return cached_call(
            f"oracle-{mlip_name}",
            [structure.as_dict(), mlip_name, args, kwargs],
            lambda: original_single(structure, calculator, mlip_name, *args, **kwargs),
            valid=complete,
        )

    multi_mlip._process_single_mlip = single

    original_reference = NoveltyMetric._load_reference_dataset

    @functools.wraps(original_reference)
    def reference(self):
        information = original_reference(self)
        information["_zevals_resume_identity"] = {
            "configuration": vars(self.config),
            "dataset_fingerprint": getattr(
                information.get("dataset"), "_fingerprint", None
            ),
        }
        return information

    NoveltyMetric._load_reference_dataset = reference
    original_novelty = NoveltyMetric.compute_structure

    @functools.wraps(original_novelty)
    def novelty(structure, dataset_information, fingerprinter, verbose=False):
        if "_zevals_resume_identity" not in dataset_information:
            raise CacheIntegrityError(
                "Novelty reference was not prepared by resume hook"
            )
        return cached_call(
            "novelty-structure",
            [structure.as_dict(), dataset_information["_zevals_resume_identity"]],
            lambda: original_novelty(
                structure, dataset_information, fingerprinter, verbose
            ),
            valid=lambda value: math.isfinite(float(value)),
        )

    NoveltyMetric.compute_structure = staticmethod(novelty)

    def wrap_comparisons(cls, method_name, nested):
        original = getattr(cls, method_name)

        @functools.wraps(original)
        def compare(self, structures, *args, **kwargs):
            if "structure-matcher" not in self.config.fingerprint_method.lower():
                return original(self, structures, *args, **kwargs)
            metric = self.uniqueness_metric if nested else self
            matcher = metric.fingerprinter
            proxy = _ComparisonRows(
                matcher,
                structures,
                "comparison-rows-" + cls.__name__,
                vars(self.config),
            )
            # Replace only this metric's reference, never the shared matcher's
            # methods. Both pinned comparison loops execute serially.
            metric.fingerprinter = proxy
            try:
                return original(self, structures, *args, **kwargs)
            finally:
                try:
                    proxy.flush()
                finally:
                    metric.fingerprinter = matcher

        setattr(cls, method_name, compare)

    wrap_comparisons(UniquenessMetric, "compute", False)
    wrap_comparisons(SUNMetric, "_compute_uniqueness_within_set", True)

    def wrap_evaluate(cls):
        original = cls.evaluate

        @functools.wraps(original)
        def evaluate(self, structures, *args, **kwargs):
            return cached_call(
                "family-" + cls.__name__,
                [[structure.as_dict() for structure in structures], args, kwargs],
                lambda: original(self, structures, *args, **kwargs),
            )

        cls.evaluate = evaluate

    for cls in (
        DistributionBenchmark,
        DiversityBenchmark,
        HHIBenchmark,
        StabilityBenchmark,
        NoveltyBenchmark,
        SUNBenchmark,
        UniquenessBenchmark,
    ):
        wrap_evaluate(cls)
    print("ZEVALS_LEMAT_RESUME hooks installed", flush=True)
