"""Small common contracts for independently prepared observation streams."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable


class ObservationKind(StrEnum):
    STOKES = "stokes"
    IMAGE = "image"


def _identifier(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")
    return value.strip()


@runtime_checkable
class ObservationStreamDataModule(Protocol):
    """Minimal loader behavior required by the joint runtime."""

    def setup(self, stage: str | None = None) -> None:
        ...

    def run_metadata(self) -> Mapping[str, Any]:
        ...


@dataclass(frozen=True, slots=True)
class ObservationDescriptor:
    """Array-schema-independent identity of one observation product."""

    observation_id: str
    observation_type: str
    instrument_type: str
    observation_kind: ObservationKind | str
    required_resource_sets: tuple[str, ...] = ()
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("observation_id", "observation_type", "instrument_type"):
            object.__setattr__(self, name, _identifier(getattr(self, name), name))
        raw_kind = _identifier(self.observation_kind, "observation_kind")
        if (
            not raw_kind.replace("_", "a").isalnum()
            or not raw_kind[0].isalpha()
            or not raw_kind.islower()
        ):
            raise ValueError("observation_kind must be a lowercase identifier")
        try:
            kind = ObservationKind(raw_kind)
        except ValueError:
            kind = raw_kind
        resources = tuple(
            _identifier(value, "required resource set")
            for value in self.required_resource_sets
        )
        if len(set(resources)) != len(resources):
            raise ValueError("required_resource_sets must be unique.")
        if not isinstance(self.details, Mapping) or any(
            not isinstance(key, str) for key in self.details
        ):
            raise TypeError("details must be a mapping with string keys.")
        object.__setattr__(self, "observation_kind", kind)
        object.__setattr__(self, "required_resource_sets", resources)
        object.__setattr__(self, "details", MappingProxyType(dict(self.details)))

    @classmethod
    def from_spec(
        cls,
        spec: Any,
        *,
        observation_kind: ObservationKind | str | None = None,
        required_resource_sets: Sequence[str] | None = None,
    ) -> "ObservationDescriptor":
        """Normalize either the legacy Stokes spec or a new image spec."""

        kind = (
            getattr(spec, "observation_kind", None)
            if observation_kind is None
            else observation_kind
        )
        if kind is None:
            raise ValueError(
                "Legacy observation specifications require explicit observation_kind."
            )
        resources = (
            getattr(spec, "required_resource_sets", ())
            if required_resource_sets is None
            else required_resource_sets
        )
        metadata = spec.metadata()
        if not isinstance(metadata, Mapping):
            raise TypeError(
                "Observation specification metadata() must return a mapping."
            )
        return cls(
            observation_id=spec.observation_id,
            observation_type=spec.observation_type,
            instrument_type=spec.instrument_type,
            observation_kind=kind,
            required_resource_sets=tuple(resources),
            details=metadata,
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "observation_id": self.observation_id,
            "observation_type": self.observation_type,
            "instrument_type": self.instrument_type,
            "observation_kind": getattr(
                self.observation_kind, "value", self.observation_kind
            ),
            "required_resource_sets": list(self.required_resource_sets),
            "details": dict(self.details),
        }


@dataclass(frozen=True, slots=True)
class PreparedObservationStream:
    """Prepared store, loader, and scientific identity passed to a data term."""

    name: str
    descriptor: ObservationDescriptor
    data_module: ObservationStreamDataModule
    store_path: Path | str
    source_signature: str

    def __post_init__(self) -> None:
        name = _identifier(self.name, "stream name")
        if not isinstance(self.descriptor, ObservationDescriptor):
            raise TypeError("descriptor must be an ObservationDescriptor.")
        if not isinstance(self.data_module, ObservationStreamDataModule):
            raise TypeError("data_module must provide setup(stage) and run_metadata().")
        path = Path(self.store_path).expanduser().resolve()
        if not path.is_dir():
            raise NotADirectoryError(f"Prepared observation store not found: {path}.")
        signature = self.source_signature
        if (
            not isinstance(signature, str)
            or len(signature) != 64
            or any(character not in "0123456789abcdef" for character in signature)
        ):
            raise ValueError("source_signature must be a lowercase SHA-256 digest.")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "store_path", path)

    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "descriptor": self.descriptor.metadata(),
            "store_path": str(self.store_path),
            "source_signature": self.source_signature,
        }


class PreparedObservationStreams(Mapping[str, PreparedObservationStream]):
    """Ordered immutable collection used by instrument-neutral coordinators."""

    def __init__(self, streams: Sequence[PreparedObservationStream]) -> None:
        values = tuple(streams)
        if not values:
            raise ValueError("At least one prepared observation stream is required.")
        if any(not isinstance(stream, PreparedObservationStream) for stream in values):
            raise TypeError("streams must contain PreparedObservationStream values.")
        names = [stream.name for stream in values]
        if len(set(names)) != len(names):
            raise ValueError("Prepared observation stream names must be unique.")
        self._values = values
        self._mapping = MappingProxyType(dict(zip(names, values, strict=True)))

    def __getitem__(self, key: str) -> PreparedObservationStream:
        return self._mapping[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._mapping)

    def __len__(self) -> int:
        return len(self._mapping)

    def setup(self, stage: str | None = None) -> None:
        for stream in self._values:
            stream.data_module.setup(stage)

    @property
    def required_resource_sets(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                resource
                for stream in self._values
                for resource in stream.descriptor.required_resource_sets
            )
        )

    def metadata(self) -> dict[str, Any]:
        return {stream.name: stream.metadata() for stream in self._values}


__all__ = [
    "ObservationDescriptor",
    "ObservationKind",
    "ObservationStreamDataModule",
    "PreparedObservationStream",
    "PreparedObservationStreams",
]
