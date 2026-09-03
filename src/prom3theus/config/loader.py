"""Strict YAML loader for :mod:`prom3theus.config` dataclasses."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import MISSING, fields, is_dataclass
from datetime import datetime
from pathlib import Path
import types
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints

import yaml

from .resolver import resolve_path
from .schema import InversionConfig


class ConfigError(ValueError):
    """Raised for malformed, incomplete, or unsupported configuration."""


class _UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader which rejects duplicate mapping keys."""


def _construct_unique_mapping(
    loader: _UniqueKeyLoader, node: yaml.MappingNode, deep=False
):
    loader.flatten_mapping(node)
    result: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _is_union(annotation: Any) -> bool:
    return get_origin(annotation) in (Union, types.UnionType)


def _literal_values(annotation: Any) -> tuple[Any, ...]:
    return get_args(annotation) if get_origin(annotation) is Literal else ()


def _discriminated_dataclass(
    candidates: tuple[Any, ...],
    value: Mapping[str, Any],
    context: str,
) -> Any | None:
    dataclass_candidates = [
        candidate for candidate in candidates if is_dataclass(candidate)
    ]
    if len(dataclass_candidates) != len(candidates) or not dataclass_candidates:
        return None
    if "type" not in value:
        raise ConfigError(f"{context}.type is required")

    requested = value["type"]
    supported: list[Any] = []
    for candidate in dataclass_candidates:
        annotation = get_type_hints(candidate).get("type")
        literals = _literal_values(annotation)
        supported.extend(literals)
        if any(
            type(requested) is type(item) and requested == item for item in literals
        ):
            return candidate
    raise ConfigError(
        f"{context}.type must be one of {sorted(supported)!r}; got {requested!r}"
    )


def _decode_scalar(annotation: Any, value: Any, context: str) -> Any:
    if annotation is str:
        if not isinstance(value, str):
            raise ConfigError(f"{context} must be a string")
        return value
    if annotation is bool:
        if type(value) is not bool:
            raise ConfigError(f"{context} must be a boolean")
        return value
    if annotation is int:
        if type(value) is not int:
            raise ConfigError(f"{context} must be an integer")
        return value
    if annotation is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{context} must be a number")
        return float(value)
    if annotation is datetime:
        if isinstance(value, datetime):
            return value
        if not isinstance(value, str):
            raise ConfigError(f"{context} must be an ISO-8601 timestamp")
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as error:
            raise ConfigError(f"{context} must be an ISO-8601 timestamp") from error
    if annotation is type(None):
        if value is not None:
            raise ConfigError(f"{context} must be null")
        return None
    raise ConfigError(f"{context} has unsupported schema annotation {annotation!r}")


def _decode(
    annotation: Any,
    value: Any,
    *,
    context: str,
    base_directory: Path,
    environ: Mapping[str, str] | None,
) -> Any:
    if annotation is Path:
        if not isinstance(value, str):
            raise ConfigError(f"{context} must be a path string")
        try:
            return resolve_path(value, base_directory=base_directory, environ=environ)
        except ValueError as error:
            raise ConfigError(f"{context}: {error}") from error

    literals = _literal_values(annotation)
    if literals:
        if not any(type(value) is type(item) and value == item for item in literals):
            raise ConfigError(
                f"{context} must be one of {list(literals)!r}; got {value!r}"
            )
        return value

    if is_dataclass(annotation):
        return _decode_dataclass(
            annotation,
            value,
            context=context,
            base_directory=base_directory,
            environ=environ,
        )

    origin = get_origin(annotation)
    if origin is tuple:
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"{context} must be a sequence")
        arguments = get_args(annotation)
        if len(arguments) == 2 and arguments[1] is Ellipsis:
            item_types = [arguments[0]] * len(value)
        else:
            if len(value) != len(arguments):
                raise ConfigError(
                    f"{context} must contain exactly {len(arguments)} items"
                )
            item_types = list(arguments)
        return tuple(
            _decode(
                item_type,
                item,
                context=f"{context}[{index}]",
                base_directory=base_directory,
                environ=environ,
            )
            for index, (item_type, item) in enumerate(
                zip(item_types, value, strict=True)
            )
        )

    if _is_union(annotation):
        candidates = get_args(annotation)
        if isinstance(value, Mapping):
            selected = _discriminated_dataclass(candidates, value, context)
            if selected is not None:
                return _decode_dataclass(
                    selected,
                    value,
                    context=context,
                    base_directory=base_directory,
                    environ=environ,
                )
        errors: list[str] = []
        for candidate in candidates:
            try:
                return _decode(
                    candidate,
                    value,
                    context=context,
                    base_directory=base_directory,
                    environ=environ,
                )
            except ConfigError as error:
                errors.append(str(error))
        raise ConfigError(
            f"{context} does not match any allowed type: {'; '.join(errors)}"
        )

    return _decode_scalar(annotation, value, context)


def _decode_dataclass(
    target: type[Any],
    value: Any,
    *,
    context: str,
    base_directory: Path,
    environ: Mapping[str, str] | None,
) -> Any:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{context} must be a mapping")
    if any(not isinstance(key, str) for key in value):
        raise ConfigError(f"{context} keys must be strings")

    target_fields = {field.name: field for field in fields(target)}
    unknown = sorted(set(value) - set(target_fields))
    if unknown:
        raise ConfigError(f"{context} contains unknown keys: {unknown}")

    missing = sorted(
        name
        for name, field in target_fields.items()
        if name not in value
        and field.default is MISSING
        and field.default_factory is MISSING
    )
    if missing:
        raise ConfigError(f"{context} is missing required keys: {missing}")

    annotations = get_type_hints(target)
    arguments = {
        name: _decode(
            annotations[name],
            raw,
            context=f"{context}.{name}",
            base_directory=base_directory,
            environ=environ,
        )
        for name, raw in value.items()
    }
    try:
        return target(**arguments)
    except (TypeError, ValueError) as error:
        raise ConfigError(f"{context}: {error}") from error


def parse_config(
    document: Mapping[str, Any],
    *,
    base_directory: str | Path,
    environ: Mapping[str, str] | None = None,
) -> InversionConfig:
    """Validate an already parsed configuration mapping."""

    base = Path(base_directory).expanduser().resolve(strict=False)
    return _decode_dataclass(
        InversionConfig,
        document,
        context="config",
        base_directory=base,
        environ=environ,
    )


def load_config(
    path: str | Path,
    *,
    environ: Mapping[str, str] | None = None,
) -> InversionConfig:
    """Load and strictly validate a version-1 LTE YAML configuration."""

    config_path = Path(path).expanduser().resolve(strict=False)
    try:
        text = config_path.read_text(encoding="utf-8")
    except OSError as error:
        raise ConfigError(
            f"could not read configuration {config_path}: {error}"
        ) from error
    try:
        document = yaml.load(text, Loader=_UniqueKeyLoader)
    except yaml.YAMLError as error:
        raise ConfigError(f"invalid YAML in {config_path}: {error}") from error
    if not isinstance(document, Mapping):
        raise ConfigError("configuration root must be a mapping")
    return parse_config(
        document,
        base_directory=config_path.parent,
        environ=environ,
    )


__all__ = ["ConfigError", "load_config", "parse_config"]
