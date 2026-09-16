"""Strict extension schemas; register explicitly before parsing configurations."""
from dataclasses import is_dataclass
import re
from typing import Literal, get_args, get_origin, get_type_hints

OBSERVATION_SCHEMAS = {}
TERM_SCHEMAS = {}
COMPATIBLE_PAIRS = {
    ("hinode_sp", "lte_stokes"),
    ("hmi_stokes", "lte_stokes"),
    ("aia_euv", "aia_optically_thin"),
}


def _builtin_schemas():
    # Resolve after the schema module is initialized; it uses COMPATIBLE_PAIRS.
    from .joint_schema import (
        AIAObservationConfig,
        AIAOpticallyThinDataTermConfig,
        LTEStokesDataTermConfig,
    )
    from .schema import HMIObservationConfig, HinodeObservationConfig

    return (
        {
            "hmi_stokes": HMIObservationConfig,
            "hinode_sp": HinodeObservationConfig,
            "aia_euv": AIAObservationConfig,
        },
        {
            "lte_stokes": LTEStokesDataTermConfig,
            "aia_optically_thin": AIAOpticallyThinDataTermConfig,
        },
    )


def extension_candidates(candidates):
    """Extend only the observation/term unions, identified by schema identity."""
    for builtin, extensions in zip(
        _builtin_schemas(), (OBSERVATION_SCHEMAS, TERM_SCHEMAS)
    ):
        if set(candidates) == set(builtin.values()):
            return tuple(extensions.values())
    return ()


def register_stream_schema(
    observation_type, observation_schema, term_type, term_schema
):
    from .schema import ConfigNode

    entries = tuple(
        zip(
            (OBSERVATION_SCHEMAS, TERM_SCHEMAS),
            _builtin_schemas(),
            (observation_type, term_type),
            (observation_schema, term_schema),
        )
    )
    # Validate both sides before publishing either, so failed registration is atomic.
    for registry, builtin, name, schema in entries:
        if (
            not isinstance(schema, type)
            or not is_dataclass(schema)
            or not issubclass(schema, ConfigNode)
        ):
            raise TypeError("Stream schemas must be ConfigNode dataclass types")
        if not isinstance(name, str) or re.fullmatch(r"[a-z][a-z0-9_]*", name) is None:
            raise ValueError("Stream type must be a lowercase identifier")
        annotation = get_type_hints(schema).get("type")
        if get_origin(annotation) is not Literal or get_args(annotation) != (name,):
            raise ValueError(
                "Schema type must be the matching single Literal discriminator"
            )
        existing = builtin.get(name, registry.get(name))
        if existing is not None and existing is not schema:
            raise ValueError(f"Schema already registered: {name}")
    for registry, builtin, name, schema in entries:
        if name not in builtin:
            registry[name] = schema
    COMPATIBLE_PAIRS.add((observation_type, term_type))
