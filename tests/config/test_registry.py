"""Extensions cannot corrupt the strict schema decoder's dispatch tables."""
from dataclasses import dataclass
from typing import Literal

import pytest
from prom3theus.config import registry
from prom3theus.config.schema import ConfigNode
from prom3theus.config.joint_schema import LTEStokesDataTermConfig


@dataclass(frozen=True)
class Probe(ConfigNode):
    type: Literal["probe"]


@dataclass(frozen=True)
class ProbeTerm(ConfigNode):
    type: Literal["probe_term"]


@pytest.fixture(autouse=True)
def isolated_registry(monkeypatch):
    monkeypatch.setattr(registry, "OBSERVATION_SCHEMAS", {})
    monkeypatch.setattr(registry, "TERM_SCHEMAS", {})
    monkeypatch.setattr(registry, "COMPATIBLE_PAIRS", set(registry.COMPATIBLE_PAIRS))


def test_registration_reuses_builtin_term_without_duplicate_decoder_candidates():
    registry.register_stream_schema(
        "probe", Probe, "lte_stokes", LTEStokesDataTermConfig
    )
    assert registry.OBSERVATION_SCHEMAS == {"probe": Probe}
    assert registry.TERM_SCHEMAS == {}
    assert ("probe", "lte_stokes") in registry.COMPATIBLE_PAIRS


def test_failed_second_schema_leaves_no_partial_registration():
    with pytest.raises(ValueError, match="discriminator"):
        registry.register_stream_schema("probe", Probe, "wrong_name", ProbeTerm)
    assert registry.OBSERVATION_SCHEMAS == registry.TERM_SCHEMAS == {}
    assert not any(pair[0] == "probe" for pair in registry.COMPATIBLE_PAIRS)


def test_builtin_schema_cannot_be_shadowed():
    @dataclass(frozen=True)
    class Replacement(ConfigNode):
        type: Literal["lte_stokes"]

    with pytest.raises(ValueError, match="already registered"):
        registry.register_stream_schema("probe", Probe, "lte_stokes", Replacement)
    assert registry.OBSERVATION_SCHEMAS == {}


def test_extension_union_dispatch_uses_class_identity():
    registry.register_stream_schema("probe", Probe, "probe_term", ProbeTerm)
    builtins, _ = registry._builtin_schemas()
    assert registry.extension_candidates(tuple(builtins.values())) == (Probe,)
    # A similarly named class in an unrelated config union must not acquire extensions.
    unrelated = type("AIAObservationConfig", (), {})
    assert registry.extension_candidates((unrelated,)) == ()
