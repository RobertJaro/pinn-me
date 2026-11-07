"""Synthetic recovery tests must identify their target forward model."""

import warnings
from copy import deepcopy

import pytest

from prom3theus.muram.provenance import (
    SyntheticForwardModelWarning,
    _digest,
    check_stokes_forward_provenance,
    stokes_forward_provenance,
)


def test_current_forward_model_has_verified_code_and_resource_identity():
    provenance = stokes_forward_provenance("hmi_filter_profiles")
    assert provenance["contract_version"] == 1
    assert "validation/reference_transfer.py" in provenance["implementation_sha256"]
    assert "common/stic_continuum_table.json" in provenance["resource_sha256"]
    with warnings.catch_warnings(record=True) as caught:
        result = check_stokes_forward_provenance(provenance, "hmi_filter_profiles")
    assert not caught
    assert result["status"] == "matched"


@pytest.mark.parametrize("legacy", [None, {}])
def test_legacy_synthetic_targets_warn_but_remain_loadable(legacy):
    with pytest.warns(
        SyntheticForwardModelWarning, match="lack forward-model provenance"
    ):
        result = check_stokes_forward_provenance(legacy, "hmi_filter_profiles")
    assert result["status"] == "legacy_unverified"


def test_changed_forward_model_warns_and_corrupt_fingerprint_fails():
    old = deepcopy(stokes_forward_provenance("hmi_filter_profiles"))
    old["implementation_sha256"]["validation/reference_transfer.py"] = "0" * 64
    old["fingerprint_sha256"] = _digest(
        {k: v for k, v in old.items() if k != "fingerprint_sha256"}
    )
    with pytest.warns(SyntheticForwardModelWarning, match="different forward-model"):
        result = check_stokes_forward_provenance(old, "hmi_filter_profiles")
    assert result["status"] == "mismatch"
    old["fingerprint_sha256"] = "1" * 64
    with pytest.raises(ValueError, match="fingerprint is inconsistent"):
        check_stokes_forward_provenance(old, "hmi_filter_profiles")


def test_malformed_forward_model_provenance_fails():
    with pytest.raises(ValueError, match="Invalid MURaM"):
        check_stokes_forward_provenance({"unexpected": "value"}, "hmi_filter_profiles")


def test_cached_synthetic_targets_recheck_the_current_forward_model(
    tmp_path, monkeypatch
):
    """A serialization-cache hit must still flag an RT/resources change."""
    from types import SimpleNamespace

    import prom3theus.muram.provenance as provenance_module
    from prom3theus.observations import registry

    stored = stokes_forward_provenance("hmi_filter_profiles")
    current = deepcopy(stored)
    current["resource_sha256"]["common/lines.json"] = "0" * 64
    current["fingerprint_sha256"] = _digest(
        {k: v for k, v in current.items() if k != "fingerprint_sha256"}
    )
    monkeypatch.setattr(
        provenance_module, "stokes_forward_provenance", lambda _: current
    )
    raster = SimpleNamespace(
        metadata={
            "muram_synthesis": {
                "instrument": {"type": "hmi_filter_profiles"},
                "forward_model_provenance": stored,
            }
        }
    )
    source = tmp_path / "source"
    source.write_text("synthetic fixture")
    adapter = SimpleNamespace(name="hmi_stokes", source_files=lambda _: (source,))
    monkeypatch.setattr(registry, "get_observation_adapter", lambda _: adapter)
    monkeypatch.setattr(
        registry, "observation_store_signature", lambda *a, **kw: "cached"
    )
    monkeypatch.setattr(
        registry.ObservationStore,
        "load_sequence",
        lambda *a, **kw: (
            [raster],
            ["synthetic"],
            {"adapter": "hmi_stokes", "observation": {}},
        ),
    )
    monkeypatch.setattr(registry, "_spec_from_metadata", lambda _: "spec")
    monkeypatch.setattr(registry, "_stored_data_module", lambda *a: "data")
    (tmp_path / "hmi_stokes-cached.observation").mkdir()
    with pytest.warns(SyntheticForwardModelWarning, match="different forward-model"):
        data, spec, _ = registry.load_or_prepare_observation(
            {"type": "hmi_stokes"}, {}, tmp_path
        )
    assert data == "data" and spec == "spec"
