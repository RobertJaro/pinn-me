"""Resume never combines snapshots from different generators or resources."""

from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from prom3theus.muram import checkpoint, provenance


def args(tmp_path, *, output_format="store", overwrite=False):
    return SimpleNamespace(
        source_path=tmp_path / "source", out=tmp_path / "output",
        overwrite=overwrite, output_format=output_format, stride=None,
        binning=None, dtype="float32", vertical_axis=None, component_order=(0, 1, 2),
    )


def test_combined_fingerprint_is_deterministic_and_covers_both_generators():
    first = provenance.synthesis_run_provenance(("aia_euv", "hmi_filter_profiles"))
    second = provenance.synthesis_run_provenance(("hmi_filter_profiles", "aia_euv"))
    assert first == second
    assert first["fingerprint_sha256"] == provenance._digest(
        {key: value for key, value in first.items() if key != "fingerprint_sha256"}
    )
    for name in ("validation/reference_transfer.py", "instruments/hmi/operator.py",
                 "instruments/aia_euv/operator.py", "instruments/aia_euv/response.py",
                 "rt/optically_thin.py", "muram/synthesize.py", "muram/atmosphere_builder.py"):
        assert name in first["implementation_sha256"]
    for name in ("common/stic_continuum_table.json", "common/chianti_thermodynamic_table.json",
                 "aia_euv_v1:aia_temperature_response"):
        assert name in first["resource_sha256"]


def test_aia_response_changes_generator_identity(monkeypatch):
    before = provenance.synthesis_run_provenance(("aia_euv",))
    file_sha256 = provenance.file_sha256

    def changed_response(path):
        return "0" * 64 if path.name == "aia_temperature_response.json" else file_sha256(path)

    monkeypatch.setattr(provenance, "file_sha256", changed_response)
    after = provenance.synthesis_run_provenance(("aia_euv",))
    assert before["fingerprint_sha256"] != after["fingerprint_sha256"]
    assert before["implementation_sha256"] == after["implementation_sha256"]


@pytest.mark.parametrize("output_format,types", [
    ("store", ("hinode_sp",)), ("store", ("aia_euv",)),
    ("sdo-fits", ("hmi_filter_profiles", "aia_euv")),
])
def test_resume_rejects_changed_generator_before_publishing(tmp_path, monkeypatch, output_format, types):
    options = {"instrument": {"type": "hinode_sp"}} if types == ("hinode_sp",) else {}
    request = args(tmp_path, output_format=output_format)
    previous = checkpoint.prepare_synthesis_output(request, options, instrument_types=types)
    contract_path = request.out / "synthesis_run.json"
    original = contract_path.read_bytes()
    marker = request.out / "saved_snapshot"
    marker.write_bytes(b"published")
    changed = deepcopy(previous["generator_provenance"])
    changed["fingerprint_sha256"] = "0" * 64
    monkeypatch.setattr(checkpoint, "synthesis_run_provenance", lambda *a, **kw: changed)
    with pytest.raises(ValueError, match="settings changed.*generator_provenance"):
        checkpoint.prepare_synthesis_output(request, options, instrument_types=types)
    assert contract_path.read_bytes() == original
    assert marker.read_bytes() == b"published"
    request.overwrite = True
    checkpoint.prepare_synthesis_output(request, options, instrument_types=types)
    assert not marker.exists()
    assert json.loads(contract_path.read_text())["generator_provenance"] == changed


def test_missing_generator_fingerprint_is_not_silently_adopted(tmp_path):
    request = args(tmp_path)
    checkpoint.prepare_synthesis_output(request, {})
    path = request.out / "synthesis_run.json"
    legacy = json.loads(path.read_text())
    legacy.pop("generator_provenance")
    path.write_text(json.dumps(legacy))
    with pytest.raises(ValueError, match="generator_provenance"):
        checkpoint.prepare_synthesis_output(request, {})


def test_invalid_generator_does_not_overwrite_output(tmp_path):
    request = args(tmp_path, overwrite=True)
    request.out.mkdir()
    marker = request.out / "saved_snapshot"
    marker.write_bytes(b"published")
    with pytest.raises(ValueError, match="Unsupported synthesis instruments"):
        checkpoint.prepare_synthesis_output(request, {"instrument": {"type": "invalid"}})
    assert marker.read_bytes() == b"published"
