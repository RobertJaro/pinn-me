import importlib.resources
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import pytest

import pme.lte.fetch_data as atomic_fetch
from pme.lte.resources import validate_resource_bundle


def test_lte_resources_are_available_through_importlib():
    data = importlib.resources.files("pme.lte").joinpath("data")
    assert data.joinpath("lines.json").is_file()
    assert data.joinpath("abundances.json").is_file()
    assert data.joinpath("barklem_collet_2016_table4.dat").is_file()
    assert data.joinpath("barklem_collet_2016_table8.dat").is_file()
    assert data.joinpath("barklem_collet_2016_ReadMe.txt").is_file()
    assert data.joinpath("hminus_continuum.json").is_file()
    assert data.joinpath("blend_inventory.json").is_file()
    assert data.joinpath("instrument_hinode_sp.json").is_file()
    assert data.joinpath("sources.json").is_file()
    assert data.joinpath("eos_table.json").is_file()
    assert data.joinpath("stic_continuum_table.json").is_file()
    assert data.joinpath("solar_reference_630nm.json").is_file()


def test_standalone_lte_api_is_exported_from_package():
    from pme.lte import LTESynthesizer, StratifiedAtmosphere, StratifiedAtmosphereModel

    assert LTESynthesizer.__name__ == "LTESynthesizer"
    assert StratifiedAtmosphere.__name__ == "StratifiedAtmosphere"
    assert StratifiedAtmosphereModel.__name__ == "StratifiedAtmosphereModel"


def test_atomic_source_manifest_files_match_recorded_checksums():
    data = importlib.resources.files("pme.lte").joinpath("data")
    manifest = json.loads(data.joinpath("sources.json").read_text(encoding="utf-8"))
    assert manifest["sources"] == atomic_fetch.SOURCES
    assert set(manifest["source_roles"]["bundle_input_ids"]) == set(
        atomic_fetch.BUNDLE_INPUT_SOURCE_IDS
    )
    assert set(manifest["source_roles"]["citation_only_ids"]) == set(
        atomic_fetch.CITATION_ONLY_SOURCE_IDS
    )
    records = [
        *manifest.get("vendored_files", {}).items(),
        *manifest.get("generated_files", {}).items(),
        *manifest.get("reviewed_files", {}).items(),
    ]
    assert records
    for filename, record in records:
        resource = data.joinpath(filename)
        assert resource.is_file(), f"manifest resource is missing: {filename}"
        digest = hashlib.sha256(resource.read_bytes()).hexdigest()
        assert digest == record["sha256"], f"checksum mismatch for {filename}"

    stic = json.loads(
        data.joinpath("stic_continuum_table.json").read_text(encoding="utf-8")
    )
    assert stic["schema_version"] == 2
    assert "hybrid_eos_comparison" not in stic
    assert stic["log10_neutral_hydrogen_density_m3"]
    assert stic["log10_fe_i_population_over_partition_m3"]


def test_atomic_downloader_identifies_itself_and_verifies_payload(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger=atomic_fetch.LOGGER.name)
    payload = b"pinned atomic fixture"
    seen = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return payload

    def fake_urlopen(request, timeout):
        seen["request"] = request
        seen["timeout"] = timeout
        return Response()

    monkeypatch.setattr(atomic_fetch, "urlopen", fake_urlopen)
    source = {
        "url": "https://example.invalid/atomic.dat",
        "filename": "atomic.dat",
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    assert atomic_fetch._fetch("fixture", source, None) == payload
    assert seen["request"].get_header("User-agent") == atomic_fetch.USER_AGENT
    assert seen["timeout"] == 60
    assert "Downloading fixture" in caplog.text
    assert "Downloaded and verified: fixture" in caplog.text


def test_atomic_downloader_reuses_a_verified_cache_without_network(
    monkeypatch, tmp_path, caplog
):
    caplog.set_level(logging.INFO, logger=atomic_fetch.LOGGER.name)
    payload = b"cached pinned source"
    source = {
        "url": "https://example.invalid/cached.dat",
        "filename": "cached.dat",
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    (tmp_path / source["filename"]).write_bytes(payload)
    monkeypatch.setattr(
        atomic_fetch,
        "urlopen",
        lambda *args, **kwargs: pytest.fail("verified cache triggered network access"),
    )
    assert atomic_fetch._fetch("cached", source, tmp_path) == payload
    assert "Cache hit: cached" in caplog.text


def test_default_fetch_excludes_citation_only_endpoints(monkeypatch, tmp_path, caplog):
    caplog.set_level(logging.INFO, logger=atomic_fetch.LOGGER.name)
    requested = []

    def record_fetch(name, source, cache_dir, *, refresh=False):
        requested.append(name)
        return name.encode("ascii")

    monkeypatch.setattr(atomic_fetch, "_fetch", record_fetch)
    payloads = atomic_fetch._fetch_all(tmp_path, workers=3, refresh=False)
    assert set(requested) == set(atomic_fetch.BUNDLE_INPUT_SOURCE_IDS)
    assert set(payloads) == set(atomic_fetch.BUNDLE_INPUT_SOURCE_IDS)
    assert set(requested).isdisjoint(atomic_fetch.CITATION_ONLY_SOURCE_IDS)
    assert f"Source progress: {len(requested)}/{len(requested)} ready" in caplog.text
    assert f"All {len(requested)} generation inputs are checksum-verified" in caplog.text


def test_eos_only_cli_upgrades_existing_bundle_without_network(
    monkeypatch, tmp_path
):
    output = tmp_path / "existing_bundle"
    called = []
    monkeypatch.setattr(
        atomic_fetch,
        "prepare_eos_table",
        lambda directory: called.append(Path(directory)),
    )
    monkeypatch.setattr(
        atomic_fetch,
        "_fetch_all",
        lambda *args, **kwargs: pytest.fail("--eos-only attempted a download"),
    )
    monkeypatch.setattr(
        atomic_fetch,
        "validate_resource_bundle",
        lambda directory: {"source_manifest_sha256": "fixture-manifest"},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["pinn-me-lte-fetch-data", "--eos-only", "--output-dir", str(output)],
    )
    atomic_fetch.main()
    assert called == [output]


def test_fetch_cli_reuses_a_current_seal_before_touching_network(
    monkeypatch, tmp_path, caplog
):
    caplog.set_level(logging.INFO, logger=atomic_fetch.LOGGER.name)
    output = tmp_path / "current_bundle"
    monkeypatch.setattr(
        atomic_fetch,
        "_current_sealed_bundle",
        lambda directory: {"directory": str(directory)},
    )
    monkeypatch.setattr(
        atomic_fetch,
        "_fetch_all",
        lambda *args, **kwargs: pytest.fail("current sealed bundle triggered a download"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["pinn-me-lte-fetch-data", "--output-dir", str(output)],
    )
    atomic_fetch.main()
    assert "Bundle is current; no downloads or conversion required" in caplog.text


def test_eos_only_rejects_a_missing_or_legacy_stic_bundle(tmp_path):
    missing = tmp_path / "missing"
    with pytest.raises(RuntimeError, match="full bundle"):
        atomic_fetch.prepare_eos_table(missing)

    legacy = tmp_path / "legacy"
    legacy.mkdir()
    (legacy / "sources.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "sources": {},
                "vendored_files": {},
                "generated_files": {},
                "reviewed_files": {},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="legacy resource directory"):
        atomic_fetch.prepare_eos_table(legacy)

    schema_one = tmp_path / "schema_one"
    schema_one.mkdir()
    (schema_one / "stic_continuum_table.json").write_text(
        json.dumps({"schema_version": 1}), encoding="utf-8"
    )
    (schema_one / "sources.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "sources": {
                    source_id: atomic_fetch.SOURCES[source_id]
                    for source_id in atomic_fetch.STIC_SOURCE_IDS
                },
                "vendored_files": {},
                "generated_files": {
                    "stic_continuum_table.json": {"sha256": "not-used"}
                },
                "reviewed_files": {},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="legacy resource directory"):
        atomic_fetch.prepare_eos_table(schema_one)


def test_prepared_bundle_uses_preconverted_partition_table(tmp_path, monkeypatch):
    data = importlib.resources.files("pme.lte").joinpath("data")
    payloads = {name: b"unused citation payload" for name in atomic_fetch.SOURCES}
    for source_id in (
        "barklem_collet_readme",
        "barklem_collet_ionization",
        "barklem_collet_partitions",
    ):
        filename = atomic_fetch.SOURCES[source_id]["filename"]
        payloads[source_id] = data.joinpath(filename).read_bytes()
    payloads["lightweaver_background"] = b"""
static constexpr double lambdaBF[2] = {0.0, 1641.9};
static constexpr double alphaBF[2] = {0.0, 1.0};
static constexpr double lambdaFF[2] = {0.0, 9113.0};
static constexpr double thetaFF[2] = {0.5, 2.0};
static constexpr double kappaFF[4] = {1.0, 2.0, 3.0, 4.0};
"""
    for source_id, payload in payloads.items():
        monkeypatch.setitem(
            atomic_fetch.SOURCES[source_id],
            "sha256",
            hashlib.sha256(payload).hexdigest(),
        )
    # This test exercises bundle assembly/checksums, not the comparatively
    # expensive pinned STiC solver.  Reuse the already generated, verified
    # package table as its deterministic fixture.
    monkeypatch.setattr(
        atomic_fetch,
        "_write_stic_continuum_table",
        lambda payloads, destination, **kwargs: destination.write_bytes(
            data.joinpath("stic_continuum_table.json").read_bytes()
        ),
    )
    def write_solar_reference(data_directory, payloads):
        del payloads
        destination = data_directory / "solar_reference_630nm.json"
        destination.write_bytes(data.joinpath(destination.name).read_bytes())
        manifest_path = data_directory / "sources.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest.setdefault("generated_files", {})[destination.name] = {
            "sha256": hashlib.sha256(destination.read_bytes()).hexdigest()
        }
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        return manifest

    monkeypatch.setattr(atomic_fetch, "prepare_solar_reference", write_solar_reference)
    bundle_directory = tmp_path / "hinode_sp_v1"
    atomic_fetch.prepare_bundle(
        bundle_directory,
        payloads,
        reviewed_source=Path(str(data)),
    )
    metadata = validate_resource_bundle(bundle_directory)
    assert metadata["training_contract"] == (
        "offline-read-only; no downloads or table conversion"
    )
    assert metadata["required_production_files"]
    assert set(metadata["optional_reference_files"]) == {
        "barklem_collet_2016_ReadMe.txt",
        "barklem_collet_2016_table4.dat",
        "barklem_collet_2016_table8.dat",
        "eos_table.json",
        "hminus_continuum.json",
        "partition_functions.json",
    }

    monkeypatch.setattr(
        atomic_fetch,
        "_write_stic_continuum_table",
        lambda *args, **kwargs: pytest.fail("valid sealed bundle regenerated STiC"),
    )
    monkeypatch.setattr(
        atomic_fetch,
        "_write_eos_table",
        lambda *args, **kwargs: pytest.fail("valid sealed bundle regenerated EOS"),
    )
    reused = atomic_fetch.prepare_bundle(
        bundle_directory,
        payloads,
        reviewed_source=Path(str(data)),
    )
    assert reused["source_manifest_sha256"] == metadata["source_manifest_sha256"]

    production_only = tmp_path / "production_only"
    shutil.copytree(bundle_directory, production_only)
    production_manifest_path = production_only / "sources.json"
    production_manifest = json.loads(
        production_manifest_path.read_text(encoding="utf-8")
    )
    optional_files = set(metadata["optional_reference_files"])
    for group in ("vendored_files", "generated_files", "reviewed_files"):
        for filename in optional_files:
            production_manifest.get(group, {}).pop(filename, None)
    for filename in optional_files:
        (production_only / filename).unlink(missing_ok=True)
    production_manifest_path.write_text(
        json.dumps(production_manifest, indent=2) + "\n", encoding="utf-8"
    )
    production_bundle_path = production_only / "bundle.json"
    production_bundle = json.loads(
        production_bundle_path.read_text(encoding="utf-8")
    )
    production_bundle["source_manifest_sha256"] = hashlib.sha256(
        production_manifest_path.read_bytes()
    ).hexdigest()
    production_bundle["runtime_files"] = sorted(
        [
            *production_manifest["vendored_files"],
            *production_manifest["generated_files"],
            *production_manifest["reviewed_files"],
            "sources.json",
        ]
    )
    production_bundle["optional_reference_files"] = []
    production_bundle_path.write_text(
        json.dumps(production_bundle, indent=2) + "\n", encoding="utf-8"
    )
    production_metadata = validate_resource_bundle(production_only)
    assert production_metadata["optional_reference_files"] == []

    from pme.lte.atomic import AtomicDatabase
    from pme.lte.instrument import HinodeSpectralPSF

    database = AtomicDatabase(data_directory=bundle_directory)
    assert database.partition_table.source.endswith("partition_functions.json")
    assert len(database.partition_table.values) == 284
    instrument = HinodeSpectralPSF(data_directory=bundle_directory)
    assert instrument.metadata()["provenance"]["instrument"] == "Hinode/SOT-SP"

    (bundle_directory / "partition_functions.json").write_bytes(b"altered")
    with pytest.raises(RuntimeError, match="failed SHA256 verification"):
        validate_resource_bundle(bundle_directory)


def test_runtime_atomic_database_rejects_a_modified_scientific_resource(tmp_path):
    from pme.lte.atomic import AtomicDatabase

    data = importlib.resources.files("pme.lte").joinpath("data")
    altered = tmp_path / "lines.json"
    altered.write_bytes(data.joinpath("lines.json").read_bytes() + b"\n")
    with pytest.raises(RuntimeError, match="failed SHA256 verification"):
        AtomicDatabase(line_file=altered)


def test_wheel_contains_lte_resources_and_loads_outside_checkout(tmp_path):
    repository = Path(__file__).resolve().parents[2]
    checkout = tmp_path / "source"
    checkout.mkdir()
    for name in ("pyproject.toml", "README.md", "LICENSE"):
        shutil.copy2(repository / name, checkout / name)
    shutil.copytree(
        repository / "pme",
        checkout / "pme",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )

    wheel_dir = tmp_path / "wheel"
    wheel_dir.mkdir()
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    subprocess.run(
        [sys.executable, "-m", "build", "--wheel", "--no-isolation", "--outdir", str(wheel_dir)],
        cwd=checkout,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    wheels = list(wheel_dir.glob("*.whl"))
    assert len(wheels) == 1
    with zipfile.ZipFile(wheels[0]) as archive:
        names = set(archive.namelist())
        entry_points_name = next(
            name for name in names if name.endswith(".dist-info/entry_points.txt")
        )
        entry_points = archive.read(entry_points_name).decode("utf-8")
    assert "pme/lte/data/lines.json" in names
    assert "pme/lte/data/abundances.json" in names
    assert "pme/lte/data/sources.json" in names
    assert "pme/lte/data/barklem_collet_2016_table4.dat" in names
    assert "pme/lte/data/barklem_collet_2016_table8.dat" in names
    assert "pme/lte/data/barklem_collet_2016_ReadMe.txt" in names
    assert "pme/lte/data/hminus_continuum.json" in names
    assert "pme/lte/data/blend_inventory.json" in names
    assert "pme/lte/data/instrument_hinode_sp.json" in names
    assert "pme/lte/data/eos_table.json" in names
    assert "pme/lte/data/stic_continuum_table.json" in names
    assert "pinn-me-lte = pme.inversion_lte:main" in entry_points
    assert "pinn-me-lte-export = pme.evaluation.lte:main" in entry_points
    assert "pinn-me-lte-fetch-data = pme.lte.fetch_data:main" in entry_points
    assert "pinn-me-lte-recovery = pme.evaluation.lte_synthetic_recovery:main" in entry_points

    target = tmp_path / "installed"
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-deps", "--target", str(target), str(wheels[0])],
        cwd=tmp_path,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    check = (
        "from importlib.resources import files; "
        "import torch; import pme.lte; "
        "from pme.lte.atomic import AtomicDatabase; "
        "database = AtomicDatabase(); assert len(database.lines) >= 2; "
        "q=torch.linspace(-4.,1.,5); "
        "atmosphere=pme.lte.StratifiedAtmosphere("
        "log_tau500=q,temperature=torch.full((1,5),5500.),"
        "velocity_field=torch.zeros(1,5,3),microturbulence=torch.full((1,5),1000.),"
        "magnetic_field=torch.zeros(1,5,3),"
        "gas_pressure=torch.logspace(0.,5.,5).reshape(1,5)); "
        "synthesizer=pme.lte.LTESynthesizer(log_tau500=q); "
        "stokes=synthesizer(atmosphere,torch.linspace(6301.3,6302.7,9)); "
        "assert stokes.shape == (1,4,9); assert torch.isfinite(stokes).all(); "
        "assert files('pme.lte').joinpath('data/lines.json').is_file()"
    )
    installed_environment = environment.copy()
    installed_environment["PYTHONPATH"] = str(target)
    subprocess.run(
        [sys.executable, "-c", check],
        cwd=tmp_path,
        env=installed_environment,
        check=True,
        capture_output=True,
        text=True,
    )


def test_atomic_submodule_import_does_not_load_training_or_plotting_stack():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import pme.lte.atomic; "
                "blocked=('pytorch_lightning','matplotlib','torchmetrics'); "
                "loaded=[name for name in blocked if name in sys.modules]; "
                "assert not loaded, loaded"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
