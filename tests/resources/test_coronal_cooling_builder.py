import json
import os
from pathlib import Path

import numpy as np
import pytest


def test_builder_records_si_components_and_provenance(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2]))
    from resource_builder import coronal_cooling

    database = tmp_path / "chianti"
    (database / "ascii/abundance").mkdir(parents=True)
    (database / "ascii/ioneq").mkdir()
    (database / "ascii/VERSION").write_text("11.0.2\n")
    (database / "ascii/abundance/test.abund").write_text("synthetic fixture")
    (database / "ascii/ioneq/chianti.ioneq").write_text("synthetic fixture")
    (database / "chianti_11.0.2.h5").write_bytes(b"synthetic fixture")

    def calculate(path, abundance, density, log_temperature):
        assert path == database / "chianti_11.0.2.h5"
        assert abundance == "test"
        assert density == 1e9
        return {"lines": np.ones_like(log_temperature) * 1e-35}, [], ["H 1"]

    monkeypatch.setattr(coronal_cooling, "_calculate", calculate)
    monkeypatch.setattr(
        coronal_cooling.importlib.metadata, "version", lambda _: "0.8.2"
    )
    output = tmp_path / "resources/cooling.json"
    coronal_cooling.build(output, database, "test", 1e9)
    document = json.loads(output.read_text())
    np.testing.assert_allclose(document["log10_lambda_w_m3"], -35.0)
    assert document["density_convention"] == "n_e * n_H"
    assert document["provenance"]["database_version"] == "11.0.2"
    assert len(document["provenance"]["abundance_sha256"]) == 64


@pytest.mark.skipif(
    not os.environ.get("CHIANTI_DATABASE_ROOT"),
    reason="Requires local CHIANTI/fiasco data",
)
def test_fiasco_chunking_and_two_photon_quadrature(monkeypatch):
    import astropy.units as u

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2]))
    from resource_builder.coronal_cooling import _calculate, _fiasco

    fiasco = _fiasco()
    database = Path(os.environ["CHIANTI_DATABASE_ROOT"]) / "chianti_11.0.2.h5"
    options = dict(
        hdf5_dbase_root=database,
        abundance="sun_coronal_2021_chianti",
        ionization_fraction="chianti",
    )
    axis = np.array([5.9, 6.0, 6.1])
    carbon = fiasco.Ion("C 4", 10**axis * u.K, **options)
    native = (
        fiasco.IonCollection(carbon)
        .radiative_loss(
            1e9 * u.cm**-3,
            include_protons=False,
            use_two_ion_model=True,
        )
        .to_value(u.W * u.m**3)[:, 0]
    )
    components, _, _ = _calculate(database, options["abundance"], 1e9, axis, ["C 4"])
    np.testing.assert_allclose(sum(components.values()), native, rtol=1e-10, atol=0)

    helium = fiasco.Ion("He 2", [1e5] * u.K, **options)
    edge = (
        helium.levels.energy[helium.levels.label == "2s 2S1/2"]
        .to_value(u.AA, equivalencies=u.spectral())
        .item()
    )
    rates = []
    for count in (2049, 4097, 8193):
        wavelength = edge * np.geomspace(1, 1e5, count)
        spectrum = helium.two_photon(
            wavelength * u.AA,
            1e9 * u.cm**-3,
            include_protons=False,
            use_two_ion_model=False,
        )
        rates.append(
            np.trapezoid(spectrum.to_value("erg cm3 s-1 Angstrom-1")[0, 0], wavelength)
        )
    errors = np.abs(np.array(rates[:2]) / rates[2] - 1)
    assert errors[0] < 1e-4
    assert errors[1] < errors[0] / 3
