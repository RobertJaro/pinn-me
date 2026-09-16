"""Array-schema-independent prepared observation stream contracts."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from prom3theus.observations import (
    ImageObservationSpec,
    ObservationDescriptor,
    ObservationKind,
    PreparedObservationStream,
    PreparedObservationStreams,
)


@dataclass
class _DataModule:
    setup_stage: str | None = None

    def setup(self, stage: str | None = None) -> None:
        self.setup_stage = stage

    def run_metadata(self):
        return {"setup_stage": self.setup_stage}


def _image_descriptor() -> ObservationDescriptor:
    spec = ImageObservationSpec(
        observation_id="aia-euv",
        observation_type="aia_euv",
        instrument_type="aia_temperature_response",
        intensity_unit="DN s^-1 pixel^-1",
        channels_angstrom=(171, 193, 211),
        exposure_groups=("group-0",),
        required_resource_sets=("aia_euv_v1", "shared_plasma_v1"),
    )
    return ObservationDescriptor.from_spec(spec)


def test_descriptor_normalizes_image_spec_without_sharing_array_schema():
    descriptor = _image_descriptor()
    assert descriptor.observation_kind is ObservationKind.IMAGE
    assert descriptor.required_resource_sets == ("aia_euv_v1", "shared_plasma_v1")
    assert descriptor.details["channels_angstrom"] == [171, 193, 211]
    assert descriptor.metadata()["observation_kind"] == "image"

    class LegacySpec:
        observation_id = "hmi"
        observation_type = "hmi_stokes"
        instrument_type = "hmi_filter_profiles"

        @staticmethod
        def metadata():
            return {"legacy": True}

    with pytest.raises(ValueError, match="explicit observation_kind"):
        ObservationDescriptor.from_spec(LegacySpec())
    legacy = ObservationDescriptor.from_spec(
        LegacySpec(),
        observation_kind="stokes",
        required_resource_sets=("legacy_lte_v2",),
    )
    assert legacy.observation_kind is ObservationKind.STOKES
    assert legacy.details == {"legacy": True}


def test_prepared_stream_collection_is_ordered_immutable_and_sets_up_all(tmp_path):
    image_store = tmp_path / "image-store"
    stokes_store = tmp_path / "stokes-store"
    image_store.mkdir()
    stokes_store.mkdir()
    image_data = _DataModule()
    stokes_data = _DataModule()
    image = PreparedObservationStream(
        name="aia",
        descriptor=_image_descriptor(),
        data_module=image_data,
        store_path=image_store,
        source_signature="a" * 64,
    )
    stokes = PreparedObservationStream(
        name="hmi",
        descriptor=ObservationDescriptor(
            observation_id="hmi",
            observation_type="hmi_stokes",
            instrument_type="hmi_filter_profiles",
            observation_kind="stokes",
            required_resource_sets=("legacy_lte_v2", "shared_plasma_v1"),
        ),
        data_module=stokes_data,
        store_path=stokes_store,
        source_signature="b" * 64,
    )
    streams = PreparedObservationStreams([stokes, image])
    assert list(streams) == ["hmi", "aia"]
    assert streams["aia"].store_path == image_store.resolve()
    assert streams.required_resource_sets == (
        "legacy_lte_v2",
        "shared_plasma_v1",
        "aia_euv_v1",
    )
    streams.setup("validate")
    assert stokes_data.setup_stage == image_data.setup_stage == "validate"
    assert list(streams.metadata()) == ["hmi", "aia"]

    with pytest.raises(ValueError, match="names must be unique"):
        PreparedObservationStreams([image, image])


def test_prepared_stream_requires_a_real_store_and_valid_signature(tmp_path):
    with pytest.raises(NotADirectoryError, match="store not found"):
        PreparedObservationStream(
            name="aia",
            descriptor=_image_descriptor(),
            data_module=_DataModule(),
            store_path=tmp_path / "missing",
            source_signature="a" * 64,
        )
    store = tmp_path / "store"
    store.mkdir()
    with pytest.raises(ValueError, match="SHA-256"):
        PreparedObservationStream(
            name="aia",
            descriptor=_image_descriptor(),
            data_module=_DataModule(),
            store_path=store,
            source_signature="not-a-digest",
        )
