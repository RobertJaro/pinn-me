"""Schema-v3 AIA optically thin image likelihood."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any

import torch

from prom3theus.inversion.depth_sampling import DepthRefinement

from prom3theus.instruments.aia_euv import (
    AIAChannelCalibration,
    AIAEmissionOperator,
)
from prom3theus.inversion.ray_integral import RayIntegralForwardComposition

from .aia_objective import AsinhMSEImageObjective
from .base import DataTermBatchResult, ObservationDataTerm


class AIAEUVObservationTerm(ObservationDataTerm):
    """One AIA image term coupled to the shared atmosphere through ``T``/``n_e``.

    The term owns only AIA-specific response and calibration modules.  Its
    ray composition keeps a non-registering reference to the one joint
    atmosphere, avoiding duplicate parameters or a second thermodynamic field.
    """

    distributed_channel_reduction = True
    observation_kind = "image"

    def __init__(
        self,
        *,
        atmosphere_model,
        scene,
        observation_id: str,
        channels_angstrom: Sequence[int],
        response_resource: str,
        ray_samples: int,
        coarse_to_fine: dict | None = None,
        training_jitter: bool = True,
        height_sampling_power: float,
        asinh_scales: Sequence[float],
        channel_weights: Sequence[float],
        intensity_scales: Sequence[float] | None = None,
        calibration_enabled: bool = True,
        calibration_absolute_prior_fraction: float = 0.25,
        calibration_relative_prior_fraction: float = 0.15,
    ) -> None:
        super().__init__()
        if not isinstance(observation_id, str) or not observation_id:
            raise ValueError("observation_id must be non-empty.")
        self.observation_id = observation_id
        self.channels_angstrom = tuple(int(value) for value in channels_angstrom)
        scales = (
            tuple(float(value) for value in intensity_scales)
            if intensity_scales is not None
            else (1.0,) * len(self.channels_angstrom)
        )
        if len(scales) != len(self.channels_angstrom) or any(
            not math.isfinite(value) or value <= 0 for value in scales
        ):
            raise ValueError(
                "One finite positive intensity scale is required per AIA channel."
            )
        self.register_buffer(
            "intensity_scales", torch.tensor(scales, dtype=torch.float32)
        )
        self.ray_samples = int(ray_samples)
        self.training_jitter = training_jitter
        self.height_sampling_power = float(height_sampling_power)
        self.emission_operator = AIAEmissionOperator(
            self.channels_angstrom,
            response_resource=response_resource,
        )
        refinement = coarse_to_fine or {
            "enabled": False,
            "fine_sample_count": 1,
            "uniform_weight_floor": 0.0,
        }
        self._composition = RayIntegralForwardComposition(
            depth_refinement=DepthRefinement(
                enabled=refinement["enabled"],
                sample_count=refinement["fine_sample_count"],
                uniform_weight_floor=refinement["uniform_weight_floor"],
            ),
            atmosphere_model=atmosphere_model,
            scene=scene,
            emission_operator=self.emission_operator,
        )
        self.calibration = AIAChannelCalibration(
            self.channels_angstrom,
            enabled=calibration_enabled,
            absolute_prior_fraction=calibration_absolute_prior_fraction,
            relative_prior_fraction=calibration_relative_prior_fraction,
        )
        self.objective = AsinhMSEImageObjective(
            self.channels_angstrom,
            asinh_scales=[
                float(a) / scale for a, scale in zip(asinh_scales, scales, strict=True)
            ],
            channel_weights=channel_weights,
        )

    @property
    def response_metadata(self) -> dict:
        return self.emission_operator.metadata()

    def synthesize(
        self,
        batch: Mapping[str, Any],
        *,
        ray_samples: int | None = None,
        randomize: bool = False,
        refine: bool = True,
    ):
        """Render one image batch without evaluating its objective."""

        required = {
            "surface_position_m",
            "ray_direction",
            "absolute_tai_seconds",
            "channel_index",
        }
        missing = sorted(required - set(batch))
        if missing:
            raise KeyError(f"AIA image batch is missing fields: {missing}.")
        return self._composition.synthesize(
            surface_position_m=batch["surface_position_m"],
            ray_direction=batch["ray_direction"],
            absolute_tai_seconds=batch["absolute_tai_seconds"],
            channel_index=batch["channel_index"],
            sample_count=self.ray_samples if ray_samples is None else int(ray_samples),
            height_sampling_power=self.height_sampling_power,
            randomize=randomize,
            refine=refine,
        )

    def predict(self, batch):
        return self.synthesize(batch).raw_prediction

    def _evaluate_batch(
        self,
        batch: Mapping[str, Any],
        *,
        require_all_channels: bool,
    ) -> DataTermBatchResult:
        required = {
            "intensity",
            "channel_angstrom",
            "channel_index",
            "image_index",
        }
        missing = sorted(required - set(batch))
        if missing:
            raise KeyError(f"AIA image batch is missing fields: {missing}.")
        result = self.synthesize(
            batch, randomize=self.training and self.training_jitter
        )
        channel_index = batch["channel_index"].to(device=result.prediction.device)
        channel_angstrom = batch["channel_angstrom"].to(device=result.prediction.device)
        expected_channel = torch.as_tensor(
            self.channels_angstrom,
            device=result.prediction.device,
            dtype=channel_angstrom.dtype,
        )[channel_index.long()]
        if not torch.equal(channel_angstrom, expected_channel):
            raise ValueError(
                "AIA channel_angstrom does not match the configured channel_index."
            )
        # Dataset targets are already dimensionless. Normalize physical synthesis
        # once, before applying the dimensionless gain and the image objective.
        raw_prediction = (
            result.prediction
            / self.intensity_scales.to(result.prediction)[channel_index.long()]
        )
        target = batch["intensity"].to(raw_prediction)
        calibrated = self.calibration(raw_prediction, channel_index)
        total, channel_components, asinh_residual = self.objective(
            calibrated,
            target,
            channel_index,
            batch.get("valid_mask"),
            require_all_channels=require_all_channels,
        )
        scales = self.objective.asinh_scales.to(calibrated)[channel_index.long()]
        fractional_residual = (calibrated - target) / torch.sqrt(
            target.square() + scales.square()
        )
        valid = batch.get("valid_mask")
        fit_residual = (
            asinh_residual
            if valid is None
            else asinh_residual[valid.to(asinh_residual.device)]
        )
        metrics: dict[str, torch.Tensor] = {
            "asinh_rmse": fit_residual.square().mean().sqrt()
        }
        for index, channel in enumerate(self.channels_angstrom):
            selected = channel_index.long() == index
            if not torch.any(selected):
                continue
            metrics[f"{channel}_gain"] = self.calibration.gains[index]
        if result.channel_emission_contribution is None:
            contribution = result.emission_measure_contribution_cm5
        else:
            contribution = torch.gather(
                result.channel_emission_contribution,
                2,
                channel_index[:, None, None]
                .expand(-1, result.channel_emission_contribution.shape[1], 1)
                .long(),
            ).squeeze(2)
        diagnostics = {
            "prediction_raw": raw_prediction,
            "prediction_calibrated": calibrated,
            "target": target,
            "channel_angstrom": channel_angstrom,
            "channel_index": channel_index,
            "pixel_index": batch.get("pixel_index"),
            "image_index": batch["image_index"],
            "valid_mask": batch.get("valid_mask"),
            "asinh_residual": asinh_residual,
            "fractional_residual": fractional_residual,
            "contribution": contribution,
            "height_m": result.geometric_height_m,
            "temperature_k": result.temperature_k,
            "electron_density_m3": result.electron_density_m3,
        }
        return DataTermBatchResult(
            likelihood_loss=total,
            component_losses={
                f"channel_{channel}": value
                for channel, value in channel_components.items()
            },
            metrics=metrics,
            sample_count=int(raw_prediction.numel()),
            diagnostics=diagnostics,
        )

    def evaluate_batch(self, batch: Mapping[str, Any]) -> DataTermBatchResult:
        """Evaluate a training/validation batch containing every AIA channel."""

        return self._evaluate_batch(batch, require_all_channels=True)

    def evaluate_diagnostic_batch(
        self, batch: Mapping[str, Any]
    ) -> DataTermBatchResult:
        """Evaluate a bounded render batch, which may contain one channel only.

        Native AIA rasters can have different valid-pixel counts.  Diagnostics
        therefore stream them independently without weakening the strict,
        channel-balanced contract used by :meth:`evaluate_batch` for fitting.
        """

        return self._evaluate_batch(batch, require_all_channels=False)

    def nuisance_losses(self) -> Mapping[str, torch.Tensor]:
        if not self.calibration.enabled:
            return {}
        return {"calibration_prior": self.calibration.prior_loss()}

    def metadata(self) -> dict:
        return {
            "observation_id": self.observation_id,
            "observation_kind": self.observation_kind,
            "ray_samples": self.ray_samples,
            "height_sampling_power": self.height_sampling_power,
            "intensity_unit": "dimensionless",
            "intensity_scales_dn_s_pixel": self.intensity_scales.detach()
            .cpu()
            .tolist(),
            "response": self.response_metadata,
            "objective": dict(self.objective.configuration()),
            "calibration": self.calibration.metadata(),
        }


__all__ = ["AIAEUVObservationTerm"]
