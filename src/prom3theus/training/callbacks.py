"""PyTorch Lightning orchestration for LTE training diagnostics."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch
from pytorch_lightning import Callback

from prom3theus.diagnostics.evaluation import AtmosphereEvaluator, AtmosphereSampling
from prom3theus.diagnostics.rendering import AtmosphereRenderer
from prom3theus.diagnostics.sampling import ValidationSampleCollector, display_grid


class AtmosphereVisualizationCallback(Callback):
    """Schedule bounded atmosphere and Stokes diagnostics during training."""

    def __init__(
        self,
        output_directory,
        *,
        every_n_validations: int = 5,
        ray_sampling: Mapping | None = None,
        slice_sampling: Mapping | None = None,
        dpi: int = 180,
        include_initial: bool = True,
        meridional_slice: Mapping | None = None,
    ):
        super().__init__()
        if every_n_validations < 1:
            raise ValueError("every_n_validations must be positive.")

        self.sampling = AtmosphereSampling.from_options(
            ray_sampling=ray_sampling,
            slice_sampling=slice_sampling,
            meridional_slice=meridional_slice,
        )
        self.evaluator = AtmosphereEvaluator(self.sampling)
        self.renderer = AtmosphereRenderer(
            output_directory,
            dpi=dpi,
            evaluator=self.evaluator,
        )
        self.collector = ValidationSampleCollector(self.sampling.max_profile_samples)

        self.output_directory = self.renderer.output_directory
        self.every_n_validations = int(every_n_validations)
        self.dpi = int(dpi)
        self.include_initial = bool(include_initial)
        self._last_rendered_step: int | None = None
        self._validation_event_count = 0
        self._active_validation_event: int | None = None
        self._collect_validation_event = False

    def _display_indices(self, trainer, raster) -> tuple[np.ndarray, np.ndarray]:
        return display_grid(trainer, raster, self.sampling.max_ray_pixels)

    def render(
        self,
        trainer,
        pl_module,
        raster,
        *,
        label: str,
        rows: np.ndarray | None = None,
        columns: np.ndarray | None = None,
    ) -> list[Path]:
        """Evaluate and save one complete atmosphere diagnostic snapshot."""

        return self.renderer.render_atmosphere(
            trainer,
            pl_module,
            raster,
            label=label,
            rows=rows,
            columns=columns,
        )

    @staticmethod
    def _raster(trainer):
        data_module = getattr(trainer, "datamodule", None)
        raster = getattr(data_module, "raster", None)
        if raster is None:
            raise RuntimeError(
                "AtmosphereVisualizationCallback requires a data module whose "
                "setup() method has populated a ray raster."
            )
        return raster

    def on_fit_start(self, trainer, pl_module) -> None:
        if self.include_initial and bool(getattr(trainer, "is_global_zero", True)):
            step = int(getattr(trainer, "global_step", 0))
            self.render(trainer, pl_module, self._raster(trainer), label="initial")
            self._last_rendered_step = step

    def on_validation_epoch_start(self, trainer, pl_module) -> None:
        del pl_module
        self.collector.reset()
        if bool(getattr(trainer, "sanity_checking", False)):
            self._active_validation_event = None
            self._collect_validation_event = False
            return
        self._validation_event_count += 1
        self._active_validation_event = self._validation_event_count
        self._collect_validation_event = (
            self._active_validation_event % self.every_n_validations == 0
        )

    def on_validation_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
        dataloader_idx=0,
    ) -> None:
        del trainer, batch, batch_idx, dataloader_idx
        if not self._collect_validation_event or not isinstance(outputs, dict):
            return
        required = ("stokes_pred", "stokes_reference", "pixel_index")
        if not all(name in outputs for name in required):
            return
        self.collector.add(
            outputs["stokes_pred"],
            outputs["stokes_reference"],
            outputs["pixel_index"],
            pl_module.wavelength_angstrom,
            pl_module.wavelength_weights,
        )

    def _local_validation_payload(self) -> dict[str, torch.Tensor] | None:
        return self.collector.local_payload()

    def _merge_validation_payloads(self, payloads) -> dict[str, torch.Tensor] | None:
        return self.collector.merge_payloads(payloads)

    def _gather_validation_outputs(self) -> dict[str, torch.Tensor] | None:
        return self.collector.gather()

    def on_validation_end(self, trainer, pl_module) -> None:
        """Publish scheduled diagnostics after Lightning finalizes validation."""

        if bool(getattr(trainer, "sanity_checking", False)):
            return
        if not self._collect_validation_event:
            return
        if self._active_validation_event is None:
            raise RuntimeError(
                "Validation ended without a matching validation start event."
            )

        outputs = self._gather_validation_outputs()
        if not bool(getattr(trainer, "is_global_zero", True)):
            return
        if outputs is None:
            raise RuntimeError(
                "Scheduled LTE validation produced no Stokes diagnostic payload."
            )

        step = int(getattr(trainer, "global_step", 0))
        label = f"validation_{self._active_validation_event:04d}_step_{step:08d}"
        raster = self._raster(trainer)
        rows, columns = self._display_indices(trainer, raster)
        self.renderer.render_stokes_validation(
            trainer,
            outputs,
            raster,
            pl_module.wavelength_angstrom,
            label,
            rows=rows,
            columns=columns,
            exclusion_windows_angstrom=(pl_module.wavelength_exclude_windows_angstrom),
            line_centers_angstrom=(
                line.wavelength_air_angstrom
                for line in getattr(
                    getattr(pl_module, "synthesizer", None), "lines", ()
                )
            ),
        )
        self.render(
            trainer,
            pl_module,
            raster,
            label=label,
            rows=rows,
            columns=columns,
        )
        self._last_rendered_step = step

    def on_fit_end(self, trainer, pl_module) -> None:
        if not bool(getattr(trainer, "is_global_zero", True)):
            return
        step = int(getattr(trainer, "global_step", 0))
        if self._last_rendered_step == step:
            return
        self.render(
            trainer,
            pl_module,
            self._raster(trainer),
            label=f"final_step_{step:08d}",
        )
        self._last_rendered_step = step


__all__ = ["AtmosphereVisualizationCallback"]
