"""Per-stream scientific evaluation facade over a reconstructed model."""
from torch import nn


class StreamEvaluationView(nn.Module):
    def __init__(self, model, stream_id):
        super().__init__()
        self.model = model
        self.stream_id = stream_id

    @property
    def atmosphere_model(self):
        return self.model.atmosphere_model

    @property
    def term(self):
        return self.model.terms[self.stream_id]

    @property
    def synthesizer(self):
        return self.term.synthesizer

    @property
    def forward_composition(self):
        return self.term._composition

    @property
    def velocity_synthesis_mode(self):
        return self.term.velocity_synthesis_mode.value

    @property
    def instrument_line_of_sight_velocity_correction_m_per_s(self):
        return self.term.instrument_line_of_sight_velocity_correction_m_per_s

    def sample_depth_grid(self, *, randomize=False):
        return self.forward_composition.sample_depth_grid(
            self.term.coarse_depth_grid, randomize=randomize
        )

    def synthesize(self, coordinates, **kwargs):
        return self.forward_composition.synthesize(
            coordinates, runtime=self.term._runtime(), **kwargs
        )
