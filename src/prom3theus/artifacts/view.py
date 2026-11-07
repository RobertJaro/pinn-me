"""Per-stream scientific evaluation facade over a reconstructed LTE model."""
from torch import nn


class StreamEvaluationView(nn.Module):
    """Expose physical fields and observation prediction through formal LTE integration."""

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
    def synthesis_window(self):
        return self.term.synthesis_window

    @property
    def local_coefficients(self):
        return self.synthesis_window.coefficients

    @property
    def velocity_synthesis_mode(self):
        return self.term.velocity_synthesis_mode.value

    @property
    def instrument_line_of_sight_velocity_correction_m_per_s(self):
        return self.term.instrument_line_of_sight_velocity_correction_m_per_s

    def predict(self, batch):
        """Query outgoing radiation and apply the stream's instrument operator."""
        return self.term.predict(batch)
