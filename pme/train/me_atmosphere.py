'''
Includes all the internal workings of a ME synthesis

The formalism follows the computations on p. 159 in Landi Degl'Innocenti & Landolfi 2004

Also a lot of inspiration is taken from AAR github: https://github.com/aasensio/milne/blob/master/maths.f90

'''
from functools import lru_cache

import torch
from astropy import constants as const
from astropy import units as u
from torch import nn

from pme.train.atomic_functions import load_zeeman_lookup, get_zeeman_lookup_id
from pme.train.profile_functions import Voigt, FaradayVoigt


class MEAtmosphere(nn.Module):
    ''' Class to contain the ME atmosphere properties'''

    def __init__(self, wavelength_center, j_up, j_low, g_up, g_low):
        super().__init__()

        self.voigt = Voigt()
        # Retain the legacy module/buffer names for existing checkpoints.
        # Synthesis uses voigt.forward_pair_unit to share its Faddeeva evaluation.
        self.faraday_voigt = FaradayVoigt()

        self.register_buffer('c', torch.tensor(const.c.to_value(u.m / u.s), dtype=torch.float32))  # Speed of light
        self.register_buffer('wavelength_center', torch.tensor(wavelength_center.to_value(u.m), dtype=torch.float32))
        self.register_buffer('j_up', torch.tensor(j_up, dtype=torch.float32))  # Upper level angular momentum
        self.register_buffer('j_low', torch.tensor(j_low, dtype=torch.float32))  # Lower level angular momentum
        self.register_buffer('g_up', torch.tensor(g_up, dtype=torch.float32))  # Lande factor for upper level
        self.register_buffer('g_low', torch.tensor(g_low, dtype=torch.float32))

        zeeman_strength_lookup = load_zeeman_lookup(j_up, j_low)
        zeeman_strength_lookup = {k: nn.Parameter(torch.tensor(v, dtype=torch.float32), requires_grad=False)
                                  for k, v in zeeman_strength_lookup.items()}
        self.zeeman_strength_lookup = nn.ParameterDict(zeeman_strength_lookup)
        self._build_transition_cache()

    def _build_transition_cache(self):
        # Atomic levels are fixed. Read their scalars once, never per synthesis.
        j_up, j_low = (float(value.detach().cpu()) for value in (self.j_up, self.j_low))
        assert (2 * j_up + 1) % 1 == 0, 'nUp must be integer'
        transitions, splittings = [], []
        for i_up in range(int(2 * j_up + 1)):
            m_up = j_up - i_up
            for component, delta_m in zip(('b', 'p', 'r'), (-1, 0, 1)):
                m_low = m_up + delta_m
                if abs(m_low) <= abs(j_low):
                    key = get_zeeman_lookup_id(j_up, j_low, m_up, m_low)
                    transitions.append((component, key))
                    splittings.append(self.g_up * m_up - self.g_low * m_low)
        self._transitions = tuple(transitions)
        # Derived data must move with the model, but add no checkpoint keys.
        values = torch.stack(splittings) if splittings else self.g_up.new_empty(0)
        self.register_buffer('_transition_splittings', values, persistent=False)

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse=recurse)
        # Recompute in the new dtype rather than widening rounded splittings.
        self._build_transition_cache()
        return self

    def _load_from_state_dict(self, *args, **kwargs):
        super()._load_from_state_dict(*args, **kwargs)
        # Rebuild from the loaded atomic constants, including legacy checkpoints.
        self._build_transition_cache()

    def calculate_voigt_faraday_profiles(self, nu, nu_m, damping, lambda_dop, **kwargs):
        """Synthesize physical profiles with nonnegative damping and unit sigma."""
        if not hasattr(self, '_transitions'):  # Legacy whole-module pickle.
            self._build_transition_cache()
        profiles = {}
        for (component, key), splitting in zip(self._transitions, self._transition_splittings.unbind()):
            shift = lambda_dop - splitting * nu_m
            phi, psi = self.voigt.forward_pair_unit(nu - shift, damping)
            strength = self.zeeman_strength_lookup[key]
            for prefix, value in (('phi', phi), ('psi', psi)):
                name = f'{prefix}_{component}'
                contribution = strength * value
                profiles[name] = profiles[name] + contribution if name in profiles else contribution
        for component in ('b', 'p', 'r'):
            for prefix in ('phi', 'psi'):
                name = f'{prefix}_{component}'
                if name not in profiles:
                    profiles[name] = torch.zeros_like(nu)
        return profiles

    def zeeman_strength(self, j_up, j_low, MUp, MLow):
        # avoid recomputing the same strength
        z_id = get_zeeman_lookup_id(j_up, j_low, MUp, MLow)
        return self.zeeman_strength_lookup[z_id]

    # Defining the propagation matrix elements from L^2 book
    def eta_I(self, phi_p, phi_r, phi_b, sin_inc2, cos_inc, kl, **kwargs):
        eta_I = (phi_p * sin_inc2 + (phi_r + phi_b) / 2 * (1 + cos_inc ** 2))
        eta_I *= kl
        return eta_I

    def eta_Q(self, phi_p, phi_r, phi_b, sin_inc2_cos2azi, kl, **kwargs):
        eta_Q = ((phi_p - 0.5 * (phi_r + phi_b)) * sin_inc2_cos2azi) * kl

        return eta_Q

    def eta_U(self, phi_p, phi_r, phi_b, sin_inc2_sin2azi, kl, **kwargs):
        eta_U = ((phi_p - 0.5 * (phi_r + phi_b)) * sin_inc2_sin2azi) * kl
        return eta_U

    def eta_V(self, phi_r, phi_b, cos_inc, kl, **kwargs):
        eta_V = (phi_r - phi_b) * cos_inc * kl
        return eta_V

    def rho_Q(self, psi_p, psi_r, psi_b, sin_inc2_cos2azi, kl, **kwargs):
        rho_Q = ((psi_p - 0.5 * (psi_r + psi_b)) * sin_inc2_cos2azi) * kl
        return rho_Q

    def rho_U(self, psi_p, psi_r, psi_b, sin_inc2_sin2azi, kl, **kwargs):
        rho_U = ((psi_p - 0.5 * (psi_r + psi_b)) * sin_inc2_sin2azi) * kl

        return rho_U

    def rho_V(self, psi_r, psi_b, cos_inc, kl, **kwargs):
        rho_V = kl * (psi_r - psi_b) * cos_inc

        return rho_V

    def delta(self, eta_I, eta_Q, eta_U, eta_V, rho_Q, rho_U, rho_V, **kwargs):
        delta = ((1 + eta_I) ** 2 * ((1 + eta_I) ** 2
                                     - eta_Q ** 2 - eta_U ** 2 - eta_V ** 2
                                     + rho_Q ** 2 + rho_U ** 2 + rho_V ** 2)
                 - (eta_Q * rho_Q + eta_U * rho_U + eta_V * rho_V) ** 2)
        return delta

    def compute_I(self, b0, b1, delta, mu, eta_I, rho_Q, rho_U, rho_V, **kwargs):
        stokes_i = b0 + mu * b1 / delta * (
            (1 + eta_I) * ((1 + eta_I) ** 2 + rho_Q ** 2 + rho_U ** 2 + rho_V ** 2)
        )
        return stokes_i

    def compute_Q(self, b1, delta, mu, eta_I, rho_Q, rho_U, rho_V, eta_Q, eta_V, eta_U, **kwargs):
        Q = - mu * b1 / delta * ((1 + eta_I) ** 2 * eta_Q
                                 + (1 + eta_I) * (eta_V * rho_U - eta_U * rho_V)
                                 + rho_Q * (eta_Q * rho_Q + eta_U * rho_U + eta_V * rho_V))
        return Q

    def compute_U(self, b1, delta, mu, eta_I, rho_Q, rho_U, rho_V, eta_Q, eta_V, eta_U, **kwargs):
        U = -1 * mu * b1 / delta * ((1 + eta_I) ** 2 * eta_U
                                    + (1 + eta_I) * (eta_Q * rho_V - eta_V * rho_Q)
                                    + rho_U * (eta_Q * rho_Q + eta_U * rho_U + eta_V * rho_V))
        return U

    def compute_V(self, b1, delta, mu, eta_I, rho_Q, rho_U, rho_V, eta_Q, eta_V, eta_U, **kwargs):
        V = - mu * b1 / delta * ((1 + eta_I) ** 2 * eta_V
                                 + (1 + eta_I) * (eta_U * rho_Q - eta_Q * rho_U)
                                 + rho_V * (eta_Q * rho_Q + eta_U * rho_U + eta_V * rho_V))
        return V

    def nu_m(self, b_field, d_lambda, **kwargs):
        dlambda_B = 4.6686e-3 * (self.wavelength_center ** 2) * b_field
        return dlambda_B / d_lambda

    def lambda_dop(self, vdop, d_lambda, **kwargs):
        return self.wavelength_center * vdop / self.c / d_lambda

    def d_lambda(self, vmac, **kwargs):
        return self.wavelength_center * vmac / self.c

    def nu(self, d_lambda, wavelength_grid, **kwargs):
        return wavelength_grid / d_lambda

    def _forward_monochromatic(self, wavelength_grid, b_field, sin_inc2_cos2azi, sin_inc2_sin2azi,
                               sin_inc2, cos_inc,
                               vmac, damping, b0, b1, mu, vdop, kl, **kwargs):
        # sin_inc2 = sin(inc) ** 2
        # The azimuth inputs already contain sin(inc) ** 2, avoiding a
        # singular division by the transverse field strength.
        # cos_inc = cos(inc)
        # init state
        state = {'b_field': b_field,
                 'sin_inc2': sin_inc2, 'cos_inc': cos_inc,
                 'sin_inc2_cos2azi': sin_inc2_cos2azi,
                 'sin_inc2_sin2azi': sin_inc2_sin2azi,
                 'vmac': vmac, 'damping': damping,
                 'b0': b0, 'b1': b1, 'mu': mu, 'vdop': vdop, 'kl': kl,
                 'wavelength_grid': wavelength_grid}

        # base profile properties
        state['d_lambda'] = self.d_lambda(**state)
        state['nu'] = self.nu(**state)
        state['nu_m'] = self.nu_m(**state)
        state['lambda_dop'] = self.lambda_dop(**state)
        # voigt and faraday voigt profiles
        profiles = self.calculate_voigt_faraday_profiles(**state)
        state.update(profiles)
        mode = getattr(self, '_compile_solver_mode', None)
        if mode is not None and self.training and torch.is_grad_enabled():
            # Compile only real tensor algebra. Replay, Faddeeva and instrument
            # response integration remain eager. Validation/export use eager too.
            return compiled_profile_solver(mode)(**state)
        return self._solve_profiles(**state)

    @staticmethod
    def _solve_profiles(phi_p, phi_r, phi_b, psi_p, psi_r, psi_b,
                        sin_inc2, cos_inc, sin_inc2_cos2azi, sin_inc2_sin2azi,
                        kl, b0, b1, mu, **kwargs):
        # Shared propagation coefficients and Unno-Rachkovsky intermediates.
        phi_sigma = (phi_r + phi_b) * 0.5
        phi_linear = phi_p - phi_sigma
        psi_linear = psi_p - (psi_r + psi_b) * 0.5
        eta_i = (phi_p * sin_inc2 + phi_sigma * (1 + cos_inc.square())) * kl
        eta_q = phi_linear * sin_inc2_cos2azi * kl
        eta_u = phi_linear * sin_inc2_sin2azi * kl
        eta_v = (phi_r - phi_b) * cos_inc * kl
        rho_q = psi_linear * sin_inc2_cos2azi * kl
        rho_u = psi_linear * sin_inc2_sin2azi * kl
        rho_v = kl * (psi_r - psi_b) * cos_inc
        k = 1 + eta_i
        k2 = k.square()
        rho_q2, rho_u2, rho_v2 = rho_q.square(), rho_u.square(), rho_v.square()
        dot = eta_q * rho_q + eta_u * rho_u + eta_v * rho_v
        delta = k2 * (k2 - eta_q.square() - eta_u.square() - eta_v.square()
                      + rho_q2 + rho_u2 + rho_v2) - dot.square()
        source = mu * b1 / delta
        polarized_source = -source
        intensity = b0 + source * (k * (k2 + rho_q2 + rho_u2 + rho_v2))
        q = polarized_source * (k2 * eta_q + k * (eta_v * rho_u - eta_u * rho_v) + rho_q * dot)
        u = polarized_source * (k2 * eta_u + k * (eta_q * rho_v - eta_v * rho_q) + rho_u * dot)
        v = polarized_source * (k2 * eta_v + k * (eta_u * rho_q - eta_q * rho_u) + rho_v * dot)
        return intensity, q, u, v

    def forward(self, wavelength_grid, b_field, sin_inc2_cos2azi, sin_inc2_sin2azi,
                sin_inc2, cos_inc,
                vmac, damping, b0, b1, mu, vdop, kl,
                spectral_offsets=None, spectral_weights=None, continuum_weights=None, **kwargs):
        atmosphere_parameters = {
            'b_field': b_field,
            'sin_inc2_cos2azi': sin_inc2_cos2azi,
            'sin_inc2_sin2azi': sin_inc2_sin2azi,
            'sin_inc2': sin_inc2,
            'cos_inc': cos_inc,
            'vmac': vmac,
            'damping': damping,
            'b0': b0,
            'b1': b1,
            'mu': mu,
            'vdop': vdop,
            'kl': kl,
            **kwargs,
        }
        response_values = (spectral_offsets, spectral_weights, continuum_weights)
        if all(value is None for value in response_values):
            return self._forward_monochromatic(wavelength_grid=wavelength_grid, **atmosphere_parameters)
        if any(value is None for value in response_values):
            raise ValueError(
                'spectral_offsets, spectral_weights, and continuum_weights must be provided together.'
            )
        if spectral_offsets.ndim == 2:
            spectral_offsets = spectral_offsets.unsqueeze(0)
        if spectral_weights.ndim == 2:
            spectral_weights = spectral_weights.unsqueeze(0)
        if spectral_offsets.shape != spectral_weights.shape:
            raise ValueError('spectral_offsets and spectral_weights must have matching shapes.')
        if spectral_offsets.ndim != 3:
            raise ValueError('Batch spectral responses must have shape [batch, filter, sample].')
        if continuum_weights.shape != spectral_weights.shape[:-1]:
            raise ValueError('continuum_weights must have shape [batch, filter].')
        if spectral_offsets.shape[-2] not in (1, wavelength_grid.shape[-1]):
            raise ValueError(
                f'Tabulated profile has {spectral_offsets.shape[-2]} filters, but wavelength grid has '
                f'{wavelength_grid.shape[-1]} positions.'
            )

        sampled_grid = wavelength_grid[..., :, None] + spectral_offsets
        n_profile_samples = sampled_grid.shape[-1]
        flat_grid = sampled_grid.reshape(*wavelength_grid.shape[:-1], -1)
        monochromatic = self._forward_monochromatic(wavelength_grid=flat_grid, **atmosphere_parameters)

        output_shape = (*wavelength_grid.shape, n_profile_samples)
        integrated = [(component.reshape(output_shape) * spectral_weights).sum(dim=-1)
                      for component in monochromatic]
        continuum_intensity = b0 + mu * b1
        integrated[0] = integrated[0] + continuum_weights * continuum_intensity
        return tuple(integrated)


@lru_cache(maxsize=4)
def compiled_profile_solver(mode='default'):
    """Process-local callable, excluded from state dicts and module exports.

    Compilation is lazy on the first training invocation. Failures are surfaced
    rather than silently falling back to eager execution.
    """
    return torch.compile(MEAtmosphere._solve_profiles, backend='inductor',
                         mode=mode, fullgraph=True)


class HMIMEAtmosphere(MEAtmosphere):
    ''' Class to contain the HMI ME atmosphere properties'''

    def __init__(self, **kwargs):
        j_up = 1.0
        j_low = 0.0
        g_up = 2.50
        g_low = 0.0
        super().__init__(j_up=j_up, j_low=j_low, g_up=g_up, g_low=g_low, **kwargs)

    def forward(self, sin_inc2_sin2azi, sin_inc2_cos2azi, **kwargs):
        # apply angle transformation to the HMI polarizer
        # sin(2*(x + pi/2)) = sin(2*x + pi) = -sin(2*x)
        # cos(2*(x + pi/2)) = cos(2*x + pi) = -cos(2*x)
        sin_inc2_sin2azi = -sin_inc2_sin2azi
        sin_inc2_cos2azi = -sin_inc2_cos2azi
        return super().forward(
            sin_inc2_sin2azi=sin_inc2_sin2azi,
            sin_inc2_cos2azi=sin_inc2_cos2azi,
            **kwargs,
        )

class PHIMEAtmosphere(MEAtmosphere):
    ''' Class to contain the PHI ME atmosphere properties'''

    def __init__(self, **kwargs):
        j_up = 1.0
        j_low = 0.0
        g_up = 2.50
        g_low = 0.0
        super().__init__(j_up=j_up, j_low=j_low, g_up=g_up, g_low=g_low, **kwargs)

    def forward(self, **kwargs):
        return super().forward(**kwargs)
