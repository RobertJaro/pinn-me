"""Shared spherical parameter transformations used by training and evaluation."""

import torch

from pme.data.differential_rotation import carrington_rotation_velocity
from pme.data.util import cartesian_to_spherical
from pme.train.util import acos_safe, atan2_safe


def transform_spherical_parameters(output, coords, cartesian_to_spherical_transform,
                                   rtp_to_img_transform, meters_per_ds, seconds_per_dt):
    """Transform model-frame Cartesian fields into spherical and image frames."""
    b_xyz = torch.cat([output['b_x'], output['b_y'], output['b_z']], dim=-1)
    b_rtp = torch.einsum("...ij,...j->...i", cartesian_to_spherical_transform, b_xyz)
    b_img = torch.einsum("...ij,...j->...i", rtp_to_img_transform, b_rtp)

    b_field2 = b_img.square().sum(dim=-1, keepdim=True)
    b_field = torch.linalg.vector_norm(b_img, dim=-1, keepdim=True)
    transverse_b2 = b_img[..., 0:1].square() + b_img[..., 1:2].square()
    field_denominator = b_field2 + 1e-8
    sin_inc2 = transverse_b2 / field_denominator
    cos_inc = b_img[..., 2:3] / (b_field + 1e-8)
    sin_inc2_sin2azi = -2 * b_img[..., 0:1] * b_img[..., 1:2] / field_denominator
    sin_inc2_cos2azi = -(b_img[..., 0:1].square() - b_img[..., 1:2].square()) / field_denominator
    inc = acos_safe(cos_inc)
    azi = atan2_safe(-b_img[..., 0:1], b_img[..., 1:2])

    spherical_coords = cartesian_to_spherical(coords[..., 1:], torch)
    latitude = torch.pi / 2 - spherical_coords[..., 1]
    radius = spherical_coords[..., 0] * meters_per_ds
    v_rot = carrington_rotation_velocity(latitude, radius)
    v_rot = v_rot / meters_per_ds * seconds_per_dt

    v_xyz = torch.cat([output['v_x'], output['v_y'], output['v_z']], dim=-1)
    v_rtp = torch.einsum("...ij,...j->...i", cartesian_to_spherical_transform, v_xyz)
    v_rtp_inertial = torch.stack([v_rtp[..., 0], v_rtp[..., 1], v_rtp[..., 2] + v_rot], dim=-1)
    v_img = torch.einsum("...ij,...j->...i", rtp_to_img_transform, v_rtp_inertial)
    vdop = -v_img[..., 2:3]

    return {
        'b_field': b_field,
        'sin_inc2_sin2azi': sin_inc2_sin2azi,
        'sin_inc2_cos2azi': sin_inc2_cos2azi,
        'azi': azi,
        'sin_inc2': sin_inc2,
        'cos_inc': cos_inc,
        'inc': inc,
        'vdop': vdop,
        'v_rtp': v_rtp,
        'b_rtp': b_rtp,
        'v_img': v_img,
        'b_img': b_img,
        'v_xyz': v_xyz,
        'b_xyz': b_xyz,
    }


def scale_spherical_forward_parameters(output, transformed_output, v_obs_los,
                                       gauss_per_dB, meters_per_ds, seconds_per_dt):
    """Convert transformed model outputs to the units expected by ME synthesis."""
    vdop = transformed_output['vdop'] * meters_per_ds / seconds_per_dt + v_obs_los
    return {
        'b_field': transformed_output['b_field'] * gauss_per_dB,
        'sin_inc2': transformed_output['sin_inc2'],
        'cos_inc': transformed_output['cos_inc'],
        'inc': transformed_output['inc'],
        'sin_inc2_sin2azi': transformed_output['sin_inc2_sin2azi'],
        'sin_inc2_cos2azi': transformed_output['sin_inc2_cos2azi'],
        'azi': transformed_output['azi'],
        'vdop': vdop,
        'vmac': output['vmac'],
        'damping': output['damping'],
        'b0': output['b0'],
        'b1': output['b1'],
        'kl': output['kl'],
    }


def correct_spherical_limb_effects(parameters, limb_correction):
    """Apply the optional learned limb corrections without mutating inputs."""
    corrected = dict(parameters)
    corrected['b0'] = parameters['b0'] * limb_correction['c_b0']
    corrected['b1'] = parameters['b1'] * limb_correction['c_b1']
    corrected['vdop'] = parameters['vdop'] + limb_correction['c_vdop']
    return corrected


def field_free_forward_parameters(parameters):
    """Return the same atmosphere with a zero magnetic field."""
    field_free = dict(parameters)
    field_free['b_field'] = torch.zeros_like(parameters['b_field'])
    return field_free


def mix_magnetic_filling_factor(magnetic_stokes, field_free_stokes, filling_factor):
    """Mix magnetic and field-free Stokes profiles using an observer-frame factor."""
    if filling_factor.shape != magnetic_stokes.shape[:-2] + (1,):
        raise ValueError(
            'filling_factor must have shape [...,1] matching the Stokes samples; '
            f'got {tuple(filling_factor.shape)} for {tuple(magnetic_stokes.shape)}.'
        )
    if field_free_stokes.shape != magnetic_stokes.shape:
        raise ValueError('Magnetic and field-free Stokes profiles must have matching shapes.')
    factor = filling_factor.unsqueeze(-1)
    return field_free_stokes + factor * (magnetic_stokes - field_free_stokes)
