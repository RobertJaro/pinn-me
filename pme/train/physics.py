import torch

from pme.data.util import cartesian_to_spherical
from pme.model import jacobian


def compute_physics_losses(b, v, a_jac_matrix, coords):
    # compute B derivatives
    jac_matrix = jacobian(b, coords)
    dBx_dt = jac_matrix[..., 0, 0]
    dBx_dx = jac_matrix[..., 0, 1]
    dBx_dy = jac_matrix[..., 0, 2]
    dBx_dz = jac_matrix[..., 0, 3]
    dBy_dt = jac_matrix[..., 1, 0]
    dBy_dx = jac_matrix[..., 1, 1]
    dBy_dy = jac_matrix[..., 1, 2]
    dBy_dz = jac_matrix[..., 1, 3]
    dBz_dt = jac_matrix[..., 2, 0]
    dBz_dx = jac_matrix[..., 2, 1]
    dBz_dy = jac_matrix[..., 2, 2]
    dBz_dz = jac_matrix[..., 2, 3]
    # compute V derivatives
    v_jac = jacobian(v, coords)
    dVx_dt = v_jac[:, 0, 0]
    dVx_dx = v_jac[:, 0, 1]
    dVx_dy = v_jac[:, 0, 2]
    dVx_dz = v_jac[:, 0, 3]
    dVy_dt = v_jac[:, 1, 0]
    dVy_dx = v_jac[:, 1, 1]
    dVy_dy = v_jac[:, 1, 2]
    dVy_dz = v_jac[:, 1, 3]
    dVz_dt = v_jac[:, 2, 0]
    dVz_dx = v_jac[:, 2, 1]
    dVz_dy = v_jac[:, 2, 2]
    dVz_dz = v_jac[:, 2, 3]

    # compute j = curl(B)
    rot_x = dBz_dy - dBy_dz
    rot_y = dBx_dz - dBz_dx
    rot_z = dBy_dx - dBx_dy
    j = torch.stack([rot_x, rot_y, rot_z], -1)

    # compute induction loss
    div_V = (dVx_dx + dVy_dy + dVz_dz)[..., None]
    div_B = (dBx_dx + dBy_dy + dBz_dz)[..., None]
    dB_dt = torch.stack([dBx_dt, dBy_dt, dBz_dt], -1)
    B_nabla_V = torch.stack([b[:, 0] * dVx_dx + b[:, 1] * dVx_dy + b[:, 2] * dVx_dz,
                             b[:, 0] * dVy_dx + b[:, 1] * dVy_dy + b[:, 2] * dVy_dz,
                             b[:, 0] * dVz_dx + b[:, 1] * dVz_dy + b[:, 2] * dVz_dz, ], -1)
    V_nabla_B = torch.stack([v[:, 0] * dBx_dx + v[:, 1] * dBx_dy + v[:, 2] * dBx_dz,
                             v[:, 0] * dBy_dx + v[:, 1] * dBy_dy + v[:, 2] * dBy_dz,
                             v[:, 0] * dBz_dx + v[:, 1] * dBz_dy + v[:, 2] * dBz_dz, ], -1)

    vxB = torch.cross(v, b, dim=-1)

    if a_jac_matrix is not None:
        # A derivatives
        dAx_dt = a_jac_matrix[:, 0, 0]
        dAx_dx = a_jac_matrix[:, 0, 1]
        dAx_dy = a_jac_matrix[:, 0, 2]
        dAx_dz = a_jac_matrix[:, 0, 3]
        dAy_dt = a_jac_matrix[:, 1, 0]
        dAy_dx = a_jac_matrix[:, 1, 1]
        dAy_dy = a_jac_matrix[:, 1, 2]
        dAy_dz = a_jac_matrix[:, 1, 3]
        dAz_dt = a_jac_matrix[:, 2, 0]
        dAz_dx = a_jac_matrix[:, 2, 1]
        dAz_dy = a_jac_matrix[:, 2, 2]
        dAz_dz = a_jac_matrix[:, 2, 3]
        dA_dt = torch.stack([dAx_dt, dAy_dt, dAz_dt], -1)

        induction_equation = dA_dt - vxB
        induction_loss = induction_equation.pow(2).sum(-1)

        # compute Coulomb gauge condition
        gauge_loss = (dAx_dx + dAy_dy + dAz_dz).pow(2)
    else:
        induction_rhs = B_nabla_V - V_nabla_B - b * div_V + v * div_B
        induction_equation = dB_dt - induction_rhs
        induction_loss = induction_equation.pow(2).sum(-1)
        gauge_loss = torch.zeros_like(induction_loss)

    # compute divergence loss
    divergence_loss = div_B.pow(2).sum(-1)

    # compute force-free condition
    force_free_loss = torch.cross(j, b, dim=-1).pow(2).sum(-1)

    # compute potential field loss
    potential_loss = j.pow(2).sum(-1)

    # x = r * sin(t) * cos(p)
    # y = r * sin(t) * sin(p)
    # z = r * cos(t)
    spherical_coords = cartesian_to_spherical(coords[..., 1:], torch)
    dx_dr = torch.sin(spherical_coords[..., 1]) * torch.cos(spherical_coords[..., 2])
    dy_dr = torch.sin(spherical_coords[..., 1]) * torch.sin(spherical_coords[..., 2])
    dz_dr = torch.cos(spherical_coords[..., 1])
    dBx_dr = dBx_dx * dx_dr + dBx_dy * dy_dr + dBx_dz * dz_dr
    dBy_dr = dBy_dx * dx_dr + dBy_dy * dy_dr + dBy_dz * dz_dr
    dBz_dr = dBz_dx * dx_dr + dBz_dy * dy_dr + dBz_dz * dz_dr
    dB_dr = torch.stack([dBx_dr, dBy_dr, dBz_dr], -1)

    return {'divergence': divergence_loss,
            'force_free': force_free_loss,
            'potential': potential_loss,
            'j': j.pow(2).sum(-1),
            'induction': induction_loss,
            'gauge': gauge_loss,
            'dB_dt': dB_dt.pow(2).sum(-1).pow(0.5),
            'curl_VxB': vxB.pow(2).sum(-1).pow(0.5),
            'dB_dr': dB_dr.pow(2).sum(-1).pow(0.5),
            }
