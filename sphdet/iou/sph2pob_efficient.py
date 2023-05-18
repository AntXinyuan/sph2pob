import torch
import torch.nn.functional as F
from torch import cos, sin


# ---------------------------------------------------------------------------- #
#                             Sph2Pob BoxTransfrom                             #
# ---------------------------------------------------------------------------- #
def sph2pob_efficient(sph_gt, sph_pred, rbb_angle_version='deg', rbb_edge='arc', rbb_angle='equator'):
    """Transform spherical boxes to planar oriented boxes.
    NOTE: It's a efficient implement of Sph2Pob, and the results is equivalent to original algorithm.

    Args:
        sph_gt (torch.Tensor): N x 4(5), deg
        sph_pred (torch.Tensor): N x 4(5), deg
        rbb_angle_version (str, optional): The angle version of output boxes. Defaults to 'deg'.
        rbb_edge (str, optional): Algorithm option. Defaults to 'arc'.
        rbb_angle (str, optional): Algorithm option. Defaults to 'equator'.

    Returns:
        plannar_gt (torch.tensor): N x 5
        plannar_pred (torch.tensor): N x 5
    """

    assert rbb_angle_version in ['deg', 'rad']
    assert rbb_edge  in ['arc', 'chord', 'tangent']
    assert rbb_angle in ['equator', 'project']

    sph_gt   = torch.deg2rad(sph_gt)
    sph_pred = torch.deg2rad(sph_pred)

    theta_g, phi_g, alpha_g, beta_g = torch.chunk(sph_gt[:, :4], chunks=4, dim=1)   # Nx1
    theta_p, phi_p, alpha_p, beta_p = torch.chunk(sph_pred[:, :4], chunks=4, dim=1) # Nx1

    sin_theta, cos_theta = sin(theta_g), cos(theta_g)
    sin_phi, cos_phi = sin(phi_g), cos(phi_g)
    sin_cos_cache = sin_theta, cos_theta, sin_phi, cos_phi
    coor_g = compute_3d_coordinate(theta_g, phi_g, sin_cos_cache) # Nx3x1
    dir_g = compute_tangential_direction_along_longitude(theta_g, phi_g, sin_cos_cache) # Nx3x1

    sin_theta, cos_theta = sin(theta_p), cos(theta_p)
    sin_phi, cos_phi = sin(phi_p), cos(phi_p)
    sin_cos_cache = sin_theta, cos_theta, sin_phi, cos_phi
    coor_p = compute_3d_coordinate(theta_p, phi_p, sin_cos_cache) # Nx3x1
    dir_p = compute_tangential_direction_along_longitude(theta_p, phi_p, sin_cos_cache) # Nx3x1

    sin_theta = cos_theta = sin_phi = cos_phi = sin_cos_cache = None

    dir_z = torch.cross(coor_g, coor_p, dim=1)
    dir_ref = (coor_g + coor_p) / 2
    arc = compute_angle_between_direction(coor_g, coor_p)

    angle_g_ = compute_internal_angle(dir_g, dir_ref, dir_z, rbb_angle) # Nx1
    angle_p_ = compute_internal_angle(dir_p, dir_ref, dir_z, rbb_angle) # Nx1
    if sph_gt.size(1) == 5 and sph_pred.size(1) == 5:
        angle_g_ -= sph_gt[:, -1].view((-1, 1))
        angle_p_ -= sph_pred[:, -1].view((-1, 1))

    zeros = torch.zeros_like(angle_g_)
    theta_g_, phi_g_ = zeros, zeros
    theta_p_, phi_p_ = arc , zeros

    alpha_g_ = compute_edge_length(alpha_g, rbb_edge)
    beta_g_  = compute_edge_length(beta_g, rbb_edge)
    alpha_p_ = compute_edge_length(alpha_p, rbb_edge)
    beta_p_  = compute_edge_length(beta_p, rbb_edge)

    plannar_gt = torch.concat([theta_g_, phi_g_, alpha_g_, beta_g_, angle_g_], dim=1)
    plannar_pred = torch.concat([theta_p_, phi_p_, alpha_p_, beta_p_, angle_p_], dim=1)

    plannar_gt, plannar_pred = standardize_rotated_box(plannar_gt, plannar_pred, rbb_angle_version)

    return plannar_gt, plannar_pred


# ---------------------------------------------------------------------------- #
#                            Auxiliary ComputeMethod                           #
# ---------------------------------------------------------------------------- #

# --------------------------- Helper ComputeMethod -------------------------- #
def compute_internal_angle(dir_, dir_ref, dir_z_, rbb_angle='equator'):
    """Compute signed angle between given direction dir_ and reference direction z-axis on prejected yOz-surface.  

    Args:
        dir_ (torch.Tensor): Nx3x1

    Returns:
        angle_ (torch.Tensor): Nx1
    """
    assert rbb_angle in ['equator', 'project']

    if rbb_angle == 'project':
        dir_[:, 0, :] = 0
    angle_ = compute_angle_between_direction(dir_, dir_z_) # Nx1
    sign_mask = compute_clockwise_or_anticlockwise_between_direction(dir_z_, dir_, dir_ref) # N
    angle_ = angle_ * sign_mask.view((-1, 1)) # Nx3x1
    return angle_


def compute_edge_length(fov_angle, mode='arc'):
    if mode == 'arc':
        return fov_angle
    elif mode == 'tangent':
        return 2 * torch.tan(fov_angle / 2)
    elif mode =='chord':
        return 2 * torch.sin(fov_angle / 2)
    else:
        raise NotImplemented('Not supported edge mode!')

# ---------------------------- Basic ComputeMethod --------------------------- #
def compute_tangential_direction_along_longitude(theta, phi, sin_cos_cache=None):
    """Compute tangential direction along longitude, where
        x = sin(\phi)cos(\theta)
        y = sin(\phi)sin(\theta)
        z = cos(\phi)
    and just \phi is variable.

    Args:
        theta (torch.Tensor): N
        phi (torch.Tensor): N

    Returns:
        direction (torch.Tensor): Nx3
    """
    if sin_cos_cache is None:
        from torch import cos, sin
        sin_theta, cos_theta = sin(theta), cos(theta)
        sin_phi, cos_phi = sin(phi), cos(phi)
    else:
        sin_theta, cos_theta, sin_phi, cos_phi = sin_cos_cache

    dx =  cos_phi * cos_theta
    dy =  cos_phi * sin_theta
    dz = -sin_phi
    direction = torch.stack([dx, dy, dz], dim=1)
    return direction


def compute_3d_coordinate(theta, phi, sin_cos_cache=None):
    """Compute 3D corordinate (x, y, z) of a sperical point at (theta, phi), where
        x = sin(\phi)cos(\theta)
        y = sin(\phi)sin(\theta)
        z = cos(\phi)

    Args:
        theta (torch.Tensor): N
        phi (torch.Tensor): N

    Returns:
        direction (torch.Tensor): Nx3
    """
    if sin_cos_cache is None:
        from torch import cos, sin
        sin_theta, cos_theta = sin(theta), cos(theta)
        sin_phi, cos_phi = sin(phi), cos(phi)
    else:
        sin_theta, cos_theta, sin_phi, cos_phi = sin_cos_cache
    x = sin_phi * cos_theta
    y = sin_phi * sin_theta
    z = cos_phi
    direction = torch.stack([x, y, z], dim=1)
    return direction


def compute_spherical_coordinate(coor):
    """Compute sperical coordinate (theta, phi) given a 3D coordinate of a point (x, y, z).
    It should be the inverse operator for compute_3d_coordinate(theta, phi).

    Args:
        coor (torch.Tensor): Nx3x1

    Returns:
        theta (torch.Tensor): Nx1
        phi (torch.Tensor): Nx1
    """
    z = torch.tensor([0., 0., 1.], device=coor.device)[None, :, None]
    phi = compute_angle_between_direction(coor, z)
    
    # Don't delete the clone operator to autograd works, see also 
    # https://pytorch.org/docs/stable/autograd.html#in-place-operations-on-tensors
    # https://zhuanlan.zhihu.com/p/38475183 
    coor_xy = coor.clone()

    coor_xy[:, 2, ...] = 0 
    x = torch.tensor([1., 0., 0.], device=coor.device)[None, :, None]
    theta = compute_angle_between_direction(coor_xy, x)
    sign_mask = compute_clockwise_or_anticlockwise_between_direction(x, coor_xy, -z)
    theta = theta * sign_mask.view((-1, 1))
    return theta, phi


def compute_angle_between_direction(a, b):
    """Compute angle between tow directions(a, b)
    \theta = arccos(\frac{a}{|a|} \frac{b}{|b|})

    Args:
        a (torch.Tensor):  N
        b (torch.Tensor):  N
    Return:
        angle(torch.Tensor): N, rad
    """
    a = F.normalize(a, dim=1)
    b = F.normalize(b, dim=1)

    cos_val = torch.clamp(torch.sum(a*b, dim=1), min=-1+1e-7, max=1-1e-7)
    radian = torch.arccos(cos_val)

    return torch.abs(radian)


def compute_clockwise_or_anticlockwise_between_direction(a, b, ref):
    """Compute clockwise or anticlockwise relationship between two directions (a->b) based on another reference direction.

    Args:
        a (torch.Tensor): Nx3 or Nx3x1
        b (torch.Tensor): Nx3 or Nx3x1
        ref (torch.Tensor): Nx3 or Nx3x1. Point to in-direction.

    Returns:
        sign_mask (torch.Tensor): N. +1 means clockwise while -1 means anticlockwise.
    """
    a, b, ref = a.view((-1, 3)), b.view((-1, 3)), ref.view((-1, 3))
    cross_a_b = torch.cross(a, b, dim=1) # Nx3 
    criterion = torch.sum(cross_a_b * ref, dim=1) < 0 # Nx3
    sign_mask = torch.zeros_like(criterion) + criterion * 1. + ~criterion * (-1.)
    return sign_mask

# ---------------------------- Standardize Method ---------------------------- #
def standardize_rotated_box(rotated_gt, rotated_pred, rbb_angle='deg'):
    """Standardize rotated box to meet the format requirements of downstearm method.

    Args:
        rotated_gt (torch.Tensor): Nx5
        rotated_pred (torch.Tensor): Nx5

    Returns:
        rotated_gt (torch.Tensor): Nx5
        rotated_pred (torch.Tensor): Nx5
    """
    assert rbb_angle in ['rad', 'deg']

    theta_g_, phi_g_, alpha_g, beta_g, angle_g_ = torch.chunk(rotated_gt, chunks=5, dim=1)   # Nx1
    theta_p_, phi_p_, alpha_p, beta_p, angle_p_ = torch.chunk(rotated_pred, chunks=5, dim=1) # Nx1

    if rbb_angle == 'deg':
        angle_g_ = torch.rad2deg(angle_g_)
        angle_p_ = torch.rad2deg(angle_p_)
    
    rotated_gt = torch.concat([theta_g_, phi_g_, alpha_g, beta_g, angle_g_], dim=1)
    rotated_pred = torch.concat([theta_p_, phi_p_, alpha_p, beta_p, angle_p_], dim=1)
    return rotated_gt, rotated_pred