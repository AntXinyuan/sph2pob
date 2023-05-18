import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------- #
#                             Sph2Pob BoxTransfrom                             #
# ---------------------------------------------------------------------------- #
def sph2pob_standard(sph_gt, sph_pred, rbb_angle_version='deg', rbb_edge='arc', rbb_angle='equator'):
    """Transform spherical boxes to planar oriented boxes.
    NOTE: It's a standard implement of Sph2Pob.

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
    sph_gt   = torch.deg2rad(sph_gt)
    sph_pred = torch.deg2rad(sph_pred)

    theta_g, phi_g, alpha_g, beta_g = torch.chunk(sph_gt[:, :4], chunks=4, dim=1)   # Nx1
    theta_p, phi_p, alpha_p, beta_p = torch.chunk(sph_pred[:, :4], chunks=4, dim=1) # Nx1
    theta_r, phi_r = (theta_g+theta_p) / 2, (phi_g + phi_p) / 2 

    from torch import cos, sin
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

    R = compute_rotate_matrix_auto(coor_g, coor_p, theta_r, phi_r)

    if sph_gt.size(1) == 5:
        gamma = sph_gt[:, -1].view((-1, 1))
        R_gamma = compute_gamma_matrix(theta_g, phi_g, -gamma)
        dir_g = torch.bmm(R_gamma, dir_g)

        gamma = sph_pred[:, -1].view((-1, 1))
        R_gamma = compute_gamma_matrix(theta_p, phi_p, -gamma)
        dir_p = torch.bmm(R_gamma, dir_p)

        del R_gamma
        #torch.cuda.empty_cache()
    
    coor_g = torch.bmm(R, coor_g) # Nx3x1
    coor_p = torch.bmm(R, coor_p) # Nx3x1
    dir_g  = torch.bmm(R, dir_g)  # Nx3x1
    dir_p  = torch.bmm(R, dir_p)  # Nx3x1

    angle_g_ = compute_internal_angle(dir_g, rbb_angle) # Nx1
    angle_p_ = compute_internal_angle(dir_p, rbb_angle) # Nx1

    theta_g_, phi_g_ = compute_spherical_coordinate(coor_g) # Nx1
    theta_p_, phi_p_ = compute_spherical_coordinate(coor_p) # Nx1

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
def compute_internal_angle(dir_, rbb_angle='equator'):
    """Compute signed angle between given direction dir_ and reference direction z-axis on prejected yOz-surface.  

    Args:
        dir_ (torch.Tensor): Nx3x1

    Returns:
        angle_ (torch.Tensor): Nx1
    """
    assert rbb_angle in ['equator', 'project']

    if rbb_angle == 'project':
        dir_[:, 0, :] = 0

    dir_z_ = torch.tensor([0., 0., 1.], device=dir_.device)[None, :, None]
    angle_ = compute_angle_between_direction(dir_, dir_z_) # Nx1

    dir_ref = torch.tensor([1., 0., 0.], device=dir_.device)[None, :, None]
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
    return torch.deg2rad(theta), torch.deg2rad(phi)


def compute_angle_between_direction(a, b):
    """Compute angle between tow directions(a, b)
    \theta = arccos(\frac{a}{|a|} \frac{b}{|b|})

    Args:
        a (torch.Tensor):  N
        b (torch.Tensor):  N
    Return:
        angle(torch.Tensor): N
    """
    a = F.normalize(a, dim=1)
    b = F.normalize(b, dim=1)
    cos_val = torch.clamp(torch.sum(a*b, dim=1), min=-1+1e-7, max=1-1e-7)
    radian = torch.arccos(cos_val)
    angle = radian / torch.pi * 180
    return torch.abs(angle)


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


# -------------------------- Geometry TranformMethod ------------------------- #
def compute_rotate_matrix(theta, phi):
    """Compute rotate matrix to rotate spherical coordinate, s.t. 
    given point(theta, phi) will move to the front of sphere, i.e. (1, 0, 0).

    Args:
        theta (torch,Tensor): Nx1
        phi (torch,Tensor): Nx1

    Returns:
        R (torch,Tensor): Nx3x3
    """
    from torch import cos, sin
    sin_theta, cos_theta = sin(theta), cos(theta)
    sin_phi, cos_phi = sin(phi), cos(phi)
    zero = torch.zeros_like(theta)
    v_look  = torch.stack([sin_phi*cos_theta, sin_phi*sin_theta, cos_phi], dim=1) #Nx3x1
    v_right = torch.stack([sin_theta, -cos_theta, zero], dim=1) #Nx3x1
    v_down    = torch.stack([cos_phi*cos_theta, cos_phi*sin_theta, -sin_phi], dim=1) #Nx3x1

    R = torch.concat([v_look, v_down, v_right], dim=-1)
    # Compute inverse tranform matrix R^{-1}, and note R^{-1} is equtal to R^{T}.
    R = torch.transpose(R, dim0=-1 ,dim1=-2)
    return R


def compute_rotate_matrix_better(coor_g, coor_p):
    """Compute rotate matrix to rotate spherical coordinate, s.t. 
    1. the mid-point of coor_g & coor_p will move to the front of sphere, i.e. (1, 0, 0).
    2. coor_g & coor_p will move to the equtor, i.e. (x, -y, 0) & (x, +y, 0). 

    Args:
        coor_g (torch,Tensor): Nx3x1
        coor_p (torch,Tensor): Nx3x1

    Returns:
        R (torch,Tensor): Nx3x3
    """    
    v_look = F.normalize(coor_g + coor_p, dim=1).squeeze(-1)  # Nx3
    v_right = F.normalize(coor_p - coor_g, dim=1).squeeze(-1) # Nx3
    v_up = torch.cross(v_look, v_right, dim=1) # Nx3
    R = torch.stack([v_look, v_right, v_up], dim=-1) #Nx3x3

    # Compute inverse tranform matrix R^{-1}, and note R^{-1} is equtal to R^{T}.
    R = torch.transpose(R, dim0=-1 ,dim1=-2)
    return R


def compute_rotate_matrix_auto(coor_g, coor_p, theta=None, phi=None, eps=1e-8):
    if theta is None or phi is None:
        v_look = F.normalize(coor_g + coor_p, dim=1) #Nx3x1
        theta, phi = compute_spherical_coordinate(v_look) # Nx1
        theta = (theta + 2*torch.pi) % (2*torch.pi)

    normal_mask = torch.abs(coor_g - coor_p).squeeze(dim=-1).sum(dim=1) > eps
    R = torch.empty((coor_g.size(0), 3, 3), device=coor_g.device)
    R[normal_mask]  = compute_rotate_matrix_better(coor_g[normal_mask], coor_p[normal_mask])
    R[~normal_mask] = compute_rotate_matrix(theta[~normal_mask], phi[~normal_mask])
    #R = compute_rotate_matrix_better(coor_g, coor_p)
    return R


def compute_gamma_matrix(theta, phi, gamma):
    T = compute_rotate_matrix(theta, phi) #Nx3x3
    sin_gamma, cos_gamma = torch.sin(gamma), torch.cos(gamma)
    zero, one = torch.zeros_like(gamma), torch.ones_like(gamma)
    Rx = torch.stack([one, zero, zero], dim=1)             #Nx3x1
    Ry = torch.stack([zero, cos_gamma, sin_gamma], dim=1)  #Nx3x1
    Rz = torch.stack([zero, -sin_gamma, cos_gamma], dim=1) #Nx3x1
    R = torch.concat([Rx, Ry, Rz], dim=-1) #Nx3x3

    # R' = T^ @ R @ T
    del Rx, Ry, Rz, sin_gamma, cos_gamma, zero, one
    R = torch.bmm(R, T)
    T = torch.transpose(T, dim0=-1 ,dim1=-2)
    R = torch.bmm(T, R)
    return R


# ---------------------------- Standardize Method ---------------------------- #
def standardize_spherical_box(sph_gt, sph_pred):
    """Standardize sperical box to overcome cross-boundary problem.
    Specificly, the method shifts theta from (-180, 180) to (-90, 90).

    Args:
        sph_gt (torch.Tensor): Nx4
        sph_pred (torch.Tensor): Nx4

    Returns:
        sph_gt (torch.Tensor): Nx4
        sph_pred (torch.Tensor): Nx4
    """
    #sph_gt = sph_gt.clone()
    #sph_pred = sph_pred.clone()

    theta_g, theta_p = sph_gt[:, 0], sph_pred[:, 0] #N

    move_mask = torch.abs(theta_g - theta_p) > 180 # N
    sph_gt[move_mask, 0] = (sph_gt[move_mask, 0] + 180) % 360 
    sph_pred[move_mask, 0] = (sph_pred[move_mask, 0] + 180) % 360
    
    return sph_gt, sph_pred


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

    if rbb_angle == 'rad':
        angle_g_ = torch.deg2rad(angle_g_)
        angle_p_ = torch.deg2rad(angle_p_)
    
    rotated_gt = torch.concat([theta_g_, phi_g_, alpha_g, beta_g, angle_g_], dim=1)
    rotated_pred = torch.concat([theta_p_, phi_p_, alpha_p, beta_p, angle_p_], dim=1)
    return rotated_gt, rotated_pred