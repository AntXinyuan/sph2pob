import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------- #
#                             Sph2Pob BoxTransfrom                              #
# ---------------------------------------------------------------------------- #
def sph2pob_legacy(sph_gt, sph_pred, rbb_angle_version='deg', rbb_edge='arc', rbb_angle=None):
    """Transform spherical boxes to planar oriented boxes.
    NOTE: It's a legacy implement of Sph2Pob based on handcraft rules,
    where the calculated inernal angle is not accurate.

    Args:
        sph_gt (torch.Tensor): N x 4(5), deg
        sph_pred (torch.Tensor): N x 4(5), deg
        rbb_angle_version (str, optional): The angle version of output boxes. Defaults to 'deg'.
        rbb_edge (str, optional): Algorithm option. Defaults to 'arc'.
        rbb_angle (str, optional): Useless option for legacy version.

    Returns:
        plannar_gt (torch.tensor): N x 5
        plannar_pred (torch.tensor): N x 5
    """
    sph_gt, sph_pred = standardize_spherical_box(sph_gt, sph_pred)
    position_gt, position_pred = transform_position(sph_gt, sph_pred) # Nx2
    edge_gt, edge_pred = transform_edge(sph_gt, sph_pred, rbb_edge)
    angle_gt, angle_pred = transfrom_anlge(sph_gt, sph_pred) # Nx1
    planar_gt   = torch.concat([position_gt, edge_gt, angle_gt], dim=1)
    planar_pred = torch.concat([position_pred, edge_pred, angle_pred], dim=1)
    planar_gt, planar_pred = standardize_rotated_box(planar_gt, planar_pred, rbb_angle_version)
    return planar_gt, planar_pred #Nx5


# ---------------------------------------------------------------------------- #
#                            Auxiliary ComputeMethod                           #
# ---------------------------------------------------------------------------- #
# --------------------------- Helper ComputeMethod -------------------------- #
def transform_position(sph_gt, sph_pred):
    """Transfrom position of spherical box from anywhere to equator.

    Args:
        sph_gt (torch.Tensor): Spherical box, Nx4
        sph_pred (torch.Tensor): Sperical box, Nx4

    Returns:
        position_gt (torch.Tensor) : Nx2
        position_pred (torch.Tensor): Nx2
    """
    sph_gt   = angle2radian(sph_gt, mode='convention')
    sph_pred = angle2radian(sph_pred, mode='convention')

    theta_g, phi_g, alpha_g, beta_g = torch.chunk(sph_gt, chunks=4, dim=1)   # Nx1
    theta_p, phi_p, alpha_p, beta_p = torch.chunk(sph_pred, chunks=4, dim=1) # Nx1
    
    # Compute phi_g_ & phi_p_ on equator relative to mid-point of boxes.
    phi_i = (phi_g + phi_p) / 2
    phi_g_ = phi_g - phi_i
    phi_p_  = phi_p - phi_i

    # Compute length of arc on original position
    # See also in https://www.jianshu.com/p/d4adcaf1f459
    delta_phi   = torch.abs(phi_g - phi_p)
    delta_theta = torch.abs(theta_g - theta_p)
    R = 1
    L = 2*R*torch.arcsin(torch.sqrt(     \
        torch.sin(delta_phi / 2)**2 +    \
        torch.cos(phi_g) * torch.cos(phi_p) * torch.sin(delta_theta / 2)**2))
    
    # Resolve delta_phi_ & delta_theta_ on equator based on the equality of arc-length
    delta_phi_   = delta_phi
    delta_theta_ = torch.abs(2*torch.arcsin(torch.sqrt(       \
        (torch.sin(L / (2*R))**2 - torch.sin(delta_phi_ / 2)**2) /   \
        (torch.cos(phi_g_)* torch.cos(phi_p_)))))
    
    # Compute theta_g_ & theta_p_ on equator relative to mid-point of boxes.
    theta_g_ = torch.zeros_like(theta_g)
    sign_mask_theta_p_ = torch.zeros_like(theta_p) + (theta_p > theta_g) * 1.0 + (theta_p <= theta_g) * (-1.0)
    theta_p_ = delta_theta_ * sign_mask_theta_p_
    
    position_gt = torch.concat([theta_g_, phi_g_], dim=1)   # Nx2
    position_pred = torch.concat([theta_p_, phi_p_], dim=1) # Nx2

    return position_gt, position_pred

def transform_edge(sph_gt, sph_pred, rbb_edge):
    sph_gt   = angle2radian(sph_gt, mode='convention')
    sph_pred = angle2radian(sph_pred, mode='convention')

    theta_g, phi_g, alpha_g, beta_g = torch.chunk(sph_gt, chunks=4, dim=1)   # Nx1
    theta_p, phi_p, alpha_p, beta_p = torch.chunk(sph_pred, chunks=4, dim=1) # Nx1

    alpha_g_ = compute_edge_length(alpha_g, rbb_edge)
    beta_g_  = compute_edge_length(beta_g, rbb_edge)
    alpha_p_ = compute_edge_length(alpha_p, rbb_edge)
    beta_p_  = compute_edge_length(beta_p, rbb_edge)

    edge_gt   = torch.concat([alpha_g_, beta_g_], dim=1)   # Nx2
    edge_pred = torch.concat([alpha_p_, beta_p_], dim=1)   # Nx2

    return edge_gt, edge_pred

def transfrom_anlge(sph_gt, sph_pred):
    """Transfrom anlge of spherical box based on relative position angle between gt&pred boxes.

    Args:
        sph_gt (torch.Tensor): Spherical box, Nx4
        sph_pred (torch.Tensor): Spherical box, Nx4

    Returns:
        angle_gt (torch.Tensor) : Nx1
        angle_pred (torch.Tensor): Nx1
    """
    sph_gt   = angle2radian(sph_gt, mode='math')
    sph_pred = angle2radian(sph_pred, mode='math')

    theta_g, phi_g = sph_gt[:, 0], sph_gt[:, 1]     # N
    theta_p, phi_p = sph_pred[:, 0], sph_pred[:, 1] # N

    theta_mid = (theta_g + theta_p) / 2

    theta_gr, phi_gr = theta_mid, phi_g
    theta_pr, phi_pr = theta_mid, phi_p

    angle_gt = _transfrom_angle_aux(theta_g, phi_g, theta_gr, phi_gr)   # Nx1
    angle_pred = _transfrom_angle_aux(theta_p, phi_p, theta_pr, phi_pr) # Nx1
    return angle_gt, angle_pred

def _transfrom_angle_aux(theta_box, phi_box, theta_ref, phi_ref):
    dir_box = compute_tangential_direction_along_longitude(theta_box, phi_box)
    dir_ref = compute_tangential_direction_along_longitude(theta_ref, phi_ref)
    angle = compute_angle_between_direction(dir_box, dir_ref)
    sign_mask = (theta_box >= theta_ref) & (phi_box < (torch.pi / 2)) | (theta_box <= theta_ref) & (phi_box > (torch.pi / 2))
    angle[~sign_mask] *= -1
    return angle.unsqueeze_(1)

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

# ---------------------------- Standardize Method ---------------------------- #
def angle2radian(angle_sph_box, mode='convention'):
    """Tranfrom angle to radian based on specific mode.

    Args:
        angle_sph_box (_type_): box with angle-repretation.
        mode (str, optional): mode. Defaults to 'convention'.
            'convention': (90, -90), (-180, 180)
            'math': (0, 180), (0, 360)

    Returns:
        radian_sph_box: box with radian-repretation.
    """
    assert mode in ['math', 'convention']
    radian_sph_box = torch.deg2rad(angle_sph_box)
    if mode == 'convention': #(-180, 180), (90, -90)
        radian_sph_box[:, 0] = radian_sph_box[:, 0] - torch.pi
        radian_sph_box[:, 1] = torch.pi / 2 - radian_sph_box[:, 1]
    return radian_sph_box

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
