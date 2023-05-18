import torch

def sph_iou_aligned(sph_gt, sph_pred):
    """Spherical criteria for fast and accurate 360 object detection.
    See also https://ojs.aaai.org/index.php/AAAI/article/view/6995
    """
    eps = 1e-8
    sph_gt, sph_pred = standardize_spherical_box(sph_gt, sph_pred)
    sph_gt = angle2radian(sph_gt, mode='convention')
    sph_pred = angle2radian(sph_pred, mode='convention')

    theta_g, phi_g, alpha_g, beta_g = torch.chunk(sph_gt, chunks=4, dim=1)   # Nx1
    theta_p, phi_p, alpha_p, beta_p = torch.chunk(sph_pred, chunks=4, dim=1) # Nx1

    alpha_g_2, beta_g_2 = alpha_g / 2, beta_g / 2
    alpha_p_2, beta_p_2 = alpha_p / 2, beta_p / 2

    theta_min = torch.max(theta_g-alpha_g_2, theta_p-alpha_p_2)
    theta_max = torch.min(theta_g+alpha_g_2, theta_p+alpha_p_2)
    phi_min   = torch.max(phi_g-beta_g_2, phi_p-beta_p_2)
    phi_max   = torch.min(phi_g+beta_g_2, phi_p+beta_p_2)

    area_i = ((theta_max - theta_min).clip(min=0) * (phi_max - phi_min).clip(min=0)).flatten()
    area_u = area_boxes(sph_gt) + area_boxes(sph_pred) - area_i
    iou = area_i / (area_u + eps)

    return iou


def fov_iou_aligned(sph_gt, sph_pred):
    """Field-of-view iou for object detection in 360° images.
    See also https://arxiv.org/pdf/2202.03176.pdf
    """
    eps = 1e-8
    sph_gt, sph_pred = standardize_spherical_box(sph_gt, sph_pred)
    sph_gt = angle2radian(sph_gt, mode='convention')
    sph_pred = angle2radian(sph_pred, mode='convention')

    theta_g, phi_g, alpha_g, beta_g = torch.chunk(sph_gt, chunks=4, dim=1)   # Nx1
    theta_p, phi_p, alpha_p, beta_p = torch.chunk(sph_pred, chunks=4, dim=1) # Nx1

    alpha_g_2, beta_g_2 = alpha_g / 2, beta_g / 2
    alpha_p_2, beta_p_2 = alpha_p / 2, beta_p / 2
    delta_fov = (theta_p - theta_g) * torch.cos((phi_g+phi_p) / 2)

    theta_min = torch.max(-alpha_g_2, delta_fov-alpha_p_2)
    theta_max = torch.min(alpha_g_2,  delta_fov+alpha_p_2)
    phi_min   = torch.max(phi_g-beta_g_2, phi_p-beta_p_2)
    phi_max   = torch.min(phi_g+beta_g_2, phi_p+beta_p_2)

    area_i = ((theta_max - theta_min).clip(min=0) * (phi_max - phi_min).clip(min=0)).flatten()
    area_u = area_boxes(sph_gt) + area_boxes(sph_pred) - area_i
    iou = area_i / (area_u + eps)

    return iou

def area_boxes(boxes):
    return boxes[..., 2] * boxes[..., 3]

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