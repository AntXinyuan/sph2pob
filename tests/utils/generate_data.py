import torch
import numpy as np

def _ensure_tuple(val):
    if isinstance(val, int):
        val = (val, val)
    assert isinstance(val, tuple)
    return val

def generate_boxes(num, theta_range=(0, 360), phi_range=(0, 180), alpha_range=(1, 180), beta_range=(1, 180), gamma_range=(-90, 90), version='deg', dtype='int', box='bfov'):
    assert version in ['deg', 'rad']
    assert dtype in ['int', 'float']
    assert box in ['bfov', 'rbfov']
    
    if dtype == 'int':
        if theta_range[0] == theta_range[1]:
            theta = torch.empty((num,)).fill_(theta_range[0]).float()
        else:
            theta = torch.randint(*theta_range, (num,)).float()   
        if phi_range[0] == phi_range[1]:
            phi = torch.empty((num,)).fill_(phi_range[0]).float()
        else:
            phi   = torch.randint(*phi_range, (num,)).float()
        alpha = torch.randint(*alpha_range, (num,)).float()
        beta  = torch.randint(*beta_range, (num,)).float()
        if gamma_range[0] == gamma_range[1]:
            gamma = torch.empty((num,)).fill_(gamma_range[0]).float()
        else:
            gamma   = torch.randint(*gamma_range, (num,)).float()
    else:
        rand = torch.rand((num, 5))
        theta = rand[:, 0] * (theta_range[1] - theta_range[0]) + theta_range[0]
        phi   = rand[:, 1] * (phi_range[1] - phi_range[0]) + phi_range[0]
        alpha = rand[:, 2] * (alpha_range[1] - alpha_range[0]) + alpha_range[0]
        beta  = rand[:, 3] * (beta_range[1] - beta_range[0]) + beta_range[0]
        gamma = rand[:, 4] * (gamma_range[1] - gamma_range[0]) + gamma_range[0]
    
    if box == 'bfov':
        boxes = torch.stack([theta, phi, alpha, beta], dim=1)
    else:
        boxes = torch.stack([theta, phi, alpha, beta, gamma], dim=1)
    return torch.deg2rad(boxes) if version == 'rad' else boxes


def generate_ranges(granularity=30, flaten=False, mode='full', select_idx=0):
    assert mode in ['full', 'diag', 'row', 'col']
    max_theta, max_phi = 360, 180
    granularity = _ensure_tuple(granularity)
    x = torch.arange(0, max_theta, granularity[0])
    y = torch.arange(0, max_phi, granularity[1])
    xx, yy = torch.meshgrid(x, y, indexing='xy')
    N, M = xx.shape
    x1x2y1y2 = torch.stack([xx, xx+granularity[0], yy, yy+granularity[0]]).permute((1, 2, 0)).reshape((N, M, 2, 2))
    if mode == 'diag':
        x1x2y1y2 = torch.stack([x1x2y1y2[i, i] for i in range(min(N, M))])
    elif mode == 'row':
        x1x2y1y2 = x1x2y1y2[select_idx, :, :, :].view((-1, M, 2, 2))
    elif mode == 'col':
        x1x2y1y2 = x1x2y1y2[:, select_idx, :, :].view((N, -1, 2, 2))
    elif flaten:
        x1x2y1y2 = x1x2y1y2.reshape((-1, 2, 2))
    return x1x2y1y2