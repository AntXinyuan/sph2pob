import numpy as np
import torch

def theta_phi_to_xyz(theta, phi):
    xyz = np.concatenate((
        np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)
    ), axis=1)
    return xyz

def roll_T(n, xyz, gamma=0):
    # n:(N, 3) | xyz:(N, 3) | gamma:(N, 1)
    gamma = gamma.flatten() #/ 180 * np.pi
    cos_gamma, sin_gamma = np.cos(gamma), np.sin(gamma)
    n11 = (n[...,0] ** 2) * (1 - cos_gamma) + cos_gamma
    n12 = n[...,0] * n[...,1] * (1 - cos_gamma) - n[...,2] * sin_gamma
    n13 = n[...,0] * n[...,2] * (1 - cos_gamma) + n[...,1] * sin_gamma

    n21 = n[...,0] * n[...,1] * (1 - cos_gamma) + n[...,2] * sin_gamma
    n22 = (n[...,1] ** 2) * (1 - cos_gamma) + cos_gamma
    n23 = n[...,1] * n[...,2] * (1 -cos_gamma) - n[...,0] * sin_gamma

    n31 = n[...,0] * n[...,2] * (1 - cos_gamma) - n[...,1] * sin_gamma
    n32 = n[...,1] * n[...,2] * (1 - cos_gamma) + n[...,0] * sin_gamma
    n33 = (n[...,2] ** 2) * (1 - cos_gamma) + cos_gamma

    x, y, z = xyz[...,0], xyz[...,1], xyz[...,2]

    xx = n11 * x + n12 * y + n13 * z
    yy = n21 * x + n22 * y + n23 * z
    zz = n31 * x + n32 * y + n33 * z

    return np.stack((xx,yy, zz), axis=1)



def roArrayVector(theta, phi, v, ang):
    c_xyz = theta_phi_to_xyz(theta, phi)
    p_xyz = v
    pp_xyz = roll_T(c_xyz, p_xyz, ang)
    return pp_xyz

class Sph:
    '''Unbiased IoU Computation for Spherical Rectangles'''

    def __init__(self):
        self.visited, self.trace, self.pot = [], [], []

    def area(self, fov_x, fov_y):
        '''Area Computation'''
        return 4 * np.arccos(-np.sin(fov_x / 2) * np.sin(fov_y / 2)) - 2 * np.pi

    def getNormal(self, bbox):
        '''Normal Vectors Computation'''
        theta, phi, fov_x_half, fov_y_half, angle = bbox[:, [
            0]], bbox[:, [1]], bbox[:, [2]] / 2, bbox[:, [3]] / 2, bbox[:,[4]]
        V_lookat = np.concatenate((
            np.sin(phi) * np.cos(theta), np.sin(phi) *
            np.sin(theta), np.cos(phi)
        ), axis=1)
        V_right = np.concatenate(
            (-np.sin(theta), np.cos(theta), np.zeros(theta.shape)), axis=1)
        V_up = np.concatenate((
            -np.cos(phi) * np.cos(theta), -np.cos(phi) *
            np.sin(theta), np.sin(phi)
        ), axis=1)
        N_left = -np.cos(fov_x_half) * V_right + np.sin(fov_x_half) * V_lookat
        N_right = np.cos(fov_x_half) * V_right + np.sin(fov_x_half) * V_lookat
        N_up = -np.cos(fov_y_half) * V_up + np.sin(fov_y_half) * V_lookat
        N_down = np.cos(fov_y_half) * V_up + np.sin(fov_y_half) * V_lookat

        N_left = roArrayVector(theta, phi, N_left, angle)
        N_right = roArrayVector(theta, phi, N_right, angle)
        N_up = roArrayVector(theta, phi, N_up, angle)
        N_down = roArrayVector(theta, phi, N_down, angle)

        V = np.array([
            np.cross(N_left, N_up), np.cross(N_down, N_left),
            np.cross(N_up, N_right), np.cross(N_right, N_down)
        ])
        norm = np.linalg.norm(V, axis=2)[
            :, :, np.newaxis].repeat(V.shape[2], axis=2)
        V = np.true_divide(V, norm)
        E = np.array([
            [N_left, N_up], [N_down, N_left], [
                N_up, N_right], [N_right, N_down]
        ])
        return np.array([N_left, N_right, N_up, N_down]), V, E

    def interArea(self, orders, E):
        '''Intersection Area Computation'''
        angles = -np.matmul(E[:, 0, :][:, np.newaxis, :],
                            E[:, 1, :][:, :, np.newaxis])
        angles = np.clip(angles, -1, 1)
        whole_inter = np.arccos(angles)
        inter_res = np.zeros(orders.shape[0])
        loop = 0
        idx = np.where(orders != 0)[0]
        iters = orders[idx]
        for i, j in enumerate(iters):
            inter_res[idx[i]] = np.sum(
                whole_inter[loop:loop+j], axis=0) - (j - 2) * np.pi
            loop += j
        return inter_res

    def remove_outer_points(self, dets, gt):
        '''Remove points outside the two spherical rectangles'''
        N_dets, V_dets, E_dets = self.getNormal(dets)
        N_gt, V_gt, E_gt = self.getNormal(gt)
        N_res = np.vstack((N_dets, N_gt))
        V_res = np.vstack((V_dets, V_gt))
        E_res = np.vstack((E_dets, E_gt))

        N_dets_expand = N_dets.repeat(N_gt.shape[0], axis=0)
        N_gt_expand = np.tile(N_gt, (N_dets.shape[0], 1, 1))

        tmp1 = np.cross(N_dets_expand, N_gt_expand)
        mul1 = np.true_divide(
            tmp1, np.linalg.norm(tmp1, axis=2)[:, :, np.newaxis].repeat(tmp1.shape[2], axis=2) + 1e-10)

        tmp2 = np.cross(N_gt_expand, N_dets_expand)
        mul2 = np.true_divide(
            tmp2, np.linalg.norm(tmp2, axis=2)[:, :, np.newaxis].repeat(tmp2.shape[2], axis=2) + 1e-10)

        V_res = np.vstack((V_res, mul1))
        V_res = np.vstack((V_res, mul2))

        dimE = (E_res.shape[0] * 2, E_res.shape[1],
                E_res.shape[2], E_res.shape[3])
        E_res = np.vstack(
            (E_res, np.hstack((N_dets_expand, N_gt_expand)).reshape(dimE)))
        E_res = np.vstack(
            (E_res, np.hstack((N_gt_expand, N_dets_expand)).reshape(dimE)))

        res = np.round(np.matmul(V_res.transpose(
            (1, 0, 2)), N_res.transpose((1, 2, 0))), 8)
        value = np.all(res >= 0, axis=2)
        return value, V_res, E_res

    def computeInter(self, dets, gt):
        '''
        The whole Intersection Area Computation Process (3 Steps):
        Step 1. Compute normal vectors and point vectors of each plane for eight boundaries of two spherical rectangles;
        Step 2. Remove unnecessary points by two Substeps:
           - Substep 1: Remove points outside the two spherical rectangles;
           - Substep 2: Remove redundant Points. (This step is not required for most cases that do not have redundant points.)
        Step 3. Compute all left angles and the final intersection area.
        '''
        value, V_res, E_res = self.remove_outer_points(dets, gt)

        ind0 = np.where(value)[0]
        ind1 = np.where(value)[1]

        if ind0.shape[0] == 0:
            return np.zeros((dets.shape[0]))

        E_final = E_res[ind1, :, ind0, :]
        orders = np.bincount(ind0)

        minus = dets.shape[0] - orders.shape[0]
        if minus > 0:
            orders = np.pad(orders, (0, minus), mode='constant')

        inter = self.interArea(orders, E_final)
        return inter

    def sphIoU(self, dets, gt, is_aligned=False, eps=1e-8):
        '''Unbiased Spherical IoU Computation'''
        dets, gt = torch.deg2rad(dets), torch.deg2rad(gt)
        dets, gt = dets.cpu().numpy(), gt.cpu().numpy()
        d_size, g_size = dets.shape[0], gt.shape[0]
        if is_aligned:
            res = np.concatenate([dets, gt], axis=1)
        else:
            res = np.hstack((dets.repeat(g_size, axis=0), np.tile(
                gt, (d_size, 1)))).reshape(d_size * g_size, -1)
        area_A = self.area(res[:, 2], res[:, 3])
        area_B = self.area(res[:, 7], res[:, 8])
        inter = self.computeInter(res[:, :5], res[:, 5:])
        final = inter / (area_A + area_B - inter + eps)
        final = final if is_aligned else final.reshape(d_size, g_size)
        final = torch.from_numpy(final).float()
        return final

def transFormat(gt):
    '''
    Change the format and range of the RBFoV Representations.
    Input:
    - gt: the last dimension: [center_x, center_y, fov_x, fov_y, angle]
          center_x : [-180, 180]
          center_y : [90, -90]
          fov_x    : [0, 180]
          fov_y    : [0, 180]
          angle    : [90, -90]
          All parameters are angles.
    Output:
    - ann: the last dimension: [center_x', center_y', fov_x', fov_y', angle]
           center_x' : [0, 2 * pi]
           center_y' : [0, pi]
           fov_x'    : [0, pi]
           fov_y'    : [0, pi]
           angle     : [90, -90]
           All parameters are radians.
    '''
    import copy
    ann = copy.copy(gt)
    ann[..., 2] = ann[..., 2] / 180 * np.pi
    ann[..., 3] = ann[..., 3] / 180 * np.pi
    ann[..., 0] = ann[..., 0] / 180 *np.pi+ np.pi
    ann[..., 1] = np.pi / 2 - ann[..., 1] / 180 * np.pi
    return ann
