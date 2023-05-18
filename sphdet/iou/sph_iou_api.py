import torch
from mmcv.ops import bbox_overlaps, box_iou_rotated, diff_iou_rotated_2d

from sphdet.bbox.box_formator import (Sph2PlanarBoxTransform,)

from .approximate_ious import fov_iou_aligned, sph_iou_aligned
from .sph2pob_efficient import sph2pob_efficient
from .sph2pob_legacy import sph2pob_legacy
from .sph2pob_standard import sph2pob_standard
from .unbiased_iou_bfov import Sph as BFOV
from .unbiased_iou_rbfov import Sph as RBFOV

#from .diff_iou_rotated import diff_iou_rotated_2d # Fix some bugs in mmcv.ops.diff_iou_rotated


def _single_box_iou_rotated_cv2(box1, box2):
    import cv2
    area1 = box1[2] * box1[3]
    area2 = box2[2] * box2[3]
    r1 = ((box1[0], box1[1]), (box1[2], box1[3]), box1[4])
    r2 = ((box2[0], box2[1]), (box2[2], box2[3]), box2[4])
    int_pts = cv2.rotatedRectangleIntersection(r1, r2)[1]
    if int_pts is not None:
        order_pts = cv2.convexHull(int_pts, returnPoints=True)
        int_area = cv2.contourArea(order_pts)
        ious = int_area * 1.0 / (area1 + area2 - int_area)
    else:
        ious=0
    return ious

def _box_iou_rotated_cv2(bboxes1, bboxes2):
    bboxes1[: , 4] = torch.rad2deg(bboxes1[: , 4])
    bboxes2[: , 4] = torch.rad2deg(bboxes2[: , 4])
    device = bboxes1.device
    bboxes1 = bboxes1.tolist()
    bboxes2 = bboxes2.tolist()
    overlaps = []
    import math
    for box1, box2 in zip(bboxes1, bboxes2):
        box1[0] = box1[0] + math.pi
        box1[1] = math.pi/2 - box1[1]
        box2[0] = box2[0] + math.pi
        box2[1] = math.pi/2 - box2[1]
        overlaps.append(_single_box_iou_rotated_cv2(box1, box2))
    return torch.tensor(overlaps, device=device)

def _sph2pob_iou_auxiliary(bboxes1, bboxes2, transform, mode, is_aligned, calculator, rbb_edge, rbb_angle):
    assert mode in ['iou', 'iof']
    assert calculator in ['common', 'diff']
    assert rbb_edge in ['arc', 'chord', 'tangent']

    rows = bboxes1.size(0)
    cols = bboxes2.size(0)

    if rows * cols == 0:
        return bboxes1.new(rows, 1) if is_aligned else bboxes1.new(rows, cols)

    if not is_aligned:
        bboxes1 = bboxes1.repeat_interleave(cols, dim=0)
        bboxes2 = bboxes2.repeat((rows, 1))
    else:
        bboxes1 = bboxes1.clone()
        bboxes2 = bboxes2.clone()
    assert bboxes1.size(0) == bboxes2.size(0)
    
    bboxes1, bboxes2 = jiter_spherical_bboxes(bboxes1, bboxes2)

    bboxes1, bboxes2 = transform(bboxes1, bboxes2, rbb_angle_version='rad', rbb_edge=rbb_edge, rbb_angle=rbb_angle)
    # Serial processing to save some memory.
    # mid = bboxes1.size(0) // 2
    # bboxes1a, bboxes2a = transform(bboxes1[:mid], bboxes2[:mid], rbb_angle_version='rad', rbb_edge=rbb_edge, rbb_angle=rbb_angle)
    # bboxes1b, bboxes2b = transform(bboxes1[mid:], bboxes2[mid:], rbb_angle_version='rad', rbb_edge=rbb_edge, rbb_angle=rbb_angle)
    # bboxes1, bboxes2 = torch.concat([bboxes1a, bboxes1b]), torch.concat([bboxes2a, bboxes2b])

    bboxes1, bboxes2 = jiter_rotated_bboxes(bboxes1, bboxes2)

    if calculator == 'common':
        overlaps = box_iou_rotated(bboxes1, bboxes2, mode, aligned=True, clockwise=True)
    elif calculator == 'diff':
        overlaps = diff_iou_rotated_2d(bboxes1.unsqueeze(0).cuda(), bboxes2.unsqueeze(0).cuda()).squeeze(0).to(bboxes1)
    else:
        raise NotImplemented('Not supported calculator!')

    overlaps = overlaps if is_aligned else overlaps.view((rows, cols))
    return overlaps.clamp(min=0, max=1)

# ---------------------------------------------------------------------------- #
#                                  Sph2Pob-IoU                                 #
# ---------------------------------------------------------------------------- #
def sph2pob_legacy_iou(bboxes1, bboxes2, mode='iou', is_aligned=False, calculator='common', rbb_edge='arc'):
    return _sph2pob_iou_auxiliary(bboxes1, bboxes2, sph2pob_legacy, mode, is_aligned, calculator, rbb_edge, None)

def sph2pob_standard_iou(bboxes1, bboxes2, mode='iou', is_aligned=False, calculator='common', rbb_edge='arc', rbb_angle='equator'):
    return _sph2pob_iou_auxiliary(bboxes1, bboxes2, sph2pob_standard, mode, is_aligned, calculator, rbb_edge, rbb_angle)

def sph2pob_efficient_iou(bboxes1, bboxes2, mode='iou', is_aligned=False, calculator='common', rbb_edge='arc', rbb_angle='equator'):
    return _sph2pob_iou_auxiliary(bboxes1, bboxes2, sph2pob_efficient, mode, is_aligned, calculator, rbb_edge, rbb_angle)

# ---------------------------------------------------------------------------- #
#                                 Unbiased-IoU                                 #
# ---------------------------------------------------------------------------- #
def unbiased_iou(bboxes1, bboxes2, mode='iou', is_aligned=False):
    assert mode in ['iou']

    rows = bboxes1.size(0)
    cols = bboxes2.size(0)

    if rows * cols == 0:
        return bboxes1.new(rows, 1) if is_aligned else bboxes1.new(rows, cols)

    if not is_aligned:
        bboxes1 = bboxes1.repeat_interleave(cols, dim=0)
        bboxes2 = bboxes2.repeat((rows, 1))
    else:
        bboxes1 = bboxes1.clone()
        bboxes2 = bboxes2.clone()
    assert bboxes1.size(0) == bboxes2.size(0)

    calculator = BFOV() if bboxes1.size(1) == 4 else RBFOV()
    bboxes1, bboxes2 = jiter_spherical_bboxes(bboxes1, bboxes2)
    overlaps = calculator.sphIoU(bboxes1, bboxes2, is_aligned=True)

    overlaps = overlaps if is_aligned else overlaps.view((rows, cols))
    return overlaps.clamp(min=0, max=1)

# ---------------------------------------------------------------------------- #
#                                    Sph-IoU                                   #
# ---------------------------------------------------------------------------- #
def sph_iou(bboxes1, bboxes2, mode='iou', is_aligned=False, calculator='diff'):
    assert mode in ['iou']

    rows = bboxes1.size(0)
    cols = bboxes2.size(0)

    if rows * cols == 0:
        return bboxes1.new(rows, 1) if is_aligned else bboxes1.new(rows, cols)

    if not is_aligned:
        bboxes1 = bboxes1.repeat_interleave(cols, dim=0)
        bboxes2 = bboxes2.repeat((rows, 1))
    else:
        bboxes1 = bboxes1.clone()
        bboxes2 = bboxes2.clone()
    assert bboxes1.size(0) == bboxes2.size(0)

    bboxes1, bboxes2 = jiter_spherical_bboxes(bboxes1, bboxes2)
    overlaps = sph_iou_aligned(bboxes1, bboxes2)
    
    overlaps = overlaps if is_aligned else overlaps.view((rows, cols))
    return overlaps.clamp(min=0, max=1)

# ---------------------------------------------------------------------------- #
#                                    Fov-IoU                                   #
# ---------------------------------------------------------------------------- #
def fov_iou(bboxes1, bboxes2, mode='iou', is_aligned=False, calculator='diff'):
    assert mode in ['iou']

    rows = bboxes1.size(0)
    cols = bboxes2.size(0)

    if rows * cols == 0:
        return bboxes1.new(rows, 1) if is_aligned else bboxes1.new(rows, cols)

    if not is_aligned:
        bboxes1 = bboxes1.repeat_interleave(cols, dim=0)
        bboxes2 = bboxes2.repeat((rows, 1))
    else:
        bboxes1 = bboxes1.clone()
        bboxes2 = bboxes2.clone()
    assert bboxes1.size(0) == bboxes2.size(0)

    bboxes1, bboxes2 = jiter_spherical_bboxes(bboxes1, bboxes2)
    overlaps = fov_iou_aligned(bboxes1, bboxes2)

    overlaps = overlaps if is_aligned else overlaps.view((rows, cols))
    return overlaps.clamp(min=0, max=1)

# ---------------------------------------------------------------------------- #
#                                   Naive-IoU                                  #
# ---------------------------------------------------------------------------- #
def naive_iou(bboxes1, bboxes2, mode='iou', is_aligned=False, box_formator='sph2pix'):
    assert mode in ['iou']

    rows = bboxes1.size(0)
    cols = bboxes2.size(0)

    if rows * cols == 0:
        return bboxes1.new(rows, 1) if is_aligned else bboxes1.new(rows, cols)

    box_version = bboxes1.size(1)
    box_formator = Sph2PlanarBoxTransform(box_formator, box_version)
    iou_calculator = bbox_overlaps if box_version == 4 else box_iou_rotated

    # The absolute numerical value of img_size does not affect subsequent calculations
    bboxes1 = box_formator(bboxes1)
    bboxes2 = box_formator(bboxes2)
    overlaps = iou_calculator(bboxes1, bboxes2, mode, is_aligned)
    return overlaps


def jiter_rotated_bboxes(bboxes1, bboxes2):
    eps = 1e-4 * 1.2345678
    Eps1 = torch.tensor([eps, eps, 2*eps, 2*eps, eps], device=bboxes1.device).unsqueeze_(0)
    Eps2 = torch.tensor([2*eps, 2*eps, eps, eps, 5*eps], device=bboxes1.device).unsqueeze_(0)
    similar_mask = (torch.abs(bboxes1[:, [0,2,3,4]] - bboxes2[:, [0,2,3,4]]) < eps).any(dim=1)
    bboxes1[similar_mask] += Eps1
    bboxes2[similar_mask] += Eps2

    eps = 1e-3 * 1.2345678
    angle_mask = torch.abs(bboxes1[:, 4] - bboxes2[:, 4]) < eps
    bboxes1[angle_mask, 4] += eps
    bboxes2[angle_mask, 4] += 2*eps


    pi = torch.pi
    bboxes1[:, 2:4].clamp_(min=2*eps/10)
    bboxes2[:, 2:4].clamp_(min=eps/10)
    bboxes1[:, 4].clamp_(min=-2*pi+2*eps, max=2*pi-eps)
    bboxes2[:, 4].clamp_(min=-2*pi+eps, max=2*pi-2*eps)

    return bboxes1, bboxes2

def jiter_spherical_bboxes(bboxes1, bboxes2):
    eps = 1e-4 * 1.2345678
    similar_mask = (torch.abs(bboxes1 - bboxes2) < eps).any(dim=1)

    bboxes1[similar_mask] = bboxes1[similar_mask] - 2* eps
    bboxes2[similar_mask] = bboxes2[similar_mask] + eps

    pi = 180
    torch.clamp_(bboxes1[:, 0], 2*eps, 2*pi-eps)
    torch.clamp_(bboxes1[:, 1:4], 2*eps, pi-eps)
    torch.clamp_(bboxes2[:, 0], eps, 2*pi-2*eps)
    torch.clamp_(bboxes2[:, 1:4], eps, pi-2*eps)
    if bboxes1.size(1) == 5:
        torch.clamp_(bboxes2[:, 4], -2*pi+eps, max=2*pi-2*eps)
        torch.clamp_(bboxes2[:, 4], -2*pi+2*eps, max=2*pi-eps)

    return bboxes1, bboxes2