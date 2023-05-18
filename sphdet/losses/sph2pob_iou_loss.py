import math

import torch
import torch.nn as nn
from mmcv.ops import diff_iou_rotated_2d
from mmdet.models.builder import LOSSES
from mmdet.models.losses import weighted_loss
from mmrotate.models.losses import RotatedIoULoss

from sphdet.bbox.box_formator import obb2hbb_xyxy
from sphdet.iou import fov_iou, sph2pob_standard_iou, sph_iou

from .sph2pob_transform import Sph2PobTransfrom


class OBBIoULoss(nn.Module):
    def __init__(self, mode='iou', eps=1e-6, reduction='mean', loss_weight=1.0):
        super(OBBIoULoss, self).__init__()
        assert mode in ['iou', 'giou', 'diou', 'ciou']
        self.mode = mode
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        if weight is not None and not torch.any(weight > 0):
            if pred.dim() == weight.dim() + 1:
                weight = weight.unsqueeze(1)
            return (pred * weight).sum()  # 0
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if weight is not None and weight.dim() > 1:
            # TODO: remove this in the future
            # reduce the weight of shape (n, 4) to (n,) to match the
            # giou_loss of shape (n,) 
            assert weight.shape == pred.shape
            weight = weight.mean(-1)
        loss = self.loss_weight * obb_iou_loss(
            pred,
            target,
            weight,
            mode=self.mode,
            eps=self.eps,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss

@weighted_loss
def obb_iou_loss(pred, target, mode='iou', eps=1e-7):
    r"""Several versions of iou-based loss for OBB.

    Args:
        pred (Tensor): Predicted bboxes of format (cx, cy, w, h, a(rad)),
            shape (n, 5).
        target (Tensor): Corresponding gt bboxes, shape (n, 5).
        mode (str): Version of iou-based lossm, including "iou", "giou", 
            "diou", and "ciou". Default: 'iou'.
        eps (float): Eps to avoid log(0).
    Return:
        Tensor: Loss tensor.
    """
    
    #_pred, _target = pred.clone(), target.clone()
    #_pred, _target = jiter_rotated_bboxes(_pred, _target)
    ious = diff_iou_rotated_2d(pred.unsqueeze(0), target.unsqueeze(0)).squeeze().clamp(min=0, max=1.0)

    if mode == 'iou':
        loss = 1 - ious.clamp(min=0, max=1.0)
        return loss

    hbb_pred = obb2hbb_xyxy(pred)
    hbb_target = obb2hbb_xyxy(target)

    # enclose area
    enclose_x1y1 = torch.min(hbb_pred[:, :2], hbb_target[:, :2])
    enclose_x2y2 = torch.max(hbb_pred[:, 2:], hbb_target[:, 2:])
    enclose_wh = (enclose_x2y2 - enclose_x1y1).clamp(min=0)

    if mode == 'giou':
        inter_x1y1 = torch.max(hbb_pred[:, :2], hbb_target[:, :2])
        inter_x2y2 = torch.min(hbb_pred[:, 2:], hbb_target[:, 2:])
        inter_wh   = (inter_x2y2 - inter_x1y1).clamp(min=0)

        area_enclose = enclose_wh[:, 0] * enclose_wh[:, 1]
        area_inter   = inter_wh[:, 0] * inter_wh[:, 1]
        area_pred    = pred[:, 2] * pred[:, 3]
        area_target  = target[:, 2] * target[:, 3]
        area_union   = area_pred + area_target - area_inter
        
        area_ratio = (area_enclose - area_union) / (area_enclose + eps)
        gious = ious - area_ratio.clamp(min=0, max=1.0)
        loss = 1 - gious
        return loss

    cw = enclose_wh[:, 0]
    ch = enclose_wh[:, 1]

    c2 = cw**2 + ch**2 + eps

    b1_cx, b1_cy, w1, h1 = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    b2_cx, b2_cy, w2, h2 = target[:, 0], target[:, 1], target[:, 2], target[:, 3]

    left = (b2_cx - b1_cx)**2
    right = (b2_cy - b1_cy)**2
    rho2 = left + right

    if mode == 'diou':
        #print(torch.concat([pred, target], dim=1))
        #print(ious, (rho2 / c2))
        dious = ious - (rho2 / c2).clamp(min=0, max=1.0)
        loss = 1 - dious
        return loss

    factor = 4 / math.pi**2
    v = factor * torch.pow(torch.atan(w2 / (h2+eps)) - torch.atan(w1 / (h1+eps)), 2)

    with torch.no_grad():
        alpha = (ious > 0.5).float() * v / (1 - ious + v + eps)

    if mode == 'ciou':
        cious = ious - ((rho2 / c2).clamp(min=0, max=1.0) + alpha * v)
        loss = 1 - cious
        return loss
    else:
        raise NotImplemented('Not supported version of iou-based loss.')

# ---------------------------------------------------------------------------- #
@LOSSES.register_module()
@Sph2PobTransfrom()
class SphIoULossLegacy(RotatedIoULoss):
    """SphRotatedIoULoss.

    Computing the IoU loss between a set of predicted rbboxes and
    target rbboxes.
    Args:
        linear (bool): If True, use linear scale of loss else determined
            by mode. Default: False.
        eps (float): Eps to avoid log(0).
        reduction (str): Options are "none", "mean" and "sum".
        loss_weight (float): Weight of loss.
        mode (str): Loss scaling mode, including "linear", "square", and "log".
            Default: 'log'
    """
    pass

# ---------------------------------------------------------------------------- #
@LOSSES.register_module()
@Sph2PobTransfrom()
class Sph2PobIoULoss(OBBIoULoss):
    """SphOBBIoULoss.

    Computing the IoU loss between a set of predicted bboxes and target bboxes.

    Args:
        pred (Tensor): Predicted bboxes of format (cx, cy, w, h, a(rad)),
            shape (n, 5).
        target (Tensor): Corresponding gt bboxes, shape (n, 5).
        mode (str): Version of iou-based lossm, including "iou", 
            "diou", and "ciou". Default: 'iou'.
        eps (float): Eps to avoid log(0).
    Return:
        Tensor: Loss tensor.
    """
    pass
# ---------------------------------------------------------------------------- #

@LOSSES.register_module()
class SphIoULoss(nn.Module):
    def __init__(self, mode='iou', iou_calculator='sph2pob_standard_iou', eps=1e-6, reduction='mean', loss_weight=1.0):
        super().__init__()
        assert mode in ['iou', 'giou', 'diou', 'ciou']
        assert iou_calculator in ['sph2pob_standard', 'sph', 'fov']
        self.mode = mode
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight

        if iou_calculator == 'sph':
            self.iou_calculator = sph_iou
        elif iou_calculator == 'fov':
            self.iou_calculator = fov_iou
        else:
            self.iou_calculator = sph2pob_standard_iou

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        if weight is not None and not torch.any(weight > 0):
            if pred.dim() == weight.dim() + 1:
                weight = weight.unsqueeze(1)
            return (pred * weight).sum()  # 0
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if weight is not None and weight.dim() > 1:
            # TODO: remove this in the future
            # reduce the weight of shape (n, 4) to (n,) to match the
            # giou_loss of shape (n,) 
            assert weight.shape == pred.shape
            weight = weight.mean(-1)
        loss = self.loss_weight * sph_iou_loss(
            pred,
            target,
            weight,
            mode=self.mode,
            reduction=reduction,
            avg_factor=avg_factor,
            iou_calculator=self.iou_calculator,
            **kwargs)
        return loss

@weighted_loss
def sph_iou_loss(pred, target, mode='iou', iou_calculator=sph2pob_standard_iou):
    r"""Several versions of iou-based loss for spherical boxes.
    """
    ious = iou_calculator(pred, target, is_aligned=True, calculator='diff')

    if mode == 'iou':
        loss = 1 - ious.clamp(min=0, max=1.0)
        return loss
