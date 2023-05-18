import torch
from mmdet.models.builder import LOSSES
from mmdet.models.losses import L1Loss

from .sph2pob_transform import Sph2PobTransfrom
import torch.nn.functional as F


@LOSSES.register_module()
@Sph2PobTransfrom()
class Sph2PobL1Loss(L1Loss):
    """L1 loss.

    Args:
        reduction (str, optional): The method to reduce the loss.
            Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of loss.
    """
    def __init__(self, encode=True, swap=False, angle_modifier='original', *args, **kwargs):
        assert angle_modifier in ['original', 'modulus']
        super(Sph2PobL1Loss, self).__init__(*args, **kwargs)
        self.encode = encode 
        self.swap = swap
        self.angle_modifier = angle_modifier
        #self.bbox_coder = DeltaXYWHAOBBoxCoder()

    def forward(self, pred, target, weight=None, *args, **kwargs):
        if self.encode:
            if self.swap:
                pred = bbox2delta(target, pred, angle_modifier=self.angle_modifier)
            else:
                pred = bbox2delta(pred, target, angle_modifier=self.angle_modifier)
            target = torch.zeros_like(target)

        loss = super(Sph2PobL1Loss, self).forward(pred, target, weight, *args, **kwargs)
        return loss


def bbox2delta(proposals,
               gt,
               means=(0., 0., 0., 0., 0.),
               stds=(1., 1., 1., 1., 1.),
               angle_modifier='original',
               eps=1e-7):
    """We usually compute the deltas of x, y, w, h, a of proposals w.r.t ground
    truth bboxes to get regression target. This is the inverse function of
    :func:`delta2bbox`.

    Args:
        proposals (torch.Tensor): Boxes to be transformed, shape (N, ..., 5)
        gt (torch.Tensor): Gt bboxes to be used as base, shape (N, ..., 5)
        means (Sequence[float]): Denormalizing means for delta coordinates
        stds (Sequence[float]): Denormalizing standard deviation for delta
            coordinates.
        angle_range (str, optional): Angle representations. Defaults to 'oc'.
        norm_factor (None|float, optional): Regularization factor of angle.
        edge_swap (bool, optional): Whether swap the edge if w < h.
            Defaults to False.
        proj_xy (bool, optional): Whether project x and y according to angle.
            Defaults to False.

    Returns:
        Tensor: deltas with shape (N, 5), where columns represent dx, dy,
            dw, dh, da.
    """
    assert proposals.size() == gt.size()
    proposals = proposals.float()
    gt = gt.float()
    px, py, pw, ph, pa = proposals.unbind(dim=-1)
    gx, gy, gw, gh, ga = gt.unbind(dim=-1)

    pw, ph = pw.clip(min=eps), ph.clip(min=eps)
    gw, gh = gw.clip(min=eps), gh.clip(min=eps)

    dx = (gx - px) / pw
    dy = (gy - py) / ph

    da = (wrap_angle(ga, angle_modifier) - wrap_angle(pa, angle_modifier)) / torch.pi
    dw = torch.log(gw / pw)
    dh = torch.log(gh / ph)

    deltas = torch.stack([dx, dy, dw, dh, da], dim=-1)
    means = deltas.new_tensor(means).unsqueeze(0)
    stds = deltas.new_tensor(stds).unsqueeze(0)
    deltas = deltas.sub_(means).div_(stds)
    return deltas


def wrap_angle(angle, modifier):
    if modifier == 'original':
        return angle
    elif modifier == 'modulus':
        return (angle + torch.pi) % torch.pi
    else:
        raise NotImplemented('Not supported modifier.')
