from mmdet.models.builder import LOSSES
from mmrotate.models.losses import KFLoss
import torch

from .sph2pob_transform import Sph2PobTransfrom


@LOSSES.register_module()
@Sph2PobTransfrom()
class Sph2PobKFLoss(KFLoss):
    """Kalman filter based loss.

    Args:
        fun (str, optional): The function applied to distance.
            Defaults to 'log1p'.
        reduction (str, optional): The reduction method of the
            loss. Defaults to 'mean'.
        loss_weight (float, optional): The weight of loss. Defaults to 1.0.

    Returns:
        loss (torch.Tensor)
    """
    def forward(self, pred, target, *args, **kwargs):
        #wh = target[:, 2:4].clamp(min=1e-8)
        #whwhw = torch.concat([wh, wh, wh], dim=-1)[:, :5]
        return super(Sph2PobKFLoss, self).forward(pred, target, pred_decode=target, targets_decode=pred, *args, **kwargs)
