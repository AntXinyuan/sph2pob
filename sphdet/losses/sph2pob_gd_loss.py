from mmdet.models.builder import LOSSES
from mmrotate.models.losses import GDLoss

from .sph2pob_transform import Sph2PobTransfrom


@LOSSES.register_module()
@Sph2PobTransfrom()
class Sph2PobGDLoss(GDLoss):
    """Sph Gaussian based loss.

    Args:
        loss_type (str):  Type of loss.
        representation (str, optional): Coordinate System.
        fun (str, optional): The function applied to distance.
            Defaults to 'log1p'.
        tau (float, optional): Defaults to 1.0.
        alpha (float, optional): Defaults to 1.0.
        reduction (str, optional): The reduction method of the
            loss. Defaults to 'mean'.
        loss_weight (float, optional): The weight of loss. Defaults to 1.0.

    Returns:
        loss (torch.Tensor)
    """
    pass
