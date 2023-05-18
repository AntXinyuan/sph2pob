from mmdet.core.bbox.iou_calculators.builder import IOU_CALCULATORS

from .sph_iou_api import unbiased_iou, sph2pob_standard_iou, sph2pob_legacy_iou, sph2pob_efficient_iou, naive_iou, fov_iou, sph_iou
import torch


@IOU_CALCULATORS.register_module()
class SphOverlaps2D(object):
    """2D Overlaps (e.g. IoUs, GIoUs) Calculator."""

    def __init__(self, backend='unbiased_iou', box_version=4):
        self.backend = backend
        self.box_version = box_version

    def __call__(self,
                 bboxes1,
                 bboxes2,
                 mode='iou',
                 is_aligned=False):
        """Calculate IoU between 2D bboxes.

        Args:
            bboxes1 (torch.Tensor): bboxes have shape (m, 4) in
                <theta, phi, alpha, beta, (angle)> format, or shape (m, 5) in
                 <theta, phi, alpha, beta, (angle), score> format.
            bboxes2 (torch.Tensor): bboxes have shape (m, 4) in
                <theta, phi, alpha, beta, (angle)> format, shape (m, 5) in
                 <theta, phi, alpha, beta, (angle), score> format, or be empty.
                 If ``is_aligned `` is ``True``, then m and n must be equal.
            mode (str): "iou" (intersection over union), "iof" (intersection
                over foreground), or "giou" (generalized intersection over
                union).
            is_aligned (bool, optional): If True, then m and n must be equal.
                Default False.

        Returns:
            Tensor: shape (m, n) if ``is_aligned `` is False else shape (m,)
        """
        #print('SphOverlaps2D.__call__()...')
        assert bboxes1.size(-1) in [0, 4, 5, 6]
        assert bboxes2.size(-1) in [0, 4, 5, 6]

        bboxes1 = bboxes1[..., :self.box_version]
        bboxes2 = bboxes2[..., :self.box_version]
        if self.box_version == 5:
            assert self.backend in ['unbiased_iou', 'sph2pob_legacy_iou', 'sph2pob_efficient_iou']
        with torch.no_grad():
            overlaps = sph_overlaps(bboxes1, bboxes2, mode, is_aligned, self.backend)
        return overlaps

    def __repr__(self):
        """str: a string describing the module"""
        repr_str = self.__class__.__name__ + '()'
        return repr_str

def sph_overlaps(bboxes1, bboxes2, mode='iou', is_aligned=False, backend='unbiased_iou'):
    """Calculate overlap between two set of bboxes.

    Args:
        bboxes1 (torch.Tensor): shape (m, 4) in <theta, phi, alpha, beta, (angle)> format
            or empty.
        bboxes2 (torch.Tensor): shape (n, 4) in <theta, phi, alpha, beta, (angle)> format
            or empty.
        mode (str): "iou" (intersection over union), "iof" (intersection over
            foreground) or "giou" (generalized intersection over union).
            Default "iou".
        is_aligned (bool, optional): If True, then m and n must be equal.
            Default False.

    Returns:
        Tensor: shape (m, n) if ``is_aligned`` is False else shape (m,)
    """
    assert mode in ['iou', 'iof']
    assert backend in ['unbiased_iou', 'sph2pob_standard_iou', 'sph2pob_legacy_iou', 'sph2pob_efficient_iou', 'naive_iou', 'fov_iou', 'sph_iou']
    # Either the boxes are empty or the length of boxes's last dimension is 4
    #assert (bboxes1.size(-1) == 4 or bboxes1.size(0) == 0)
    #assert (bboxes2.size(-1) == 4 or bboxes2.size(0) == 0)

    device = bboxes1.device
    rows = bboxes1.size(0)
    cols = bboxes2.size(0)

    if rows * cols == 0:
        return bboxes1.new(rows, 1) if is_aligned else bboxes1.new(rows, cols)

    if backend == 'unbiased_iou':
        iou_calculator = unbiased_iou
    elif backend == 'sph2pob_legacy_iou':
        iou_calculator = sph2pob_legacy_iou
    elif backend == 'sph2pob_standard_iou':
        iou_calculator = sph2pob_standard_iou
    elif backend == 'sph2pob_efficient_iou':
        iou_calculator = sph2pob_efficient_iou
    elif backend == 'naive_iou':
        iou_calculator = naive_iou
    elif backend == 'fov_iou':
        iou_calculator = fov_iou
    elif backend == 'sph_iou':
        iou_calculator = sph_iou
    else:
        raise NotImplemented('Not supported iou_calculator.')
    
    if backend == 'unbiased_iou':
        overlaps = iou_calculator(bboxes1.cpu(), bboxes2.cpu(), mode, is_aligned).to(device)
    else:
        overlaps = iou_calculator(bboxes1, bboxes2, mode, is_aligned)
    return overlaps
