import functools

import torch

from sphdet.iou.sph2pob_efficient import sph2pob_efficient
from sphdet.iou.sph2pob_legacy import sph2pob_legacy
from sphdet.iou.sph2pob_standard import sph2pob_standard
from sphdet.iou.sph_iou_api import jiter_rotated_bboxes, jiter_spherical_bboxes


class Sph2PobTransfrom:
    def __init__(self, transform='sph2pob_standard'):
        assert transform in ['sph2pob_standard', 'sph2pob_legacy']
        if transform =='sph2pob_standard':
            self.transform = sph2pob_standard
        elif transform == 'sph2pob_legacy':
            self.transform = sph2pob_legacy
        else:
            raise NotImplemented('Not supported tranform.')

    def __call__(self, cls):
        old_forward = cls.forward
        @functools.wraps(old_forward)
        def new_forward(_self_, pred, target, weight=None, *args, **kwargs):
            box_version = target.size(-1)
            pred, target = pred.clone(), target.clone()

            pred, target = jiter_spherical_bboxes(pred, target)
            pred, target = self.transform(pred, target, rbb_angle_version='rad')
            pred, target = jiter_rotated_bboxes(pred, target)

            if weight is not None and weight.dim() > 1:
                if box_version == 4:
                    weight = torch.cat([weight, weight.mean(-1, keepdim=True)], dim=-1)
            return old_forward(_self_, pred, target, weight, *args, **kwargs)
        cls.forward = new_forward
        return cls


