def test_import():
    import utils.ext_import
    import sphdet
    
    import torch
    import numpy
    import mmcv
    import mmdet
    import mmrotate

if __name__ == '__main__':
    test_import()

    from mmrotate.core.bbox.transforms import obb2hbb, obb2poly, obb2xyxy
    from utils.generate_data import generate_boxes
    from sphdet.iou.sph2pob_standard import sph2pob_standard

    pt = generate_boxes(10)
    gt = generate_boxes(10)
    pt, gt = sph2pob_standard(pt.clone(), gt.clone(), rbb_angle_version='rad')
    _pt = obb2poly(pt)
    _pt = obb2xyxy(pt)
    _pt = obb2hbb(pt)


    