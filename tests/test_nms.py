import utils.ext_import

from sphdet.bbox.nms import SphNMS, PlanarNMS
import torch

def genetate_nms_data():
    nms_cfg = dict(
        type='nms',
        iou_threshold=0.5)
    boxes = torch.tensor([
        [20, 40, 30, 30], #0
        [20, 40, 30, 30], #1
        [22, 38, 32, 28], #2
        [60, 60, 10, 10], #3
        [60, 60, 10, 10], #4
        #
        [60, 60, 10, 10], #5
        [60, 60, 10, 10], #6
        [30, 10, 10, 10], #7
        #
        [30, 45, 45, 45], #8
        [80, 20, 66, 66]]).float()#9
    scores = torch.tensor([
        0.9, 0.8, 0.7, 0.6, 0.5, 0.85, 0.75, 0.65, 0.4, 0.3])
    idxs = torch.tensor([
        1, 1, 1, 1, 1, 2, 2, 2, 3, 3])
    return boxes, scores, idxs, nms_cfg

def test_nms():
    nms = PlanarNMS(box_formator='sph2pix')
    boxes, scores, idxs, nms_cfg = genetate_nms_data()
    _, keep = nms(boxes, scores, idxs, nms_cfg, class_agnostic=False)
    print('√: ', keep, scores[keep], idxs[keep])

    nms = SphNMS(iou_calculator='planar')
    boxes, scores, idxs, nms_cfg = genetate_nms_data()
    _, keep = nms(boxes, scores, idxs, nms_cfg)
    print('x: ', keep, scores[keep], idxs[keep])


if __name__ == '__main__':
    test_nms()