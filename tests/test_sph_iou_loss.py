import utils.ext_import

import torch
from sphdet.iou import sph2pob_standard_iou
from sphdet.iou.sph2pob_standard import sph2pob_standard
from sphdet.losses import SphIoULoss, Sph2PobIoULoss, Sph2PobGDLoss

from utils.generate_data import generate_boxes
from utils.timer import Timer


def diff_sph_iou(pred, target, is_aligned=True):
    assert is_aligned in [True]
    from mmcv.ops import diff_iou_rotated_2d
    pred, target = sph2pob_standard(pred.clone(), target.clone(), rbb_angle='rad')
    #print(pred, target, sep='\n\n')
    iou_diff = diff_iou_rotated_2d(pred.unsqueeze(0), target.unsqueeze(0))
    return iou_diff.squeeze()

# ---------------------------------------------------------------------------- #
def test_sph_iou_loss_disparate():
    num = 10
    pt = generate_boxes(num).cuda()
    gt = generate_boxes(num).cuda()

    torch.set_printoptions(precision=4, linewidth=120, sci_mode=False)

    with Timer('xiny', sync=True):
        iou1 = sph2pob_standard_iou(pt, gt, is_aligned=True)
    with Timer('diff', sync=True):  
        iou2 = diff_sph_iou(pt, gt, is_aligned=True)
    if num <= 20:
        print(f'xiny={iou1}\ndiff={iou2}')
    assert torch.abs(iou1 - iou2).mean() < 1e-6

def test_sph_iou_loss_identical():
    num = 10
    pt = generate_boxes(num).cuda()
    gt = pt.detach().clone()

    torch.set_printoptions(precision=4, linewidth=120, sci_mode=False)

    with Timer('xiny', sync=True):
        iou1 = sph2pob_standard_iou(pt, gt, is_aligned=True)
    with Timer('diff', sync=True):
        iou2 = diff_sph_iou(pt, gt, is_aligned=True)
    if num <= 20:
        print(f'xiny={iou1}\ndiff={iou2}')
    #assert torch.abs(iou1 - iou2).mean() < 1e-6

# ---------------------------------------------------------------------------- #
def test_sph_iou_loss_grad_disparate():
    pt = generate_boxes(10).cuda().requires_grad_(True)
    gt = generate_boxes(10).cuda().requires_grad_(True)

    torch.set_printoptions(precision=4, linewidth=120)
    torch.autograd.set_detect_anomaly(True)
    loss = SphIoULoss(reduction='mean', mode='linear')

    loss_val = loss(pt, gt)
    loss_val.backward()
    print(pt.grad, gt.grad, sep='\n\n')

def test_sph_iou_loss_grad_identical():
    pt = generate_boxes(10).cuda().requires_grad_(True)
    gt = pt.detach().clone().requires_grad_(True)

    torch.set_printoptions(precision=4, linewidth=120)
    torch.autograd.set_detect_anomaly(True)
    loss = SphIoULoss(reduction='mean', func='linear')

    loss_val = loss(pt, gt)
    loss_val.backward()
    print(pt.grad, gt.grad, sep='\n\n')

# ---------------------------------------------------------------------------- #
def test_sphobb_iou_loss_and_sph_iou_loss():
    pt = generate_boxes(100).cuda().requires_grad_(True)
    gt = generate_boxes(100).cuda().requires_grad_(True)
    #gt = pt.detach().clone().requires_grad_(True)

    torch.set_printoptions(precision=4, linewidth=120)
    torch.autograd.set_detect_anomaly(True)
    loss1 = SphIoULoss(reduction='none', mode='linear')
    loss2 = Sph2PobIoULoss(reduction='none', mode='iou')

    loss1_val = loss1(pt.clone(), gt.clone())
    loss2_val = loss2(pt.clone(), gt.clone())
    err = torch.abs(loss1_val - loss2_val)
    print('err: mean={:.6f}, var={:.6f}, median={:.6f}, max={:.6f}, min={:.6f}' \
        .format(err.mean().item(), err.var().item(), err.median().item(), \
                err.max().item(), err.min().item()))

# ---------------------------------------------------------------------------- #
def test_sphobb_iou_loss():
    pt = generate_boxes(10000).cuda().requires_grad_(True)
    #gt = generate_boxes(10000).cuda().requires_grad_(True)
    gt = pt.detach().clone().requires_grad_(True)

    torch.set_printoptions(precision=4, linewidth=120)
    torch.autograd.set_detect_anomaly(True)
    loss1 = SphIoULoss(reduction='none', mode='linear')
    loss2 = Sph2PobIoULoss(reduction='none', mode='ciou')

    loss1_val = loss1(pt.clone(), gt.clone())
    loss2_val = loss2(pt.clone(), gt.clone())
    err = torch.abs(loss1_val - loss2_val)
    print('err: mean={:.6f}, var={:.6f}, median={:.6f}, max={:.6f}, min={:.6f}' \
        .format(err.mean().item(), err.var().item(), err.median().item(), \
                err.max().item(), err.min().item()))
    
def test_sph_gaussian_loss_grad_disparate():
    pt = generate_boxes(10).cuda().requires_grad_(True)
    gt = generate_boxes(10).cuda().requires_grad_(True)

    torch.set_printoptions(precision=4, linewidth=120)
    torch.autograd.set_detect_anomaly(True)
    loss = Sph2PobGDLoss(loss_type='kld', reduction='none')
    loss_val = loss(pt, gt)
    loss_val.mean().backward()

    print(loss_val, pt.grad, gt.grad, sep='\n\n')

def test_sph_gaussian_loss_grad_identical():
    pt = generate_boxes(10).cuda().requires_grad_(True)
    gt = pt.detach().clone().requires_grad_(True)

    torch.set_printoptions(precision=4, linewidth=120)
    torch.autograd.set_detect_anomaly(True)

    loss = Sph2PobGDLoss(loss_type='kld', reduction='none')
    loss_val = loss(pt, gt)
    loss_val.mean().backward()

    print(loss_val, pt.grad, gt.grad, sep='\n\n')


# ---------------------------------------------------------------------------- #
if __name__ == '__main__':
    #test_sph_iou_loss_disparate()
    #test_sph_iou_loss_identical()
    #test_sph_iou_loss_grad_disparate()
    #test_sph_iou_loss_grad_identical()
    #test_sphobb_iou_loss_and_sph_iou_loss()
    test_sphobb_iou_loss()
    #test_sph_gaussian_loss_grad_disparate()
    #test_sph_gaussian_loss_grad_identical()
