import utils.ext_import
import os
import torch
import numpy as np
from sphdet.iou import unbiased_iou, sph2pob_standard_iou, sph2pob_legacy_iou, sph_iou, fov_iou, naive_iou
from sphdet.iou.sph_iou_api import sph2pob_efficient_iou
from sphdet.iou.sph2pob_standard import sph2pob_standard
from sphdet.visualizers.plot_visualizer import plot_scatter_single, plot_curve, plot_scatter
from sphdet.bbox.box_formator import geo2sph

from utils.generate_data import generate_boxes
from utils.timer import Timer

from collections import OrderedDict

from mmcv.ops import box_iou_rotated, diff_iou_rotated_2d, bbox_overlaps
def _rotated_box_iou(bboxes1, bboxes2, angles1=None, angles2=None, is_aligned=True, mode='common'):
    if angles1 is not None and angles2 is not None:
        bboxes1 = torch.concat([bboxes1, angles1], dim=-1)
        bboxes2 = torch.concat([bboxes2, angles2], dim=-1)
    if mode == 'common':
        return box_iou_rotated(bboxes1, bboxes2, aligned=is_aligned)
    else:
        return diff_iou_rotated_2d(bboxes1[None, ...], bboxes2[None, ...])

def _box_iou(bboxes1, bboxes2, is_aligned=True):
    return bbox_overlaps(bboxes1, bboxes2, aligned=is_aligned)

def _get_info_about_single_method(data, iou_calculator, mode='both', device='cpu', gold_iou=None, kwargs=dict()):
    assert mode in ['statistic', 'detail', 'both']
    assert device in ['cpu', 'cuda']

    pt, gt = data
    pt = pt.to(device)
    gt = gt.to(device)

    timer = Timer(name=iou_calculator.__name__, sync=(device == 'cuda'), on_off=True)
    timer.tic()
    overlaps  = iou_calculator(pt, gt, is_aligned=True, **kwargs)
    timer.toc()
    #timer.show()

    overlaps = overlaps.flatten().cpu()
    
    result = dict(
        detail=dict(
            iou=overlaps.numpy()),
        statistic=dict(
            n_samples=len(overlaps),
            time=dict(
                total=timer.duration,
                mean=timer.duration,
                avg=timer.duration / len(overlaps)),
            iou=dict(
                mean=overlaps.mean().item(),
                std=overlaps.std().item(),
                median=overlaps.median().item(),
                max=overlaps.max().item(),
                min=overlaps.min().item()),))

    if gold_iou is not None:
        err = torch.abs(overlaps - gold_iou)
        sort_idx = torch.argsort(err, descending=True)
        R=torch.corrcoef(torch.stack([overlaps, gold_iou]))
        extra_result = dict(
            detail=dict(
                err=err.numpy(),
                sort_idx=sort_idx),
            statistic=dict(
                err=dict(
                    mean=err.mean().item(),
                    std=err.std().item(),
                    median=err.median().item(),
                    max=err.max().item(),
                    min=err.min().item()),
                cor=dict(
                    R=R[0, 1].item(),
                )))
        for key in result.keys():
            result[key].update(extra_result[key])

    if mode in ['detail', 'statistic']:
        return result[mode]
    else:
        return result
    

def _get_infos_about_mutil_methods(size, theta_range=(0, 360), phi_range=(0, 180), alpha_range=(0, 360), beta_range=(0, 360), mode='both', device='cpu'):
    with Timer('data', sync=(device == 'cuda'), on_off=True):
        pt = generate_boxes(size, theta_range, phi_range, alpha_range, beta_range, dtype='float').to(device)
        gt = generate_boxes(size, theta_range, phi_range, alpha_range, beta_range, dtype='float').to(device)

    with torch.no_grad():
        gold_info = _get_info_about_single_method(data=(pt, gt), iou_calculator=unbiased_iou, device='cpu', mode='both')
    gold_iou = torch.from_numpy(gold_info['detail']['iou'])
    if mode in ['detail', 'statistic']:
        gold_info = gold_info[mode]

    with torch.no_grad():
        sph_info  = _get_info_about_single_method(data=(pt, gt), iou_calculator=sph_iou, device=device, mode=mode, gold_iou=gold_iou)
        fov_info = _get_info_about_single_method(data=(pt, gt), iou_calculator=fov_iou, device=device, mode=mode, gold_iou=gold_iou)
        sph2pob_legacy_info  = _get_info_about_single_method(data=(pt, gt), iou_calculator=sph2pob_legacy_iou, device=device, mode=mode, gold_iou=gold_iou)
        sph2pob_standard_info = _get_info_about_single_method(data=(pt, gt), iou_calculator=sph2pob_standard_iou, device=device, mode=mode, gold_iou=gold_iou)
        sph2pob_efficient_info = _get_info_about_single_method(data=(pt, gt), iou_calculator=sph2pob_efficient_iou, device=device, mode=mode, gold_iou=gold_iou)

        #sph2pob_standard_info1 = _get_info_about_single_method(data=(pt, gt), iou_calculator=sph2pob_standard_iou, device=device, mode=mode, gold_iou=gold_iou, kwargs=dict(rbb_edge='chord'))
        #sph2pob_standard_info2 = _get_info_about_single_method(data=(pt, gt), iou_calculator=sph2pob_standard_iou, device=device, mode=mode, gold_iou=gold_iou, kwargs=dict(rbb_edge='tangent'))
        #sph2pob_standard_info3 = _get_info_about_single_method(data=(pt, gt), iou_calculator=sph2pob_standard_iou, device=device, mode=mode, gold_iou=gold_iou, kwargs=dict(rbb_angle='project'))
        
        #sph2pob_standard_diff_info = _get_info_about_single_method(data=(pt, gt), iou_calculator=sph2pob_standard_iou, device=device, mode=mode, gold_iou=gold_iou, kwargs=dict(calculator='diff'))

        #angles1 = torch.randint(-180, 180, size=(size, 1), device=device).deg2rad()
        #angles2 = torch.randint(-180, 180, size=(size, 1), device=device).deg2rad()
        #_get_info_about_single_method(data=(pt, gt), iou_calculator=_rotated_box_iou, device=device, mode=mode, gold_iou=gold_iou, kwargs=dict(angles1=angles1, angles2=angles2))
        #planar_rotated_info = _get_info_about_single_method(data=(pt, gt), iou_calculator=_rotated_box_iou, device=device, mode=mode, gold_iou=gold_iou, kwargs=dict(angles1=angles1, angles2=angles2))
        
        #_get_info_about_single_method(data=(pt, gt), iou_calculator=_rotated_box_iou, device=device, mode=mode, gold_iou=gold_iou, kwargs=dict(angles1=angles1, angles2=angles2, mode='diff'))
        #planar_rotated_diff_info = _get_info_about_single_method(data=(pt, gt), iou_calculator=_rotated_box_iou, device=device, mode=mode, gold_iou=gold_iou, kwargs=dict(angles1=angles1, angles2=angles2, mode='diff'))

        #_get_info_about_single_method(data=(pt, gt), iou_calculator=_box_iou, device=device, mode=mode, gold_iou=gold_iou)
        #planar_normal_info = _get_info_about_single_method(data=(pt, gt), iou_calculator=_box_iou, device=device, mode=mode, gold_iou=gold_iou)
    
    result = OrderedDict(
        sph=sph_info,
        fov=fov_info,
        sph2pob_legacy=sph2pob_legacy_info,
        sph2pob_standard=sph2pob_standard_info,
        #sph2pob_standard_diff=sph2pob_standard_diff_info,
        #sph2pob_standard1=sph2pob_standard_info1,
        #sph2pob_standard2=sph2pob_standard_info2,
        #sph2pob_standard3=sph2pob_standard_info3,
        unbiased=gold_info,
        sph2pob_efficient=sph2pob_efficient_info,)
        #planar_rotated=planar_rotated_info,
        #planar_rotated_diff=planar_rotated_diff_info,
        #planar_normal=planar_normal_info)
    return result


def test_ious_time():
    config = dict(
        size=10000,
        theta_range=(0, 360),
        phi_range=(0, 180),
        alpha_range=(1, 100),
        beta_range=(1, 100),
        device='cpu')

    infos = _get_infos_about_mutil_methods(**config, mode='statistic')
    cond = ', '.join([f'{key}={val}' for key, val in config.items()])
    print(cond)
    log_text = ''
    for attr in ['time', 'err', 'iou']:
        log_text += f'{attr:4s}: '+ ' | '.join(['{:6s}={:4f}'.format(key, val.get(attr, {'mean':0})['mean']) for key, val in infos.items()]) + '\n'
    print(log_text)


def test_ious_time_curve():
    config = dict(
        size=10000,
        theta_range=(0, 360),
        phi_range=(1, 179),
        alpha_range=(1, 100),
        beta_range=(1, 100),
        device='cpu')
    num_exp = 10
    ratios = np.linspace(0, 1, num_exp+1)[1:]
    plot_args = dict(
        unbiased=dict(x=[], y=[], label='unbiased'),
        sph2pob_standard=dict(x=[], y=[], label='sph2pob_standard'),
        sph2pob_legacy =dict(x=[], y=[], label='sph2pob_legacy'),
        sph =dict(x=[], y=[], label='sph'),
        fov=dict(x=[], y=[], label='fov'),
        sph2pob_efficient=dict(x=[], y=[], label='sph2pob_efficient'))

    for i, r in enumerate(ratios):
        cfg = dict(config)
        cfg['size'] = int(cfg['size']*r)
        infos = _get_infos_about_mutil_methods(**cfg, mode='statistic')
        print(f'[{i:2d}/{len(ratios):2d}]')
        log_text = ''
        for attr in ['time', 'err', 'iou']:
            log_text += f'{attr:4s}: '+ ' | '.join(['{:6s}={:4f}'.format(key, val.get(attr, {'mean':0})['mean']) for key, val in infos.items()]) + '\n'
        print(log_text, '\n')
        for key, val in infos.items():
            plot_args[key]['x'].append(val['n_samples'])
            plot_args[key]['y'].append(val['time']['total'])
    
    #print(plot_args)
    plot_curve(plot_args, out_path='vis/test/all_ious/time_cureve.png')


def test_iou_scatter():
    config = dict(
        size=10000,
        theta_range=(0, 360),
        phi_range=(0, 45),
        alpha_range=(1, 100),
        beta_range=(1, 100),
        device='cpu')
    vis_args = OrderedDict(
        #unbiased=dict(x=[], y=[], label='unbiased'),
        sph =dict(label='Sph-IoU', color='#fe2c54'),
        fov=dict(label='FoV-IoU', color='#f7d560'),
        #sph2pob_legacy =dict(x=[], y=[], label='sph2pob_legacy', color='#448ee4'),
        sph2pob_standard=dict(label='Sph2Pob-IoU', color='#12e193'))

    plot_args = []
    phi_ranges = [(0, 180), (45, 90), (0, 20)]
    grid=(len(phi_ranges), len(vis_args))
    for idx, pr in enumerate(phi_ranges):
        config['phi_range'] = pr
        infos = _get_infos_about_mutil_methods(**config, mode='both')
        for key, val in vis_args.items():
            plot_args.append(dict(
                x=infos[key]['detail']['iou'],
                y=infos['unbiased']['detail']['iou'],
                R=infos[key]['statistic']['cor'],
                **val))

    with Timer('plot'):
        plot_scatter(plot_args, out_path=f'vis/test/all_ious/iou_scatter.pdf', all_in_one=False, show_text=True, grid=grid)


def test_iou_error():
    config = dict(
        size=10000,
        theta_range=(0, 360),
        phi_range=(0, 180),
        alpha_range=(1, 100),
        beta_range=(1, 100),
        device='cpu')

    infos = _get_infos_about_mutil_methods(**config, mode='statistic')
    unbiased_info = infos.pop('unbiased')

    log_text = ' | '.join(['iou: ' + ', '.join([f'{key}={val:.4f}' for key, val in unbiased_info['iou'].items()])])
    print(f'unbiased           | {log_text}')
    for method in infos.keys():
        log_text = ' | '.join([f'{attr:3s}: ' + ', '.join([f'{key}={val:.4f}' for key, val in infos[method][attr].items()]) for attr in ('iou', 'err', 'cor')])
        print(f'{method:18s} | {log_text}')

def test_ious_single_smaple():
    bboxes1 = torch.tensor([
        [40, 50, 35, 55],
        [30, 60, 60, 60],
        [50, -78, 25, 46],
        [30, 75, 30, 60],
        [40, 70, 25, 30],
        [30, 75, 30, 30],
        [30, 60, 60, 60]]).float()
    bboxes2 = torch.tensor([
        [35, 20, 37, 50],
        [55, 40, 60, 60],
        [30, -75, 26, 45],
        [60, 40, 60, 60],
        [60, 85, 30, 30],
        [60, 55, 40, 50],
        [60, 60, 60, 60]]).float()
    bboxes1 = geo2sph(bboxes1)
    bboxes2 = geo2sph(bboxes2)

    bboxes = torch.tensor([
        [[270, 40, 60, 20], [300, 50, 60, 20]],
        [[0, 127, 20, 40,], [61, 127, 20, 40,]],
    ]).view((-1, 2, 4)).float()
    #bboxes1, bboxes2 = bboxes[:, 0], bboxes[:, 1]

    from sphdet.losses import Sph2PobKFLoss, Sph2PobGDLoss

    result = dict(
        unbiased=unbiased_iou(bboxes1, bboxes2, is_aligned=True),
        planar=naive_iou(bboxes1, bboxes2, is_aligned=True),
        sph2pob_legacy=sph2pob_legacy_iou(bboxes1, bboxes2, is_aligned=True),
        sph2pob_standard=sph2pob_standard_iou(bboxes1, bboxes2, is_aligned=True),
        sph2pob_efficient=sph2pob_efficient_iou(bboxes1, bboxes2, is_aligned=True),
        sph=sph_iou(bboxes1, bboxes2, is_aligned=True),
        fov=fov_iou(bboxes1, bboxes2, is_aligned=True))
        #kfiou=SphOBBKFLoss()(bboxes1, bboxes2),
        #kld=SphGDLoss(loss_type='kld', tau=1.0, fun='log1p', sqrt=False,)(bboxes1, bboxes2),
        #gwd=SphGDLoss(loss_type='gwd', tau=1.0, fun='log1p',)(bboxes1, bboxes2),)
    
    log_text = '\n'.join(['{:18}: {}'.format(key, val) for key, val in result.items()])
    print(log_text)

    #from sphdet.visualizers import SphVisualizer
    
    #colors = [(130, 101, 245), (255, 140, 98)]
    #for i, (b1, b2) in enumerate(zip(bboxes1.numpy(), bboxes2.numpy())):
    #    vis = SphVisualizer()
        #vis.add_boxes(boxes=(b1, b2), colors=colors)
        #vis.show(out_path=f'vis/test_temp/all_ious/temp_{i+1}.png')

def test_iou_angle():
    config = dict(
        num=1000000,
        theta_range=(0, 360),
        phi_range=(0, 180),
        alpha_range=(1, 100),
        beta_range=(1, 100),
        gamma_range=(-60, 60),
        version='deg', 
        dtype='float', 
        box='bfov')

    device='cpu'
    with Timer('data'):
        boxes1 = generate_boxes(**config).to(device)
        boxes2 = generate_boxes(**config).to(device)
    
    iou = sph2pob_standard_iou(boxes1, boxes2, is_aligned=True)
    _boxes1, _boxes2 = sph2pob_standard(boxes1, boxes2, rbb_angle_version='deg')
    angle = _boxes1[:, -1] - _boxes2[:, -1]

    keep = iou > 0.5
    iou = iou[keep]
    angle = angle[keep] #+ 360) % 360 - 180

    with Timer('plot'):
        plot_scatter_single(iou.cpu().numpy(), angle.cpu().numpy(), angle.cpu().numpy(), out_path='vis/test/iou-angle.png')

def test_whether_bboxes_change_or_not():
    old_pt = generate_boxes(1)
    old_gt = generate_boxes(1)
    new_gt = old_gt.detach().clone()
    new_pt = old_pt.detach().clone()
    overlaps1 = unbiased_iou(new_pt, new_gt)
    assert (old_gt == new_gt).all() and (old_pt == new_pt).all()
    #print(old_gt, new_gt, old_pt, new_pt, sep='\n\n')
    overlaps2 = sph2pob_standard_iou(new_pt, new_gt)
    assert (old_gt == new_gt).all() and (old_pt == new_pt).all()
    #print(old_gt, new_gt, old_pt, new_pt, sep='\n\n')

if __name__ == '__main__':
    #test_ious_time()
    #test_ious_single_smaple()
    #test_ious_time_curve()
    #test_iou_scatter()
    test_iou_error()
    #test_iou_angle()
    #test_whether_bboxes_change_or_not()