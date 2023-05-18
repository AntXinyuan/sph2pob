import sys
import math
import torch
import matplotlib.pyplot as plt
import mmcv
import numpy as np
import pycocotools.mask as mask_util
from matplotlib.collections import PatchCollection
from matplotlib.patches import PathPatch, Polygon
from matplotlib.path import Path
from mmdet.core.evaluation.panoptic_utils import INSTANCE_OFFSET
from mmdet.core.visualization.image import (EPS, _get_adaptive_scales,
                                            _get_bias_color, draw_labels)
from mmdet.core.visualization.palette import get_palette, palette_val
from mmdet.core.utils import mask2ndarray

from .ImageRecorder import ImageRecorder as BFoV
from .ImageRecoderTools import ro_Shpbbox

def box_transform(box):
    eps=1
    box[0] = box[0] - 180
    box[1] = 90 - box[1]
    box[2] = max(min(box[2], 180-eps), eps)
    box[3] = max(min(box[3], 180-eps), eps)
    #box[4] = -box[4]
    return box

class PointsPatch(PathPatch):
    def __init__(self, xy, **kwargs):
        N = len(xy)
        code_mask = np.random.rand(N) < 0.7
        code_mask[0] = code_mask[-1] = True
        codes = np.repeat(Path.LINETO, len(xy))
        codes[code_mask] = Path.MOVETO

        path = Path(xy, codes)
        super(PointsPatch, self).__init__(path, **kwargs)


def draw_sph_bboxes_v1(ax, bboxes, color='g', alpha=0.8, thickness=2):
    """Draw bounding boxes on the axes.

    Args:
        ax (matplotlib.Axes): The input axes.
        bboxes (ndarray): The input bounding boxes with the shape
            of (n, 4).
        color (list[tuple] | matplotlib.color): the colors for each
            bounding boxes.
        alpha (float): Transparency of bounding boxes. Default: 0.8.
        thickness (int): Thickness of lines. Default: 2.

    Returns:
        matplotlib.Axes: The result axes.
    """
    erp_w, erp_h = 1024, 512
    polygons = []
    for i, bbox in enumerate(bboxes):
        #bbox_int = bbox.astype(np.int32)
        #poly = [[bbox_int[0], bbox_int[1]], [bbox_int[0], bbox_int[3]],
        #        [bbox_int[2], bbox_int[3]], [bbox_int[2], bbox_int[1]]]
        #np_poly = np.array(poly).reshape((4, 2))
        bfov = BFoV(erp_w, erp_h, view_angle_w=bbox[2], view_angle_h=bbox[3], long_side=erp_w)
        px, py = bfov._sample_points(np.deg2rad(bbox[0]-180), np.deg2rad(90-bbox[1]), border_only=True)
        #bfov.draw_Sphbbox(ax, px, py, border_only=True)
        np_poly = np.stack([px, py], axis=1).astype(np.int32)
        #for x, y in np_poly:
        #    pt = plt.Circle((x, y), thickness, color=(0, 0, 1), fill=True)
        #    ax.add_patch(pt)
        polygons.append(Polygon(np_poly))
    p = PatchCollection(
        polygons,
        facecolor='none',
        edgecolors=color,
        linewidths=thickness,
        alpha=alpha)
    ax.add_collection(p)

    return ax

def draw_sph_boxes_v2(img, bboxes, colors='g', alpha=0.8, thickness=2):
    width, height = img.shape[1], img.shape[0]
    if colors == 'g':
        colors = [(0, 255, 0)] * len(bboxes)
    else:
        colors = (np.array(colors)[:, [2, 1, 0]] * 255).astype(np.int32).tolist()
    for bbox, color in zip(bboxes, colors):
        bbox = box_transform(bbox)
        bfov = BFoV(width, height, view_angle_w=bbox[2], view_angle_h=bbox[3], long_side=width)
        px, py = bfov._sample_points(np.deg2rad(bbox[0]), np.deg2rad(bbox[1]), border_only=True)
        print(bboxes)
        if len(bbox) == 5:
            px, py = ro_Shpbbox(bbox, px, py, erp_w=width, erp_h=height)
        bfov.draw_Sphbbox(img ,px, py, border_only=True, color=color)


def imshow_det_bboxes(img,
                      bboxes=None,
                      labels=None,
                      segms=None,
                      class_names=None,
                      score_thr=-1,
                      bbox_color='green',
                      text_color='green',
                      mask_color=None,
                      thickness=2,
                      font_size=8,
                      win_name='',
                      show=True,
                      wait_time=0,
                      out_file=None):
    """Draw bboxes and class labels (with scores) on an image.

    Args:
        img (str | ndarray): The image to be displayed.
        bboxes (ndarray): Bounding boxes (with scores), shaped (n, 4) or
            (n, 5).
        labels (ndarray): Labels of bboxes.
        segms (ndarray | None): Masks, shaped (n,h,w) or None.
        class_names (list[str]): Names of each classes.
        score_thr (float): Minimum score of bboxes to be shown. Default: -1.
           -1 means that ignore score(i.e. for showing dataset). >0 means 
           that use score(i.e. for showing prediction).
        bbox_color (list[tuple] | tuple | str | None): Colors of bbox lines.
           If a single color is given, it will be applied to all classes.
           The tuple of color should be in RGB order. Default: 'green'.
        text_color (list[tuple] | tuple | str | None): Colors of texts.
           If a single color is given, it will be applied to all classes.
           The tuple of color should be in RGB order. Default: 'green'.
        mask_color (list[tuple] | tuple | str | None, optional): Colors of
           masks. If a single color is given, it will be applied to all
           classes. The tuple of color should be in RGB order.
           Default: None.
        thickness (int): Thickness of lines. Default: 2.
        font_size (int): Font size of texts. Default: 13.
        show (bool): Whether to show the image. Default: True.
        win_name (str): The window name. Default: ''.
        wait_time (float): Value of waitKey param. Default: 0.
        out_file (str, optional): The filename to write the image.
            Default: None.

    Returns:
        ndarray: The image with bboxes drawn on it.
    """
    assert bboxes is None or bboxes.ndim == 2, \
        f' bboxes ndim should be 2, but its ndim is {bboxes.ndim}.'
    assert labels.ndim == 1, \
        f' labels ndim should be 1, but its ndim is {labels.ndim}.'
    #assert bboxes is None or bboxes.shape[1] == 4 or bboxes.shape[1] == 5, \
    #    f' bboxes.shape[1] should be 4 or 5, but its {bboxes.shape[1]}.'
    assert bboxes is None or bboxes.shape[0] <= labels.shape[0], \
        'labels.shape[0] should not be less than bboxes.shape[0].'
    assert segms is None or segms.shape[0] == labels.shape[0], \
        'segms.shape[0] and labels.shape[0] should have the same length.'
    assert segms is not None or bboxes is not None, \
        'segms and bboxes should not be None at the same time.'

    img = mmcv.imread(img).astype(np.uint8)

    if score_thr >= 0:
        #assert bboxes is not None and bboxes.shape[1] == (box_version+1)
        scores = bboxes[:, -1]
        inds = scores > score_thr
        bboxes = bboxes[inds, :]
        labels = labels[inds]
        if segms is not None:
            segms = segms[inds, ...]

    img = mmcv.bgr2rgb(img)
    width, height = img.shape[1], img.shape[0]
    img = np.ascontiguousarray(img)

    fig = plt.figure(win_name, frameon=False)
    plt.title(win_name)
    canvas = fig.canvas
    dpi = fig.get_dpi()
    # add a small EPS to avoid precision lost due to matplotlib's truncation
    # (https://github.com/matplotlib/matplotlib/issues/15363)
    fig.set_size_inches((width + EPS) / dpi, (height + EPS) / dpi)

    # remove white edges by set subplot margin
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax = plt.gca()
    ax.axis('off')

    max_label = int(max(labels) if len(labels) > 0 else 0)
    text_palette = palette_val(get_palette(text_color, max_label + 1))
    text_colors = [text_palette[label] for label in labels]

    num_bboxes = 0
    if bboxes is not None:
        num_bboxes = bboxes.shape[0]
        bbox_palette = palette_val(get_palette(bbox_color, max_label + 1))
        colors = [bbox_palette[label] for label in labels[:num_bboxes]]
        #draw_sph_bboxes_v1(ax, bboxes, colors, alpha=0.8, thickness=thickness)
        #draw_sph_bboxes_v2(img, bboxes, colors, alpha=0.8, thickness=thickness)
        horizontal_alignment = 'left'
        positions = (bboxes[:, :2] * (width / 360.)).astype(np.int32) + thickness
        areas = bboxes[:, 2] * bboxes[:, 3] * (width / 360.)
        scales = _get_adaptive_scales(areas)
        #scores = bboxes[:, -1] if bboxes.shape[1] == (box_version+1) else None
        scores = bboxes[:, -1] if score_thr >= 0 else None
        draw_labels(
            ax,
            labels[:num_bboxes],
            positions,
            scores=scores,
            class_names=class_names,
            color=text_colors,
            font_size=font_size,
            scales=scales,
            horizontal_alignment=horizontal_alignment)

    plt.imshow(img)

    stream, _ = canvas.print_to_buffer()
    buffer = np.frombuffer(stream, dtype='uint8')
    if sys.platform == 'darwin':
        width, height = canvas.get_width_height(physical=True)
    img_rgba = buffer.reshape(height, width, 4)
    rgb, alpha = np.split(img_rgba, [3], axis=2)
    img = rgb.astype('uint8')
    img = mmcv.rgb2bgr(img)

    if bboxes is not None:
        bboxes = bboxes[:, :-1] if score_thr >= 0 else bboxes
        #if score_thr >= 0:
        #    bboxes[:, -1] = bboxes[:, -1] / math.pi * 180
        if bboxes.shape[0] > 0:
            draw_sph_boxes_v2(img, bboxes, colors)
        #print(bboxes)
        #print('\n\n')

    if show:
        # We do not use cv2 for display because in some cases, opencv will
        # conflict with Qt, it will output a warning: Current thread
        # is not the object's thread. You can refer to
        # https://github.com/opencv/opencv-python/issues/46 for details
        if wait_time == 0:
            plt.show()
        else:
            plt.show(block=False)
            plt.pause(wait_time)
    if out_file is not None:
        mmcv.imwrite(img, out_file)

    plt.close()

    return img


def imshow_gt_det_bboxes(img,
                         annotation,
                         result,
                         class_names=None,
                         score_thr=0,
                         gt_bbox_color=(61, 102, 255),
                         gt_text_color=(200, 200, 200),
                         gt_mask_color=(61, 102, 255),
                         det_bbox_color=(241, 101, 72),
                         det_text_color=(200, 200, 200),
                         det_mask_color=(241, 101, 72),
                         thickness=2,
                         font_size=13,
                         win_name='',
                         show=True,
                         wait_time=0,
                         out_file=None,
                         overlay_gt_pred=True):
    """General visualization GT and result function.

    Args:
      img (str | ndarray): The image to be displayed.
      annotation (dict): Ground truth annotations where contain keys of
          'gt_bboxes' and 'gt_labels' or 'gt_masks'.
      result (tuple[list] | list): The detection result, can be either
          (bbox, segm) or just bbox.
      class_names (list[str]): Names of each classes.
      score_thr (float): Minimum score of bboxes to be shown. Default: 0.
      gt_bbox_color (list[tuple] | tuple | str | None): Colors of bbox lines.
          If a single color is given, it will be applied to all classes.
          The tuple of color should be in RGB order. Default: (61, 102, 255).
      gt_text_color (list[tuple] | tuple | str | None): Colors of texts.
          If a single color is given, it will be applied to all classes.
          The tuple of color should be in RGB order. Default: (200, 200, 200).
      gt_mask_color (list[tuple] | tuple | str | None, optional): Colors of
          masks. If a single color is given, it will be applied to all classes.
          The tuple of color should be in RGB order. Default: (61, 102, 255).
      det_bbox_color (list[tuple] | tuple | str | None):Colors of bbox lines.
          If a single color is given, it will be applied to all classes.
          The tuple of color should be in RGB order. Default: (241, 101, 72).
      det_text_color (list[tuple] | tuple | str | None):Colors of texts.
          If a single color is given, it will be applied to all classes.
          The tuple of color should be in RGB order. Default: (200, 200, 200).
      det_mask_color (list[tuple] | tuple | str | None, optional): Color of
          masks. If a single color is given, it will be applied to all classes.
          The tuple of color should be in RGB order. Default: (241, 101, 72).
      thickness (int): Thickness of lines. Default: 2.
      font_size (int): Font size of texts. Default: 13.
      win_name (str): The window name. Default: ''.
      show (bool): Whether to show the image. Default: True.
      wait_time (float): Value of waitKey param. Default: 0.
      out_file (str, optional): The filename to write the image.
          Default: None.
      overlay_gt_pred (bool): Whether to plot gts and predictions on the
       same image. If False, predictions and gts will be plotted on two same
       image which will be concatenated in vertical direction. The image
       above is drawn with gt, and the image below is drawn with the
       prediction result. Default: True.

    Returns:
        ndarray: The image with bboxes or masks drawn on it.
    """
    assert 'gt_bboxes' in annotation
    assert 'gt_labels' in annotation
    assert isinstance(result, (tuple, list, dict)), 'Expected ' \
        f'tuple or list or dict, but get {type(result)}'

    gt_bboxes = annotation['gt_bboxes']
    gt_labels = annotation['gt_labels']
    gt_masks = annotation.get('gt_masks', None)
    if gt_masks is not None:
        gt_masks = mask2ndarray(gt_masks)

    gt_seg = annotation.get('gt_semantic_seg', None)
    if gt_seg is not None:
        pad_value = 255  # the padding value of gt_seg
        sem_labels = np.unique(gt_seg)
        all_labels = np.concatenate((gt_labels, sem_labels), axis=0)
        all_labels, counts = np.unique(all_labels, return_counts=True)
        stuff_labels = all_labels[np.logical_and(counts < 2,
                                                 all_labels != pad_value)]
        stuff_masks = gt_seg[None] == stuff_labels[:, None, None]
        gt_labels = np.concatenate((gt_labels, stuff_labels), axis=0)
        gt_masks = np.concatenate((gt_masks, stuff_masks.astype(np.uint8)),
                                  axis=0)
        # If you need to show the bounding boxes,
        # please comment the following line
        # gt_bboxes = None

    img = mmcv.imread(img)

    img_with_gt = imshow_det_bboxes(
        img,
        gt_bboxes,
        gt_labels,
        gt_masks,
        class_names=class_names,
        bbox_color=gt_bbox_color,
        text_color=gt_text_color,
        mask_color=gt_mask_color,
        thickness=thickness,
        font_size=font_size,
        win_name=win_name,
        show=False)

    if not isinstance(result, dict):
        if isinstance(result, tuple):
            bbox_result, segm_result = result
            if isinstance(segm_result, tuple):
                segm_result = segm_result[0]  # ms rcnn
        else:
            bbox_result, segm_result = result, None

        bboxes = np.vstack(bbox_result)
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
        ]
        labels = np.concatenate(labels)

        segms = None
        if segm_result is not None and len(labels) > 0:  # non empty
            segms = mmcv.concat_list(segm_result)
            segms = mask_util.decode(segms)
            segms = segms.transpose(2, 0, 1)
    else:
        assert class_names is not None, 'We need to know the number ' \
                                        'of classes.'
        VOID = len(class_names)
        bboxes = None
        pan_results = result['pan_results']
        # keep objects ahead
        ids = np.unique(pan_results)[::-1]
        legal_indices = ids != VOID
        ids = ids[legal_indices]
        labels = np.array([id % INSTANCE_OFFSET for id in ids], dtype=np.int64)
        segms = (pan_results[None] == ids[:, None, None])

    if overlay_gt_pred:
        img = imshow_det_bboxes(
            img_with_gt,
            bboxes,
            labels,
            segms=segms,
            class_names=class_names,
            score_thr=score_thr,
            bbox_color=det_bbox_color,
            text_color=det_text_color,
            mask_color=det_mask_color,
            thickness=thickness,
            font_size=font_size,
            win_name=win_name,
            show=show,
            wait_time=wait_time,
            out_file=out_file)
    else:
        img_with_det = imshow_det_bboxes(
            img,
            bboxes,
            labels,
            segms=segms,
            class_names=class_names,
            score_thr=score_thr,
            bbox_color=det_bbox_color,
            text_color=det_text_color,
            mask_color=det_mask_color,
            thickness=thickness,
            font_size=font_size,
            win_name=win_name,
            show=False)
        img = np.concatenate([img_with_gt, img_with_det], axis=0)

        plt.imshow(img)
        if show:
            if wait_time == 0:
                plt.show()
            else:
                plt.show(block=False)
                plt.pause(wait_time)
        if out_file is not None:
            mmcv.imwrite(img, out_file)
        plt.close()

    return img


def show_result(self,
                img,
                result,
                score_thr=0.3,
                bbox_color=(72, 101, 241),
                text_color=(72, 101, 241),
                mask_color=None,
                thickness=2,
                font_size=13,
                win_name='',
                show=False,
                wait_time=0,
                out_file=None):
    """Draw `result` over `img`.
    
    Args:
        img (str or Tensor): The image to be displayed.
        result (Tensor or tuple): The results to draw over `img`
            bbox_result or (bbox_result, segm_result).
        score_thr (float, optional): Minimum score of bboxes to be shown.
            Default: 0.3.
        bbox_color (str or tuple(int) or :obj:`Color`):Color of bbox lines.
           The tuple of color should be in BGR order. Default: 'green'
        text_color (str or tuple(int) or :obj:`Color`):Color of texts.
           The tuple of color should be in BGR order. Default: 'green'
        mask_color (None or str or tuple(int) or :obj:`Color`):
           Color of masks. The tuple of color should be in BGR order.
           Default: None
        thickness (int): Thickness of lines. Default: 2
        font_size (int): Font size of texts. Default: 13
        win_name (str): The window name. Default: ''
        wait_time (float): Value of waitKey param.
            Default: 0.
        show (bool): Whether to show the image.
            Default: False.
        out_file (str or None): The filename to write the image.
            Default: None.
    Returns:
        img (Tensor): Only if not `show` or `out_file`
    """
    img = mmcv.imread(img)
    img = img.copy()
    if isinstance(result, tuple):
        bbox_result, segm_result = result
        if isinstance(segm_result, tuple):
            segm_result = segm_result[0]  # ms rcnn
    else:
        bbox_result, segm_result = result, None
    bboxes = np.vstack(bbox_result)
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
    ]
    labels = np.concatenate(labels)
    # draw segmentation masks
    segms = None
    if segm_result is not None and len(labels) > 0:  # non empty
        segms = mmcv.concat_list(segm_result)
        if isinstance(segms[0], torch.Tensor):
            segms = torch.stack(segms, dim=0).detach().cpu().numpy()
        else:
            segms = np.stack(segms, axis=0)
    # if out_file specified, do not show image in window
    if out_file is not None:
        show = False
    # draw bounding boxes
    img = imshow_det_bboxes(
        img,
        bboxes,
        labels,
        segms,
        class_names=self.CLASSES,
        score_thr=score_thr,
        bbox_color=bbox_color,
        text_color=text_color,
        mask_color=mask_color,
        thickness=thickness,
        font_size=font_size,
        win_name=win_name,
        show=show,
        wait_time=wait_time,
        out_file=out_file)
    if not (show or out_file):
        return img