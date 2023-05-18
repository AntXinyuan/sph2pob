import math
import torch


def xyxy2xywh(boxes):
    x1, y1, x2, y2 = torch.chunk(boxes, 4, dim=1) # Nx1
    x = (x1 + x2) / 2
    y = (y1 + y2) / 2
    w = x2 - x1
    h = (y2 - y1)
    return torch.cat([x, y, w, h], dim=1) # Nx4

def xywh2xyxy(boxes):
    x, y, w, h = torch.chunk(boxes, 4, dim=1) # Nx1
    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2
    return torch.cat([x1, y1, x2, y2], dim=1) # Nx4

def obb2hbb_wywh(obb):
    w = obb[:, 2]
    h = obb[:, 3]
    a = obb[:, 4]#torch.abs(obb[:, 4])
    cosa = torch.cos(a).abs()
    sina = torch.sin(a).abs()
    hbbox_w = cosa * w + sina * h
    hbbox_h = sina * w + cosa * h
    cx = obb[..., 0]
    cy = obb[..., 1]
    # dw = hbbox_w.reshape(-1)
    # dh = hbbox_h.reshape(-1)
    # x1 = dx - dw / 2
    # y1 = dy - dh / 2
    # x2 = dx + dw / 2
    # y2 = dy + dh / 2
    return torch.stack((cx, cy, hbbox_w, hbbox_h), -1)

def obb2hbb_xyxy(obb):
    hbb = obb2hbb_wywh(obb)
    hbb = xywh2xyxy(hbb)
    return hbb

def bfov2rbfov(bfovs):
    N = bfovs.size(0)
    zero = torch.zeros((N, 1)).to(bfovs)
    rbfovs = torch.concat([bfovs, zero], dim=1)
    return rbfovs

# ---------------------------------------------------------------------------- #
def geo2sph(boxes):
    bboxes = boxes.clone()
    bboxes[..., 0] = boxes[..., 0] + 180
    bboxes[..., 1] = 90 - boxes[..., 1]
    return bboxes

def sph2geo(boxes):
    bboxes = boxes.clone()
    bboxes[..., 0] = boxes[..., 0] - 180
    bboxes[..., 1] = 90 - boxes[..., 1]
    return bboxes

# ---------------------------------------------------------------------------- #
def _sph2pix_box_transform(boxes, img_size):
    img_h, img_w = img_size
    theta, phi, alpha, beta = torch.chunk(boxes, 4, dim=1) # Nx1
    x = (theta / 360) * img_w
    y = (phi / 180) * img_h
    w = (alpha / 360) * img_w
    h = (beta / 180) * img_h
    return torch.cat([x, y, w, h], dim=1) # Nx4

def _pix2sph_box_transform(boxes, img_size):
    img_h, img_w = img_size
    x, y, w, h = torch.chunk(boxes, 4, dim=1) # Nx1
    theta = (x / img_w) * 360
    phi =(y / img_h) * 180
    alpha = (w / img_w) * 360
    beta = (h / img_h) * 180
    return torch.cat([theta, phi, alpha, beta], dim=1) # Nx4

def _sph2tan_box_transform(boxes, img_size):
    img_h, img_w = img_size
    theta, phi, alpha, beta = torch.chunk(boxes, 4, dim=1) # Nx1
    _2R = img_w / math.pi
    x = (theta / 360) * img_w
    y = (phi / 180) * img_h
    w = _2R * torch.tan(torch.deg2rad(alpha) / 2) 
    h = _2R * torch.tan(torch.deg2rad(beta) / 2)
    return torch.cat([x, y, w, h], dim=1) # Nx4

def _tan2sph_box_transform(boxes, img_size):
    img_h, img_w = img_size
    x, y, w, h = torch.chunk(boxes, 4, dim=1) # Nx1
    _2R = img_w / math.pi
    theta = (x / img_w) * 360
    phi =(y / img_h) * 180
    alpha = torch.rad2deg(2 * torch.atan(w / _2R))
    beta = torch.rad2deg(2 * torch.atan(h / _2R))
    return torch.cat([theta, phi, alpha, beta], dim=1) # Nx4

# ---------------------------------------------------------------------------- #
def is_valid_boxes(boxes, mode='sph', need_raise=False):
    try:
        if mode == 'sph':
            assert boxes.size(-1) in [4, 5]
            theta, phi, alpha, beta = torch.chunk(boxes[:, :4], 4, dim=-1)
            assert theta.min() >= 0 and theta.max() <= 360
            assert phi.min()   >= 0 and phi.max()   <= 180
            assert alpha.min() >= 0 and alpha.max() <= 360
            assert beta.min()  >= 0 and beta.max()  <= 180
        elif mode == 'obb':
            import math
            pi = math.pi
            half_pi = pi / 2.0
            assert boxes.size(-1) == 5
            x, y, w, h, a = torch.chunk(boxes, 5, dim=-1)
            assert w.min() >= 0 and w.max() <= pi
            assert h.min() >= 0 and h.max() <= pi
            assert a.min() >= -half_pi and a.max() <= half_pi
    except AssertionError as e:
        if need_raise:
            raise e
        return False
    else:
        return True 

# ---------------------------------------------------------------------------- #

def climp_rotated_boxes(bboxes1,bboxes2, eps=1e-7):
    pi = torch.pi
    bboxes1[:, 2:4].clamp_(min=2*eps/10)
    bboxes2[:, 2:4].clamp_(min=eps/10)
    bboxes1[:, 4].clamp_(min=-2*pi+2*eps, max=2*pi-eps)
    bboxes2[:, 4].clamp_(min=-2*pi+eps, max=2*pi-2*eps)
    return bboxes1,bboxes2

def climp_spherical_boxes(bboxes1,bboxes2, eps=1e-7):
    pi = 180
    torch.clamp_(bboxes1[:, 0], 2*eps, 2*pi-eps)
    torch.clamp_(bboxes1[:, 1:4], 2*eps, pi-eps)
    torch.clamp_(bboxes2[:, 0], eps, 2*pi-2*eps)
    torch.clamp_(bboxes2[:, 1:4], eps, pi-2*eps)
    if bboxes1.size(1) == 5:
        torch.clamp_(bboxes2[:, 4], -2*pi+eps, max=2*pi-2*eps)
        torch.clamp_(bboxes2[:, 4], -2*pi+2*eps, max=2*pi-eps)
    return bboxes1,bboxes2
  
# ---------------------------------------------------------------------------- #
class Sph2PlanarBoxTransform:
    def __init__(self, mode='sph2pix', box_version=4):
        assert mode in ['sph2pix', 'sph2tan']
        assert box_version in [4, 5]

        self.box_version = box_version
        self.transform = _sph2pix_box_transform if mode == 'sph2pix' else _sph2tan_box_transform
    
    def __call__(self, boxes, img_size=(512, 1024), box_version=None):
        box_version = self.box_version if box_version is None else box_version
        if box_version == 4:
            _boxes = xywh2xyxy(self.transform(boxes, img_size))
            return _boxes #xyxy
        else:
            _boxes, _angles = boxes[:, :4], boxes[:, [4,]]
            _boxes = self.transform(_boxes, img_size) #xywh
            _angles = torch.deg2rad(_angles) #a
            return torch.concat([_boxes, -_angles], dim=-1) #xywha


class Planar2SphBoxTransform:
    def __init__(self, mode='sph2pix', box_version=4):
        assert mode in ['sph2pix', 'pix2sph', 'sph2tan', 'tan2sph']
        assert box_version in [4, 5]

        self.box_version = box_version
        self.transform = _pix2sph_box_transform if mode in ['sph2pix', 'pix2sph'] else _tan2sph_box_transform
    
    def __call__(self, boxes, img_size=(512, 1024), box_version=None):
        box_version = self.box_version if box_version is None else box_version
        if box_version == 4:
            return self.transform(xyxy2xywh(boxes), img_size) #xywh(bfov)
        else:
            _boxes = self.transform(xyxy2xywh(boxes), img_size)
            return bfov2rbfov(_boxes) #xywha(rbfov)


def bbox2roi(bbox_list, box_version=4):
    """Convert a list of bboxes to roi format.

    Args:
        bbox_list (list[Tensor]): a list of bboxes corresponding to a batch
            of images.

    Returns:
        Tensor: shape (n, 5), [batch_ind, x1, y1, x2, y2]
    """
    rois_list = []
    for img_id, bboxes in enumerate(bbox_list):
        if bboxes.size(0) > 0:
            img_inds = bboxes.new_full((bboxes.size(0), 1), img_id)
            rois = torch.cat([img_inds, bboxes[:, :box_version]], dim=-1)
        else:
            rois = bboxes.new_zeros((0, box_version+1))
        rois_list.append(rois)
    rois = torch.cat(rois_list, 0)
    return rois
     