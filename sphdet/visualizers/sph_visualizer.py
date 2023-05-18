import os
from numbers import Number

import cv2
import numpy as np
import torch

from sphdet.iou.sph2pob_standard import (compute_3d_coordinate,
                                         compute_rotate_matrix,
                                         compute_rotate_matrix_auto,
                                         compute_rotate_matrix_better,
                                         compute_spherical_coordinate,
                                         sph2pob_standard,
                                         standardize_spherical_box)

from .ImageRecoderTools import ro_Shpbbox
from .ImageRecorder import ImageRecorder as BFoV


class SphVisualizer:
    def __init__(self, canvas='dark', canvas_size=(512, 1024), with_lonlat=False):
        if isinstance(canvas, str):
            colors = {'dark':(0, 0, 0), 'light':(255, 255, 255)}
            self.canvas = np.stack([np.ones(canvas_size)*colors[canvas][i] for i in range(3)], axis=-1)
            if with_lonlat:
                line_colors = {'dark':(255, 255, 255), 'light':(150, 150, 150)}
                line_color = line_colors.get(canvas, (0, 0, 0))
                line_thickness = self._get_best_line_thickness()
                self.add_longitudes(np.arange(0, 360, 30), line_color, line_thickness)
                self.add_latitudes(np.arange(0, 180, 45), line_color, line_thickness)
        elif isinstance(canvas, tuple):
            self.canvas = np.stack([np.ones(canvas_size)*canvas[i] for i in range(3)], axis=-1)
        else:
            self.canvas = canvas

    def _ensure_list(self, items, ndim=2, dtype=None):
        if isinstance(items, Number) or isinstance(items, list) or isinstance(items, tuple):
            items = np.array(items, dtype=dtype)

        if isinstance(items, np.ndarray):
            if items.ndim < ndim:
                items = np.expand_dims(items, axis=0)
        return items
    
    def _get_best_line_thickness(self):
        return self.canvas.shape[0] // 512
        
    def add_boxes(self, boxes, colors, center=True, thickness=None, border_only=True, alpha=1.0):
        thickness = self._get_best_line_thickness() if thickness is None else thickness
        boxes = self._ensure_list(boxes)
        colors = self._ensure_list(colors, dtype=np.uint8)

        def box_transform(box):
            eps=1
            box[0] = box[0] - 180
            box[1] = 90 - box[1]
            box[2] = max(min(box[2], 180-eps), eps)
            box[3] = max(min(box[3], 180-eps), eps)
            #box[4] = -box[4]
            return box

        H, W, _ = self.canvas.shape
        for box, color in zip(boxes, colors):
            box = box_transform(box)
            bfov = BFoV(W, H, view_angle_w=box[2], view_angle_h=box[3], long_side=W)
            
            if not border_only:
                canvas = np.zeros_like(self.canvas)
                px, py = bfov._sample_points(np.deg2rad(box[0]), np.deg2rad(box[1]), border_only)
                if len(box) == 5:
                    px, py = ro_Shpbbox(box, px, py, erp_w=W, erp_h=H)
                bfov.draw_Sphbbox(canvas ,px, py, border_only, color=color.tolist(), thickness=(thickness+1)//2)
                canvas_mask = canvas != 0
                self.canvas = (canvas_mask * (alpha * canvas + (1.0 - alpha) * self.canvas) + ~canvas_mask * self.canvas).astype(np.uint8)
            px, py = bfov._sample_points(np.deg2rad(box[0]), np.deg2rad(box[1]), border_only=True)
            if len(box) == 5:
                px, py = ro_Shpbbox(box, px, py, erp_w=W, erp_h=H)
            bfov.draw_Sphbbox(self.canvas ,px, py, border_only=True, color=color.tolist(), thickness=(thickness+1)//2)

            if center:
                cv2.circle(self.canvas, (int((box[0]+180)/360*W), int((90-box[1])/180*H)), 4*thickness, color=color.tolist(), thickness=-1)

    def add_arc(self, pt1, pt2):
        raise NotImplemented()

    def add_longitudes(self, longitudes, color=(255, 255, 255), thickness=None):
        thickness = self._get_best_line_thickness() if thickness is None else thickness
        longitudes = self._ensure_list(longitudes, ndim=1)

        H, W, _ = self.canvas.shape
        for lon in longitudes:
            x = int(lon / 360 * W)
            cv2.line(self.canvas, (x, 0), (x, H-1), color, thickness)
            #cv2.addText(self.canvas, str(int(lon)), (x, H/2), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0))

    def add_latitudes(self, latitudes, color=(255, 255, 255), thickness=None):
        thickness = self._get_best_line_thickness() if thickness is None else thickness
        latitudes = self._ensure_list(latitudes, ndim=1)

        H, W, _ = self.canvas.shape
        for lon in latitudes:
            y = int(lon / 180 * H)
            cv2.line(self.canvas, (0, y), (W-1, y), color, thickness)

    def rotate_sphere(self, R, inplace=False):
        pi = torch.pi
        H, W, _ = self.canvas.shape
        x = torch.linspace(0, W-1, W)
        y = torch.linspace(0, H-1, H)
        xx_theta, yy_phi = torch.meshgrid(x, y, indexing='xy') # HxW
        xx_theta = xx_theta.flatten() / W * 2 * pi # N=HxW
        yy_phi = yy_phi.flatten() / H * pi    # N=HxW
        pp = compute_3d_coordinate(xx_theta, yy_phi) # Nx3
        pp_ = (R @ pp.T).T # Nx3
        xx_theta_, yy_phi_ = compute_spherical_coordinate(pp_[..., None])
        xx_theta_ = ((xx_theta_.view((H, W)) + 2*pi) % (2*pi)) / (2 * pi) * W
        yy_phi_ = yy_phi_.view((H, W)) / pi * H
        
        warped_canvas = cv2.remap(self.canvas, xx_theta_.numpy(), yy_phi_.numpy(), interpolation=cv2.INTER_LINEAR)
        if inplace:
            self.warped_canvas = self.canvas
            self.canvas = warped_canvas
        else:
            self.warped_canvas = warped_canvas

    def get_rotated_matrix(self, box1, box2, mode='auto'):
        box1 = torch.tensor([box1]) # 1x4
        box2 = torch.tensor([box2]) # 1x4
        #box1, box2 = standardize_spherical_box(box1, box2)
        theta1 = torch.deg2rad(box1[0, 0]).view((1, 1))
        phi1 = torch.deg2rad(box1[0, 1]).view((1, 1))
        theta2 = torch.deg2rad(box2[0, 0]).view((1, 1))
        phi2 = torch.deg2rad(box2[0, 1]).view((1, 1))
    
        if mode == 'v1':
            theta, phi = (theta1 + theta2) / 2, (phi1 + phi2) / 2
            R = compute_rotate_matrix(theta, phi)
        elif mode == 'v2':
            vec1 = compute_3d_coordinate(theta1, phi1)
            vec2 = compute_3d_coordinate(theta2, phi2)
            R = compute_rotate_matrix_better(vec1, vec2)
        else:
            theta, phi = (theta1 + theta2) / 2, (phi1 + phi2) / 2
            vec1 = compute_3d_coordinate(theta1, phi1)
            vec2 = compute_3d_coordinate(theta2, phi2)
            R = compute_rotate_matrix_auto(vec1, vec2, theta, phi)

        return R.squeeze()

    def show(self, out_path=None, display=False):
        if out_path is not None:
            out_dir = os.path.dirname(out_path)
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            #self.canvas = cv2.resize(self.canvas, (2048, 1024))
            #self.warped_canvas = cv2.resize(self.warped_canvas, (2048, 1024))
            if hasattr(self, 'warped_canvas'):
                canvas = np.concatenate([self.canvas, self.warped_canvas], axis=0)
                cv2.imwrite(out_path, canvas)
                cv2.imwrite(out_path.replace('sph_visualizer', 'sph_visualizer-a'), self.warped_canvas)
                cv2.imwrite(out_path.replace('sph_visualizer', 'sph_visualizer-b'), self.canvas)
            else:
                cv2.imwrite(out_path, self.canvas)
            print(f'Saved canvas on {out_path}')


        if display:
            cv2.imshow('SphVisualizer', self.canvas)
            cv2.waitKey()

    def _roll_canvas(self, canvas):
        W_2 = canvas.shape[1] // 2
        canvas[:, :] = np.concatenate([canvas[:, W_2:], canvas[:, :W_2]], axis=1)

    def _add_obbs(self, boxes, colors, center=True, thickness=None):
        thickness = 2 * (self._get_best_line_thickness() if thickness is None else thickness)
        boxes = self._ensure_list(boxes)
        colors = self._ensure_list(colors, dtype=np.uint8)

        H, W, _ = self.canvas.shape
        pi = torch.pi

        self._roll_canvas(self.canvas)

        for box, color in zip(boxes, colors):
            xc, yc, w, h, ag = box

            xc = xc / (2*pi) * W + W // 2
            yc = yc / pi * H
            w  = w / (2*pi) * W
            h  = h / pi * H

            wx, wy = w / 2 * np.cos(ag), w / 2 * np.sin(ag)
            hx, hy = -h / 2 * np.sin(ag), h / 2 * np.cos(ag)
            p1 = (int(xc - wx - hx), int(yc - wy - hy))
            p2 = (int(xc + wx - hx), int(yc + wy - hy))
            p3 = (int(xc + wx + hx), int(yc + wy + hy))
            p4 = (int(xc - wx + hx), int(yc - wy + hy))
            pts = np.array([[p1, p2, p3, p4]], dtype=np.int32)
            
            cv2.polylines(self.canvas, pts, True, color.tolist(), thickness, lineType=cv2.LINE_AA)
            if center:
                cv2.circle(self.canvas, (int(xc), int(yc)), 8, color.tolist(), thickness=-1)
        
        self._roll_canvas(self.canvas)

    def quick_test(self, box1, box2, mode='v2', inplace=True, with_lonlat=True):
        self.add_longitudes(180, color=(0, 255, 0))
        self.add_latitudes(90, color=(0, 0, 255))

        colors = [(130, 101, 245), (255, 140, 98)] # blue, yellow
        self.add_boxes([box1, box2] , colors, border_only=False, alpha=0.5)
        R = self.get_rotated_matrix(box1, box2, mode)
        self.rotate_sphere(R.T, inplace)
        if with_lonlat and inplace:
            self.add_longitudes(np.arange(0, 360, 30), color=(80, 114, 181))
            self.add_latitudes(np.arange(0, 180, 30), color=(80, 114, 181))

        from sphdet.iou.sph_iou_api import sph2pob_standard_iou, unbiased_iou
        bbox1 = torch.tensor(box1).float().unsqueeze(0)
        bbox2 = torch.tensor(box2).float().unsqueeze(0)
        iou1 = unbiased_iou(bbox1, bbox2, is_aligned=True)
        iou2 = sph2pob_standard_iou(bbox1, bbox2, is_aligned=True, calculator='common', rbb_edge='arc')
        print(f'box1={box1}, box2={box2} | iou1={iou1.item():4f}, iou2={iou2.item():4f} | err={(iou1-iou2).item():4f}\n')
        
        rbb1, rbb2 = sph2pob_standard(bbox1, bbox2, rbb_angle_version='rad', rbb_edge='arc')
        self._add_obbs(torch.concat([rbb1, rbb2]), colors)

        # ---------------------------------------------------------------------------- #
        rbb1, rbb2 = sph2pob_standard(bbox1, bbox2, rbb_angle_version='rad', rbb_edge='arc')
        print(rbb1, rbb2)
        #self._add_obbs(torch.concat([rbb1, rbb2]), [(171, 214, 144), (171, 214, 144)], center=False, thickness=thickness-2)

        from mmcv.ops import box_iou_rotated
        riou = box_iou_rotated(rbb1, rbb2, aligned=True, clockwise=True)
        rbb1 = torch.rad2deg(rbb1).flatten().tolist()
        rbb2 = torch.rad2deg(rbb2).flatten().tolist()
        print(f'rbb1={rbb1}, rbb2={rbb2} | riou={riou.item():4f}')
        iou2 = sph2pob_standard_iou(bbox1, bbox2, is_aligned=True, calculator='common', rbb_edge='arc')
        print(f'box1={box1}, box2={box2} | iou1={iou1.item():4f}, iou2={iou2.item():4f} | err={(iou1-iou2).item():4f}\n')

        # ---------------------------------------------------------------------------- #
        rbb1, rbb2 = sph2pob_standard(bbox1, bbox2, rbb_angle_version='rad', rbb_edge='tangent')
        print(rbb1, rbb2)
        #self._add_obbs(torch.concat([rbb1, rbb2]), [(248, 107,142), (248, 107,142)], center=False, thickness=thickness-2)

        from mmcv.ops import box_iou_rotated
        riou = box_iou_rotated(rbb1, rbb2, aligned=True, clockwise=True)
        rbb1 = torch.rad2deg(rbb1).flatten().tolist()
        rbb2 = torch.rad2deg(rbb2).flatten().tolist()
        print(f'rbb1={rbb1}, rbb2={rbb2} | riou={riou.item():4f}')
        iou2 = sph2pob_standard_iou(bbox1, bbox2, is_aligned=True, calculator='common', rbb_edge='tangent')
        print(f'box1={box1}, box2={box2} | iou1={iou1.item():4f}, iou2={iou2.item():4f} | err={(iou1-iou2).item():4f}\n')

        # ---------------------------------------------------------------------------- #
        rbb1, rbb2 = sph2pob_standard(bbox1, bbox2, rbb_angle_version='rad', rbb_edge='chord')
        print(rbb1, rbb2)
        #self._add_obbs(torch.concat([rbb1, rbb2]), [(91, 197, 253), (91, 197, 253)], center=False, thickness=thickness-2)

        from mmcv.ops import box_iou_rotated
        riou = box_iou_rotated(rbb1, rbb2, aligned=True, clockwise=True)
        rbb1 = torch.rad2deg(rbb1).flatten().tolist()
        rbb2 = torch.rad2deg(rbb2).flatten().tolist()
        print(f'rbb1={rbb1}, rbb2={rbb2} | riou={riou.item():4f}')
        iou2 = sph2pob_standard_iou(bbox1, bbox2, is_aligned=True, calculator='common', rbb_edge='chord')
        print(f'box1={box1}, box2={box2} | iou1={iou1.item():4f}, iou2={iou2.item():4f} | err={(iou1-iou2).item():4f}\n')

        # ---------------------------------------------------------------------------- #
        self._roll_canvas(self.canvas)