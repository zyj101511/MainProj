import os

import cv2
import numpy as np
import torch

from lib.models.mistrack_plain_CA import build_model
from lib.test.data.utils.preprocessing_utils import crop_template, sample_target, _map_bbox_to_original
from lib.test.tracker.basetracker import BaseTracker
from lib.utils.box_ops import clip_box_tensor


class MISTracker(BaseTracker):
    def __init__(self, settings):
        super().__init__(settings)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = build_model(self.cfg, training=False, load_ckpt=True).to(self.device)
        self.model.eval()
        self.state = None  # 记录当前帧的预测状态，包括边界框和其他信息
        self.search_box_state = None  # 记录当前帧搜索框裁切的参考状态

        # output feature map size of the backbone and multi-timescale module
        self.feat_sz = self.cfg.DATA.SEARCH.SIZE // self.cfg.MODEL.BACKBONE.STRIDE

        # for visualization using visdom
        self.debug = self.cfg.TEST.DEBUG
        self.save_plot = self.cfg.TEST.SAVE_PLOT
        self.frame_id = 0

        if self.save_plot:
            self.save_dir = self.settings.env.result_plot_path
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)

        if self.debug > 0:
            self._init_visdom(None, self.debug)

        self.template = None  # 模板HWC

    def initialize(self, image, init_gt_bbox: torch.Tensor):
        """
        image: TCHW tensor
        T == 1, image is the first frame
        T > 1, image is the last frame in the window T
        """
        image = image.to(self.device)
        init_gt_bbox = init_gt_bbox
        assert type(init_gt_bbox) == torch.Tensor

        template_patch = crop_template(
            image,
            init_gt_bbox,
            self.settings.cfg.TEST.TEMPLATE_SCALE_FACTOR,
            reshape_sz=self.settings.cfg.DATA.TEMPLATE.SIZE,
        )

        self.template = template_patch # (T, C, H, W)
        self.state = init_gt_bbox
        self.search_box_state = init_gt_bbox
        self.frame_id = 0
        return {"pred_bbox": self.state}

    def track(self, image, gt_bbox: torch.Tensor):
        image = image.to(self.device)
        """image (T,C,H,W)"""
        T, C, H, W = image.shape
        self.frame_id += 1
        search_patch, resize_factor, search_box = sample_target(
            image,
            self.state,
            self.settings.cfg.TEST.SEARCH_SCALE_FACTOR,
            self.settings.cfg.DATA.SEARCH.SIZE
        )
        with torch.no_grad():
            '''pred_dict = {'pred_boxes': pred_box, # (B, 4) (cx, cy, w, h) normalized
                            'score_map': score_map_ctr, # (B, 1, H, W)
                            'offset_map': offset_map, # (B, 2, H, W)
                            'size_map' : size_map, # (B, 2, H, W)
                           }'''


            pred_dict = self.model(search=search_patch.unsqueeze(1), template=self.template.unsqueeze(1))

            fused_feature = pred_dict['fused_feature'][-1].mean(dim=0)  # (H, W)

            normalized_pred_box = pred_dict['pred_boxes'][-1]  # (4) (cx, cy, w, h)
            pred_box_original = _map_bbox_to_original(normalized_pred_box, self.settings.cfg.DATA.SEARCH.SIZE, search_box, resize_factor)
            self.state = clip_box_tensor(pred_box_original, H, W, margin=self.settings.cfg.TEST.CLIP_BOX_MARGIN)  # (x, y, w, h) in original image coordinates


        if self.debug > 0:
            cv_image = (image[-1].detach().cpu().numpy().transpose(1, 2, 0) * 255).clip(0,255).astype(np.uint8)
            cv_search = (search_patch[-1].detach().cpu().numpy().transpose(1, 2, 0) * 255).clip(0,255).astype(np.uint8)
            cv_template = (self.template[-1].detach().cpu().numpy().transpose(1, 2, 0) * 255).clip(0,255).astype(np.uint8)
            pred_score_map = pred_dict['score_map'][-1].detach().cpu()  # (1, H, W)
            fused_layer_feature = fused_feature.detach().cpu()  # (H, W)
            cv_image = cv_image[:, :, ::-1].copy()
            cv_search = cv_search[:, :, ::-1].copy()
            cv_template = cv_template[:, :, ::-1].copy()

            self.visdom.register((cv_image, gt_bbox.tolist(), self.state.tolist()), 'Tracking', 1, 'Tracking')
            self.visdom.register(torch.from_numpy(cv_search).permute(2, 0, 1), 'image', 1, 'search_region')
            self.visdom.register(torch.from_numpy(cv_template).permute(2, 0, 1), 'image', 1, 'template')
            self.visdom.register(pred_score_map.view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'score_map')
            self.visdom.register(fused_layer_feature.view(self.feat_sz, self.feat_sz), 'heatmap', 2, 'fused_feature')

            while self.pause_mode:
                if self.step:
                    self.step = False
                    break
        if self.save_plot:
            x, y, w, h = self.state.tolist()
            cv_image = image[-1].detach().cpu().numpy().transpose(1, 2, 0)  # HWC
            cv2.rectangle(cv_image, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 1)
            write_img = cv_image[:, :, ::-1]
            save_path = os.path.join(self.save_dir, f'{self.settings.cur_seq_name}/frame_{self.frame_id:06d}.jpg')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cv2.imwrite(save_path, write_img)

        return {"pred_bbox": self.state}


    def track_full_image(self, image, gt_bbox: torch.Tensor):
        image = image.to(self.device)
        """image (T,C,H,W)"""
        T, C, H, W = image.shape
        self.frame_id += 1
        search_box = torch.tensor([0, 0, W, H], dtype=torch.float32, device=image.device)
        search_patch, resize_factor, search_box = self._sample_target_like_train(
            image,
            search_box,
            self.settings.cfg.TEST.SEARCH_SCALE_FACTOR,
            self.settings.cfg.DATA.SEARCH.SIZE,
        )
        with torch.no_grad():
            '''pred_dict = {'pred_boxes': pred_box, # (B, 4) (cx, cy, w, h) normalized
                            'score_map': score_map_ctr, # (B, 1, H, W)
                            'offset_map': offset_map, # (B, 2, H, W)
                            'size_map' : size_map, # (B, 2, H, W)
                            'near_future_ctr': near_future_ctr, # (B, P, 2)
                            'cur_v': cur_v, # (B, 2)
                            'cur_a': cur_a, # (B, 2)
                            'a_deltas': a_deltas, # (B, df-1, 2)
                            'track_idx': idx
                           }'''
            pred_dict = self.model(search=search_patch.unsqueeze(1), template=self.template.unsqueeze(1))

            fused_feature = pred_dict['fused_feature'][-1].mean(dim=0)  # (H, W)

            normalized_pred_box = pred_dict['pred_boxes'][-1]  # (4) (cx, cy, w, h)
            pred_box_original = _map_bbox_to_original(normalized_pred_box, self.settings.cfg.DATA.SEARCH.SIZE, search_box, resize_factor)
            self.state = clip_box_tensor(pred_box_original, H, W, margin=self.settings.cfg.TEST.CLIP_BOX_MARGIN)  # (x, y, w, h) in original image coordinates


        if self.debug > 0:
            cv_image = (image[-1].detach().cpu().numpy().transpose(1, 2, 0) * 255).clip(0,255).astype(np.uint8)
            cv_search = (search_patch[-1].detach().cpu().numpy().transpose(1, 2, 0) * 255).clip(0,255).astype(np.uint8)
            cv_template = (self.template[-1].detach().cpu().numpy().transpose(1, 2, 0) * 255).clip(0,255).astype(np.uint8)
            pred_score_map = pred_dict['score_map'][-1].detach().cpu()  # (1, H, W)
            fused_layer_feature = fused_feature.detach().cpu()  # (H, W)
            cv_image = cv_image[:, :, ::-1].copy()
            cv_search = cv_search[:, :, ::-1].copy()
            cv_template = cv_template[:, :, ::-1].copy()
            self.visdom.register((cv_image, gt_bbox.tolist(), self.state.tolist()), 'Tracking', 1, 'Tracking')
            self.visdom.register(torch.from_numpy(cv_search).permute(2, 0, 1), 'image', 1, 'search_region')
            self.visdom.register(torch.from_numpy(cv_template).permute(2, 0, 1), 'image', 1, 'template')
            self.visdom.register(pred_score_map.view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'score_map')
            self.visdom.register(fused_layer_feature.view(self.feat_sz, self.feat_sz), 'heatmap', 2, 'fused_feature')

            while self.pause_mode:
                if self.step:
                    self.step = False
                    break
        if self.save_plot:
            x, y, w, h = self.state.tolist()
            cv_image = image[-1].detach().cpu().numpy().transpose(1, 2, 0)  # HWC
            cv2.rectangle(cv_image, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 1)
            write_img = cv_image[:, :, ::-1]
            save_path = os.path.join(self.save_dir, f'{self.settings.cur_seq_name}/frame_{self.frame_id:06d}.jpg')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cv2.imwrite(save_path, write_img)

        return {"pred_bbox": self.state}


    def track_and_trajectory(self, image, gt_bbox: torch.Tensor):
        raise NotImplementedError("This method is not implemented yet. Please use the 'track' method instead.")

    def track_and_trajectory_full_image(self, image, gt_bbox: torch.Tensor):
        raise NotImplementedError("This method is not implemented yet. Please use the 'track_full_image' method instead.")

    def _need_update_search_box(self, pred_box, search_box, margin_ratio=0.1):
        x, y, w, h = pred_box.tolist()
        sx1, sy1, sx2, sy2 = search_box.tolist()

        x1 = x
        y1 = y
        x2 = x + w
        y2 = y + h

        sw = sx2 - sx1
        sh = sy2 - sy1
        mx = sw * margin_ratio
        my = sh * margin_ratio

        return x1 < sx1 + mx or y1 < sy1 + my or x2 > sx2 - mx or y2 > sy2 - my



    def _map_bbox_to_original(self, normalized_bbox, search_size, search_box, H_scaler, W_scaler):
        """
        normalized_bbox: (4) (cx, cy, w, h) normalized
        search_size: reshape size of the input search image
        search_box: (4) (x1, y1, x2, y2) in original image coordinates
        H_scaler: (float) the ratio of the height of the search box to the height of the input search image
        W_scaler: (float) the ratio of the width of the search box to the width of the input search image
        """
        cx = normalized_bbox[0] * search_size
        cy = normalized_bbox[1] * search_size
        w = normalized_bbox[2] * search_size
        h = normalized_bbox[3] * search_size

        # map to search box coordinates
        cx = cx * W_scaler + search_box[0]
        cy = cy * H_scaler + search_box[1]
        w = w * W_scaler
        h = h * H_scaler

        mapped_bbox = torch.stack([cx, cy, w, h], dim=-1)
        return mapped_bbox



























