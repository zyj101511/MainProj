import torch
import warnings
from lib.train.actor.base_actor import BaseActor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
from lib.utils.heapmap_utils import generate_heatmap

class SDTrackActor(BaseActor):
    """ Actor for training SDTrack models """
    def __init__(self, net, objective, loss_weight, settings, cfg):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = settings.batch_size
        self.cfg = cfg

    def __call__(self, data_dict):
        """
        args:
            data - The input data, should contain the fields 'template_images', 'search_images', 'search_anno'.
            template_images: (N_t, T, batch, 3, H, W)
            search_images: (N_s, T, batch, 3, H, W)
            search_anno: (N_s, B, 4), 4:(x, y, w, h)
        returns:
            total_loss    - the total training loss
            detailed_loss  -  dict containing detailed losses
        """
        # forward pass
        net_out_dict = self.forward_pass(data_dict)

        # compute_loss
        total_loss, detailed_loss = self.compute_loss(net_out_dict, data_dict)

        return total_loss, detailed_loss

    def forward_pass(self, data):
        assert len(data['template_images']) == 1
        assert len(data['search_images']) == 1

        search_img = data['search_images'][0]  # (T, B, C, H, W)
        template_img = data['template_images'][0]  # (T, B, C, H, W)
        net_out_dict = self.net(search=search_img, template=template_img, return_last=True, return_max_score=False)
        return net_out_dict

    def compute_loss(self, net_out_dict, data_dict, return_status=True):
        # gt gaussian map
        gt_bbox = data_dict['search_anno'][-1]  # (Ns, B, 4) -> (B, 4)
        gt_gaussion_maps = generate_heatmap(data_dict['search_anno'],
                                            self.cfg.DATA.SEARCH.SIZE,
                                            self.cfg.MODEL.BACKBONE.STRIDE)  #  # list of length N, each elem is (B, H, W)
        gt_gaussion_maps = gt_gaussion_maps[-1].unsqueeze(1)  # (B, 1, H, W)

        # get pred boxes
        pred_boxes = net_out_dict['pred_boxes']  # (B, 4)
        if torch.isnan(pred_boxes).any():
            raise ValueError("Network outputs is NAN! Stop Training")
        pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes)  # (B, 4)
        gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)  # (B, 4)

        # compute giou and iou
        device = pred_boxes.device
        try:
            giou_loss, iou = self.objective['giou_loss'](pred_boxes_vec,gt_boxes_vec)
        except:
            warnings.warn("bad value when computing giou_loss and iou, using 0.0 instead", RuntimeWarning)
            giou_loss = torch.tensor(0.0, device=device)
            iou = torch.tensor(0.0, device=device)

        # compute l1 loss
        l1_loss = self.objective['l1_loss'](pred_boxes_vec, gt_boxes_vec)

        # compute location loss
        if 'score_map' in net_out_dict:
            score_map = net_out_dict['score_map']
            gt_gaussion_maps = gt_gaussion_maps.to(score_map.device)
            location_loss = self.objective['focal_loss'](score_map, gt_gaussion_maps)
        else:
            warnings.warn("bad value when computing location_loss, using 0.0 instead", RuntimeWarning)
            location_loss = torch.tensor(0.0, device=device)

        # weighted sum loss
        total_loss = (self.loss_weight['giou_loss'] * giou_loss +
                self.loss_weight['l1_loss'] * l1_loss +
                self.loss_weight['focal_loss'] * location_loss)

        if return_status:
            mean_iou = iou.detach().mean()
            detailed_loss = {"Loss/total": total_loss.item(),
                      "Loss/giou": giou_loss.item(),
                      "Loss/l1": l1_loss.item(),
                      "Loss/location": location_loss.item(),
                      "IoU": mean_iou.item()}
            return total_loss, detailed_loss
        else:
            return total_loss