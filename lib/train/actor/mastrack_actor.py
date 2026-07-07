import torch
import warnings
from lib.train.actor.base_actor import BaseActor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy, box_xywh_to_cxcywh
from lib.utils.heapmap_utils import generate_heatmap

class MASTrackActor(BaseActor):
    """ Actor for training MASTrack models """
    def __init__(self, net, objective, loss_weight, cfg):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.bs = cfg.TRAIN.BATCH_SIZE
        self.cfg = cfg

    def __call__(self, data_tuple):
        """
        args:
            data_tuple: (search, template, search_anno)
            search: (T, B, C, H, W)
            template: (T, B, C, H, W)
            search_anno: (B, 1+P*df, 4) 4:(x, y, w, h)
        returns:
            total_loss    - the total training loss
            detailed_loss  -  dict containing detailed losses
        """
        search, template, search_anno = data_tuple
        # forward pass
        net_out_dict = self.forward_pass(search=search, template=template)

        # compute_loss
        total_loss, detailed_loss = self.compute_loss(net_out_dict, search_anno)

        return total_loss, detailed_loss

    def forward_pass(self, search, template):
        # template, search: (T, B, C, H, W)
        net_out_dict = self.net(search=search, template=template)
        return net_out_dict

    def compute_loss(self, net_out_dict, search_anno):
        tracking_loss, tracking_detailed_loss = self._compute_tracking_loss(net_out_dict, search_anno)
        trajectory_loss, trajectory_detailed_loss = self._compute_trajectory_loss(net_out_dict, search_anno)

        total_loss = (self.loss_weight['tracking']['total'] * tracking_loss +
                      self.loss_weight['trajectory']['total'] * trajectory_loss)

        detailed_loss = {}
        detailed_loss.update(tracking_detailed_loss)
        detailed_loss.update(trajectory_detailed_loss)

        return total_loss, detailed_loss

    def _compute_tracking_loss(self, net_out_dict, search_anno):
        # gt gaussian map
        gt_bbox = search_anno[:, 0, :]  # (B, 1+P*df,4) -> ( B, 4), only use the first frame's bbox for tracking loss computation
        gt_bbox_norm = gt_bbox.clone()
        gt_bbox_norm[..., 0::2] = gt_bbox_norm[..., 0::2] / self.cfg.DATA.SEARCH.SIZE
        gt_bbox_norm[..., 1::2] = gt_bbox_norm[..., 1::2] / self.cfg.DATA.SEARCH.SIZE

        gt_gaussion_maps = generate_heatmap(
            gt_bbox_norm.unsqueeze(0),
            self.cfg.DATA.SEARCH.SIZE,
            self.cfg.MODEL.BACKBONE.STRIDE
        )#  # list of length N, each elem is (B, H, W)

        gt_gaussion_maps = gt_gaussion_maps[-1].unsqueeze(1)  # (B, 1, H, W), 符合focal loss输入

        # get pred boxes
        pred_boxes = net_out_dict['pred_boxes']  # (B, 4)
        if torch.isnan(pred_boxes).any():
            raise ValueError("Network outputs is NAN! Stop Training")
        pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes) # (B, 4)
        gt_boxes_vec = box_xywh_to_xyxy(gt_bbox_norm)  # (B, 4)

        # compute giou and iou
        device = pred_boxes.device
        try:
            giou_loss, iou = self.objective['giou_loss'](pred_boxes_vec,gt_boxes_vec)
        except:
            warnings.warn("bad value when computing giou_loss and iou, using 0.0 instead", RuntimeWarning)
            giou_loss = torch.tensor(0.0, device=device)
            iou = torch.tensor(0.0, device=device)

        # compute l1 loss
        l1_loss = self.objective['tracking_l1_loss'](pred_boxes_vec, gt_boxes_vec)

        # compute location loss
        if 'score_map' in net_out_dict:
            score_map = net_out_dict['score_map']
            gt_gaussion_maps = gt_gaussion_maps.to(score_map.device)
            location_loss = self.objective['focal_loss'](score_map, gt_gaussion_maps)
        else:
            warnings.warn("bad value when computing location_loss, using 0.0 instead", RuntimeWarning)
            location_loss = torch.tensor(0.0, device=device)

        # weighted sum loss
        total_loss = (self.loss_weight['tracking']['giou_loss'] * giou_loss +
                self.loss_weight['tracking']['l1_loss'] * l1_loss +
                self.loss_weight['tracking']['focal_loss'] * location_loss)

        mean_iou = iou.detach().mean()
        detailed_loss = {"Tracking_Loss/total": total_loss.item(),
                         "Tracking_Loss/giou": giou_loss.item(),
                         "Tracking_Loss/l1": l1_loss.item(),
                         "Tracking_Loss/location": location_loss.item(),
                         "Tracking_IoU": mean_iou.item()}
        return total_loss, detailed_loss

    def _recursive_compute_distance_pred(self, gt_cur_ctr, pred_cur_v, pred_cur_a, pred_a_deltas):

        pred_distant_future_ctr = []

        c = gt_cur_ctr
        v = pred_cur_v
        a = pred_cur_a

        # 1st segment
        c = c + v*self.cfg.MODEL.HEAD.P + a*self.cfg.MODEL.HEAD.P**2/2  # (B, 2)
        v = v + a*self.cfg.MODEL.HEAD.P  # (B, 2)

        for i in range(pred_a_deltas.size(1)):
            a_delta = pred_a_deltas[:, i, :]  # (B, 2)
            a = a + a_delta

            c = c + v * self.cfg.MODEL.HEAD.P + a * self.cfg.MODEL.HEAD.P ** 2 / 2  # (B, 2)
            v = v + a * self.cfg.MODEL.HEAD.P  # (B, 2)
            pred_distant_future_ctr.append(c)

        return torch.stack(pred_distant_future_ctr, dim=1)  # (B, df-1, 2) 最后df-1个拐点



    def _compute_trajectory_loss(self, net_out_dict, search_anno):
        pred_near_future_ctr = net_out_dict['near_future_ctr']  # (B, P, 2)
        gt_near_future_anno = search_anno[:, 1:self.cfg.MODEL.HEAD.P+1, :]  # (B, P, 4)
        search_size = self.cfg.DATA.SEARCH.SIZE
        gt_near_future_ctr = box_xywh_to_cxcywh(gt_near_future_anno)[..., :2] / search_size  # (B, P, 2)

        gt_distant_future_anno = search_anno[:, self.cfg.MODEL.HEAD.P+1::self.cfg.MODEL.HEAD.P, :]  # (B, df-1, 4)
        gt_distant_future_ctr = box_xywh_to_cxcywh(gt_distant_future_anno)[..., :2] / search_size  # (B, df - 1, 2)
        gt_cur_ctr = box_xywh_to_cxcywh(search_anno[:, 0, :])[..., :2] / search_size  # (B, 2)

        pred_cur_v = net_out_dict['cur_v']  # (B, 2)
        pred_cur_a = net_out_dict['cur_a']  # (B, 2)
        pred_a_deltas = net_out_dict['a_deltas']  # (B, df-1, 2)
        pred_distant_future_ctr = self._recursive_compute_distance_pred(gt_cur_ctr, pred_cur_v, pred_cur_a, pred_a_deltas)

        # compute near future loss
        near_future_loss = self.objective['trajectory_l1_loss'](pred_near_future_ctr, gt_near_future_ctr).mean()

        # compute distant future loss
        # 'trajectory_l1_loss'必须是reduction='none'
        distant_future_loss = self.objective['trajectory_l1_loss'](pred_distant_future_ctr, gt_distant_future_ctr).mean(dim=-1)  # (B, df-1)
        decay_factor = self.loss_weight['trajectory']['decay_factor']
        weight = torch.tensor([1 * decay_factor ** i for i in range(distant_future_loss.size(1))],
                              device=distant_future_loss.device,
                              dtype=distant_future_loss.dtype).unsqueeze(0)  # (1, df-1)
        distant_future_loss = (distant_future_loss * weight).sum(dim=1) / weight.sum()
        distant_future_loss = distant_future_loss.mean()

        # weighted sum loss
        total_loss = (self.loss_weight['trajectory']['near_future_loss'] * near_future_loss
                      + self.loss_weight['trajectory']['distant_future_loss'] * distant_future_loss)

        detailed_loss = {"Trajectory_Loss/total": total_loss.item(),
                         "Trajectory_Loss/near_future_loss": near_future_loss.item(),
                         "Trajectory_Loss/distant_future_loss": distant_future_loss.item()}
        return total_loss, detailed_loss


if __name__ == '__main__':
    import torch
    import torch.nn as nn
    from types import SimpleNamespace

    class DummyGIoULoss(nn.Module):
        def forward(self, pred, target):
            l1 = torch.abs(pred - target).mean()
            iou = torch.rand(pred.shape[0], device=pred.device)
            return l1, iou


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    P = 4
    df = 4
    search_size = 256
    stride = 16
    feat_sz = search_size // stride
    B = 2
    T = 1
    C = 3
    H = search_size
    W = search_size

    objective = {
        'giou_loss': DummyGIoULoss(),
        'tracking_l1_loss': nn.L1Loss(),
        'focal_loss': nn.MSELoss(),
        'trajectory_l1_loss': nn.L1Loss(reduction='none'),
    }

    loss_weight = {
        'tracking': {
            'total': 1.0,
            'giou_loss': 1.0,
            'l1_loss': 1.0,
            'focal_loss': 1.0,
        },
        'trajectory': {
            'total': 1.0,
            'near_future_loss': 1.0,
            'distant_future_loss': 1.0,
            'decay_factor': 0.8,
        }
    }
    from lib.models.mastrack import build_model
    from lib.config.loader import load_from_yaml
    cfg = load_from_yaml('/home/yanjiezhang/Downloads/Dissertation/MainProj/experiments/fe108_mastrack.yaml')
    net = build_model(cfg, training=False).to(device)
    actor = MASTrackActor(net=net, objective=objective,
                          loss_weight=loss_weight, cfg=cfg)

    search = torch.randn(T, B, C, H, W, device=device)
    template = torch.randn(T, B, C, H, W, device=device)

    search_anno = torch.zeros(B, 1 + P * df, 4, device=device)
    search_anno[..., 0] = torch.rand(B, 1 + P * df, device=device) *(search_size * 0.5)  # x
    search_anno[..., 1] = torch.rand(B, 1 + P * df, device=device) *(search_size * 0.5)  # y
    search_anno[..., 2] = torch.rand(B, 1 + P * df, device=device) * 40 + 20
    # w
    search_anno[..., 3] = torch.rand(B, 1 + P * df, device=device) * 40 + 20
    # h

    total_loss, detailed_loss = actor((search, template, search_anno))

    print('total_loss:', total_loss)
    print('detailed_loss:')
    for k, v in detailed_loss.items():
        print(f'  {k}: {v}')



