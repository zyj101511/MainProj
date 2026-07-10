import torch
from torch import nn
from lib.models.neuron import MILIF


class _conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
        super().__init__()
        self.channel = out_channels
        self.spike = MILIF(min_v=0.,
                           max_v=4.0,
                           norm=None,
                           decay=False,
                           decay_rate=0.25,
                           state_clip=(-0.5, 4),
                           learnable_decay=False,
                           mem=False,
                           infere_mode=False,
                           detach_reset=True,
                           store_v_seq=False,
                           reset_mode='hard',
                           step_mode='s')
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=self.channel,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              bias=bias)
        self.bn = nn.BatchNorm2d(self.channel)

    def forward(self, x):
        x = self.spike(x)
        x = self.conv(x)
        x = self.bn(x)
        return x

'''
x = torch.ones((10, 5, 3, 8, 8))
conv = _conv(t=10, in_channels=3, out_channels=3)
y = conv(x)
print(y.shape)
'''

class CenterPredictor(nn.Module):
    '''
    bbox pred: normalized (cx, cy, w, h)
    '''
    def __init__(self, in_channels=64, hidden_channels=256, search_feat_size=20, stride=16):
        super().__init__()
        self.feat_sz = search_feat_size
        self.stride=stride
        self.img_sz =  self.feat_sz * self.stride
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels

        # center predict
        self.conv1_ctr = _conv(in_channels=self.in_channels, out_channels=self.hidden_channels)
        self.conv2_ctr = _conv(in_channels=self.hidden_channels, out_channels=self.hidden_channels // 2)
        self.conv3_ctr = _conv(in_channels=self.hidden_channels // 2, out_channels=self.hidden_channels // 4)
        self.conv4_ctr = _conv(in_channels=self.hidden_channels // 4, out_channels=self.hidden_channels // 8)
        self.conv5_ctr = nn.Conv2d(in_channels=self.hidden_channels // 8, out_channels=1, kernel_size=1)

        # offset regress
        self.conv1_offset = _conv(in_channels=self.in_channels, out_channels=self.hidden_channels)
        self.conv2_offset = _conv(in_channels=self.hidden_channels, out_channels=self.hidden_channels // 2)
        self.conv3_offset = _conv(in_channels=self.hidden_channels // 2, out_channels=self.hidden_channels // 4)
        self.conv4_offset = _conv(in_channels=self.hidden_channels // 4, out_channels=self.hidden_channels // 8)
        self.conv5_offset = nn.Conv2d(in_channels=self.hidden_channels // 8, out_channels=2, kernel_size=1)

        # size regress
        self.conv1_size = _conv(in_channels=self.in_channels, out_channels=self.hidden_channels)
        self.conv2_size = _conv(in_channels=self.hidden_channels, out_channels=self.hidden_channels // 2)
        self.conv3_size = _conv(in_channels=self.hidden_channels // 2, out_channels=self.hidden_channels // 4)
        self.conv4_size = _conv(in_channels=self.hidden_channels // 4, out_channels=self.hidden_channels // 8)
        self.conv5_size = nn.Conv2d(in_channels=self.hidden_channels // 8, out_channels=2, kernel_size=1)

        for p in self.parameters():  # 除了batchnorm, 把conv都初始化
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, feature):  #  (B, C, H, W)
        # (B, 1, H, W) (B, 2, H, W) (B, 2, H, W)
        score_map_ctr, offset_map, size_map = self.get_score_map(feature)
        bbox, idx = self.cal_bbox(score_map_ctr, offset_map, size_map)  # (B, 4), 4:(cx, cy, w, h)
        return bbox, score_map_ctr, offset_map, size_map, idx

    def cal_bbox(self, score_map_ctr, offset_map, size_map):  # (B, 1, H, W) (B, 2, H, W) (B, 2, H, W)
        cx, cy, _, _, w, h, idx = self.get_pred(score_map_ctr, offset_map, size_map)
        bbox = torch.cat([cx, cy, w, h], dim=-1)
        return bbox, idx  # (B, 4), 4:(cx, cy, w, h)

    def get_score_map(self, x):
        def _clamp_sigmoid(x):
            y = torch.clamp(x.sigmoid_(), min=1e-4, max=1 - 1e-4)
            return y

        # ctr branch
        x_ctr1 = self.conv1_ctr(x)
        x_ctr2 = self.conv2_ctr(x_ctr1)
        x_ctr3 = self.conv3_ctr(x_ctr2)
        x_ctr4 = self.conv4_ctr(x_ctr3)
        score_map_ctr = self.conv5_ctr(x_ctr4)

        # offset branch
        x_offset1 = self.conv1_offset(x)
        x_offset2 = self.conv2_offset(x_offset1)
        x_offset3 = self.conv3_offset(x_offset2)
        x_offset4 = self.conv4_offset(x_offset3)
        score_map_offset = self.conv5_offset(x_offset4)

        # size branch
        x_size1 = self.conv1_size(x)
        x_size2 = self.conv2_size(x_size1)
        x_size3 = self.conv3_size(x_size2)
        x_size4 = self.conv4_size(x_size3)
        score_map_size = self.conv5_size(x_size4)
        # (B, 1, H, W) (B, 2, H, W) (B, 2, H, W)
        # offset要不要归一化到[-1, 1]?
        return _clamp_sigmoid(score_map_ctr), 2*_clamp_sigmoid(score_map_offset)-1, _clamp_sigmoid(score_map_size)

    def get_idx(self, score_map_ctr):
        score_map_ctr = score_map_ctr.squeeze(1)  # (B, H, W)
        _, idx = torch.max(score_map_ctr.flatten(1), dim=-1, keepdim=True)  # (B, 1)
        idx_y = idx // self.feat_sz
        idx_x = idx % self.feat_sz
        idx = idx.unsqueeze(1).expand(idx.shape[0], 2, 1)  # (B, 2, 1)
        return idx, idx_x, idx_y

    def get_pred(self, score_map_ctr, offset_map, size_map):
        """
        return normalized cx, cy, offset_X, offset_y, w, h and idx
        """
        # (B, 1, H, W) (B, 2, H, W) (B, 2, H, W)
        idx, idx_x, idx_y = self.get_idx(score_map_ctr)
        offset = offset_map.flatten(2).gather(dim=-1, index=idx)  # (B, 2, 1)
        size = size_map.flatten(2).gather(dim=-1, index=idx) # (B, 2, 1)
        ctr_x = ((idx_x.to(torch.float) + offset[..., 0, :]) / self.feat_sz)
        ctr_y = ((idx_y.to(torch.float) + offset[..., 1, :]) / self.feat_sz)
        offset_x = offset[..., 0, :] / self.feat_sz
        offset_y = offset[..., 1, :] / self.feat_sz
        size_w = size[..., 0, :]
        size_h = size[..., 1, :]
        # (B, 1)
        return ctr_x, ctr_y, offset_x, offset_y, size_w, size_h, idx

    def reset_neurons(self):
        for m in self.modules():
            if isinstance(m, MILIF):
                m.reset()

def build_track_head(cfg, feat_dim):

    stride = cfg.MODEL.BACKBONE.STRIDE
    if cfg.MODEL.HEAD.TYPE == "CENTER":
        in_channels = feat_dim
        out_channels = cfg.MODEL.HEAD.NUM_CHANNELS
        feat_sz = int(cfg.DATA.SEARCH.SIZE / stride)
        center_head = CenterPredictor(in_channels=in_channels,
                                      hidden_channels=out_channels,
                                      search_feat_size=feat_sz,
                                      stride=stride)
        return center_head
    else:
        raise ValueError(f"HEAD TYPE {cfg.MODEL.HEAD_TYPE} is not supported.")


if __name__ == '__main__':
    import torch

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    B = 8
    C = 256
    H = 24
    W = 24

    head = CenterPredictor(
        in_channels=C,
        hidden_channels=256,
        search_feat_size=H,
        stride=16
    ).to(device)

    head.eval()

    feature = torch.randn(B, C, H, W, device=device)

    with torch.inference_mode():
        bbox, score_map_ctr, offset_map, size_map, idx = head(
            feature
        )

        print('====== forward output ======')
        print('bbox:', bbox.shape)  # (B, 4)
        print('idx:', idx.shape)  # (B, 1)
        print('score_map_ctr:', score_map_ctr.shape)  # (B, 1, H, W)
        print('offset_map:', offset_map.shape)  # (B, 2, H, W)
        print('size_map:', size_map.shape)  # (B, 2, H, W)

        print('\n====== bbox value range ======')
        print('bbox min:', bbox.min().item())
        print('bbox max:', bbox.max().item())
        print('bbox example:', bbox[0])

        print('\n====== score / offset / size range ======')
        print('score_map_ctr min/max:', score_map_ctr.min().item(), score_map_ctr.max().item())
        print('offset_map min/max:', offset_map.min().item(), offset_map.max().item())
        print('size_map min/max:', size_map.min().item(), size_map.max().item())

        # 如果要测试 get_pred / cal_bbox，需要用完整 T 维输出
        score_map_ctr_full, offset_map_full, size_map_full = head.get_score_map(feature)

        cx, cy, offset_x, offset_y, size_w, size_h, idx = head.get_pred(
            score_map_ctr_full,
            offset_map_full,
            size_map_full
        )

        print('\n====== get_pred output ======')
        print('cx:', cx.shape)  # (B, 1)
        print('cy:', cy.shape)  # (B, 1)
        print('offset_x:', offset_x.shape)  # (B, 1)
        print('offset_y:', offset_y.shape)  # (B, 1)
        print('size_w:', size_w.shape)  # (B, 1)
        print('size_h:', size_h.shape)  # (B, 1)
        print('idx:', idx.shape)  # (B, 1)

        bbox_t, idx = head.cal_bbox(
            score_map_ctr_full,
            offset_map_full,
            size_map_full,
        )

        print('\n====== cal_bbox output ======')
        print('bbox_t:', bbox_t.shape)  # (B, 4)
        print('idx:', idx.shape)  # (B, 1)
        print('bbox_t example:', bbox_t[0, 0])

    num_p = sum(p.numel() for p in head.parameters())
    print('\n====== params ======')
    print(f'The number of parameters is {num_p:,}')


