import torch
from torch import nn
from lib.models.neuron import MILIF


class _conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
        super().__init__()
        self.channels = out_channels
        self.spike = MILIF(min_v=0.,
                           max_v=4.0,
                           norm=None,
                           decay=True,
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
                              out_channels=self.channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              bias=bias)
        self.bn = nn.BatchNorm2d(self.channels)

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

class TrajectoryPredictor(nn.Module):
    '''
    bbox pred: normalized (cx, cy, w, h)
    '''
    def __init__(self, in_channels=64, hidden_channels=256, search_feat_size=20, stride=16, P=4, distance_factor=4):
        super().__init__()
        self.feat_sz = search_feat_size
        self.stride=stride
        self.img_sz =  self.feat_sz * self.stride
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.P = P
        self.df = distance_factor

        # near future center predict
        self.conv1_ctr = _conv(in_channels=self.in_channels, out_channels=self.hidden_channels)
        self.conv2_ctr = _conv(in_channels=self.hidden_channels, out_channels=self.hidden_channels // 2)
        self.conv3_ctr = _conv(in_channels=self.hidden_channels // 2, out_channels=self.hidden_channels // 4)
        self.conv4_ctr = _conv(in_channels=self.hidden_channels // 4, out_channels=self.hidden_channels // 8)
        self.conv5_ctr = nn.Conv2d(in_channels=self.hidden_channels // 8, out_channels=self.P, kernel_size=1)

        # near future offset regress
        self.conv1_offset = _conv(in_channels=self.in_channels, out_channels=self.hidden_channels)
        self.conv2_offset = _conv(in_channels=self.hidden_channels, out_channels=self.hidden_channels // 2)
        self.conv3_offset = _conv(in_channels=self.hidden_channels // 2, out_channels=self.hidden_channels // 4)
        self.conv4_offset = _conv(in_channels=self.hidden_channels // 4, out_channels=self.hidden_channels // 8)
        self.conv5_offset = nn.Conv2d(in_channels=self.hidden_channels // 8, out_channels=2*self.P, kernel_size=1)  # 通道是x,y,x,y,...,x,y

        # current vx, vy
        self.conv1_v = _conv(in_channels=self.in_channels, out_channels=self.hidden_channels)
        self.conv2_v = _conv(in_channels=self.hidden_channels, out_channels=self.hidden_channels // 2)
        self.conv3_v = _conv(in_channels=self.hidden_channels // 2, out_channels=self.hidden_channels // 4)
        self.conv4_v = _conv(in_channels=self.hidden_channels // 4, out_channels=self.hidden_channels // 8)
        self.conv5_v = nn.Conv2d(in_channels=self.hidden_channels // 8, out_channels=2, kernel_size=1)  # 通道是vx,vy

        # current ax, ay
        self.conv1_a = _conv(in_channels=self.in_channels, out_channels=self.hidden_channels)
        self.conv2_a = _conv(in_channels=self.hidden_channels, out_channels=self.hidden_channels // 2)
        self.conv3_a = _conv(in_channels=self.hidden_channels // 2, out_channels=self.hidden_channels // 4)
        self.conv4_a = _conv(in_channels=self.hidden_channels // 4, out_channels=self.hidden_channels // 8)
        self.conv5_a = nn.Conv2d(in_channels=self.hidden_channels // 8, out_channels=2, kernel_size=1)  # 通道是ax,ay

        # distant future a delta predict
        self.conv1_a_delta = _conv(in_channels=self.in_channels, out_channels=self.hidden_channels)
        self.conv2_a_delta = _conv(in_channels=self.hidden_channels, out_channels=self.hidden_channels // 2)
        self.conv3_a_delta = _conv(in_channels=self.hidden_channels // 2, out_channels=self.hidden_channels // 4)
        self.conv4_a_delta = _conv(in_channels=self.hidden_channels // 4, out_channels=self.hidden_channels // 8)
        # 通道是ax_delta,ay_delta,ax_delta,ay_delta,...,ax_delta,ay_delta,共distance_factor-1组
        self.conv5_a_delta = nn.Conv2d(in_channels=self.hidden_channels // 8, out_channels=2*(self.df-1), kernel_size=1)

        for p in self.parameters():  # 除了batchnorm, 把conv都初始化
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, feature, track_idx):  #  (B, C, H, W)
        # (B, 1, H, W) (B, 2, H, W) (B, 2, H, W)
        score_map_ctr, map_offset, map_v, map_a, map_a_delta = self.get_score_map(feature)
        near_future_ctr, cur_v, cur_a, a_deltas = self.get_pred(score_map_ctr, map_offset,
                                                                map_v, map_a, map_a_delta, track_idx=track_idx)
        return near_future_ctr, cur_v, cur_a, a_deltas

    def get_score_map(self, x):
        def _clamp_sigmoid(x):
            y = torch.clamp(x.sigmoid_(), min=1e-4, max=1 - 1e-4)
            return y

        # near future center predict
        x_ctr1 = self.conv1_ctr(x)
        x_ctr2 = self.conv2_ctr(x_ctr1)
        x_ctr3 = self.conv3_ctr(x_ctr2)
        x_ctr4 = self.conv4_ctr(x_ctr3)
        score_map_ctr = self.conv5_ctr(x_ctr4)  # B, P, H, W

        # near future offset regress
        x_offset1 = self.conv1_offset(x)
        x_offset2 = self.conv2_offset(x_offset1)
        x_offset3 = self.conv3_offset(x_offset2)
        x_offset4 = self.conv4_offset(x_offset3)
        map_offset = self.conv5_offset(x_offset4)  # B, 2*P, H, W

        # current vx, vy
        x_v1 = self.conv1_v(x)
        x_v2 = self.conv2_v(x_v1)
        x_v3 = self.conv3_v(x_v2)
        x_v4 = self.conv4_v(x_v3)
        map_v = self.conv5_v(x_v4)  # B, 2, H, W

        # current ax, ay
        x_a1 = self.conv1_a(x)
        x_a2 = self.conv2_a(x_a1)
        x_a3 = self.conv3_a(x_a2)
        x_a4 = self.conv4_a(x_a3)
        map_a = self.conv5_a(x_a4)  # B, 2, H, W

        # distant future a delta predict
        x_a_delta1 = self.conv1_a_delta(x)
        x_a_delta2 = self.conv2_a_delta(x_a_delta1)
        x_a_delta3 = self.conv3_a_delta(x_a_delta2)
        x_a_delta4 = self.conv4_a_delta(x_a_delta3)
        map_a_delta = self.conv5_a_delta(x_a_delta4)  # B, 2*(df-1), H, W

        return _clamp_sigmoid(score_map_ctr), 2*_clamp_sigmoid(map_offset)-1, map_v, map_a, map_a_delta

    def get_near_future(self, score_map_ctr, map_offset):
        # score_map_ctr: (B, P, H, W), map_offset: (B, 2*P, H, W),
        # map_v: (B, 2, H, W), map_a: (B, 2, H, W), map_a_delta: (B, 2*(df-1), H, W)
        B, P, H, W = score_map_ctr.shape
        score_flat = score_map_ctr.flatten(2)  # (B, P, H*W)
        _, idx = torch.max(score_flat, dim=-1)  # (B, P)
        idx_y = idx // self.feat_sz
        idx_x = idx % self.feat_sz

        # idx (B, P)
        B, _, H, W = map_offset.shape
        map_offset = map_offset.view(B, P, 2, H, W)  # (B, P, 2, H, W)
        map_offset = map_offset.flatten(3)  # (B, P, 2, H*W)
        idx_expanded = idx.unsqueeze(2).unsqueeze(-1)  # (B, P, 1, 1)
        idx_expanded = idx_expanded.expand(-1, -1, 2, -1)  # (B, P, 2, 1)
        offset_xy = map_offset.gather(dim=3, index=idx_expanded)  # (B, P, 2, 1)
        offset_xy = offset_xy.squeeze(-1)  # (B, P, 2)

        ctr_x = ((idx_x.to(torch.float) + offset_xy[..., 0]) / self.feat_sz)
        ctr_y = ((idx_y.to(torch.float) + offset_xy[..., 1]) / self.feat_sz)

        near_future_ctr = torch.stack([ctr_x, ctr_y], dim=-1)  # (B, P, 2)
        return near_future_ctr

    def get_pred(self, score_map_ctr, map_offset, map_v, map_a, map_a_delta, track_idx):
        # track_idx (B, 2, 1)
        near_future_ctr = self.get_near_future(score_map_ctr, map_offset)
        cur_v = map_v.flatten(2).gather(dim=-1, index=track_idx).squeeze(-1)  # (B, 2)
        cur_a = map_a.flatten(2).gather(dim=-1, index=track_idx).squeeze(-1)  # (B, 2)
        B, _, H, W = map_a_delta.shape
        map_a_delta = map_a_delta.view(B, self.df-1, 2, H, W)  # (B, df-1, 2, H, W)
        map_a_delta = map_a_delta.flatten(3)  # (B, df-1, 2, H*W)
        idx_expanded = track_idx.unsqueeze(1)  # (B, 1, 2, 1)
        idx_expanded = idx_expanded.expand(-1, self.df-1, 2, -1)  # (B, df-1, 2, 1)
        a_deltas = map_a_delta.gather(dim=3, index=idx_expanded).squeeze(-1)  # (B, df-1, 2)
        return near_future_ctr, cur_v, cur_a, a_deltas

    def reset_neurons(self):
        for m in self.modules():
            if isinstance(m, MILIF):
                m.reset()

def build_trajectory_head(cfg, feat_dim):

    stride = cfg.MODEL.BACKBONE.STRIDE
    df = cfg.MODEL.HEAD.DISTANCE_FACTOR
    P = cfg.MODEL.HEAD.P
    in_channels = feat_dim
    out_channels = cfg.MODEL.HEAD.NUM_CHANNELS
    feat_sz = int(cfg.DATA.SEARCH.SIZE / stride)
    pred_head = TrajectoryPredictor(in_channels=in_channels,
                                      hidden_channels=out_channels,
                                      search_feat_size=feat_sz,
                                      stride=stride,
                                      P=P,
                                      distance_factor=df)
    return pred_head

