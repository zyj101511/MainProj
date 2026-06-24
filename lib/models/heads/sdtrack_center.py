import torch
import torch.nn.functional as F
from torch import nn
from lib.models.backbones.neuron import MILIF
from spikingjelly.activation_based.layer import SeqToANNContainer


class _conv(nn.Module):
    def __init__(self, t, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
        super().__init__()
        self.channel = out_channels
        self.lif = MILIF(min_v=0.,
                            max_v=4.0,
                            norm=None,
                            t=t,
                            decay=True,
                            decay_rate=0.25,
                            state_clip=(-0.5, 4),
                            learnable_decay=False,
                            mem=True,
                            infere_mode=False,
                            detach_reset=True,
                            store_v_seq=False)
        self.conv = SeqToANNContainer(nn.Conv2d(in_channels=in_channels,
                                                out_channels=self.channel,
                                                kernel_size=kernel_size,
                                                stride=stride,
                                                padding=padding,
                                                bias=bias))
        self.bn = SeqToANNContainer(nn.BatchNorm2d(self.channel))

    def forward(self, x):
        x = self.lif(x)
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
    def __init__(self, t, in_channel=64, hidden_channel=256, search_feat_size=20, stride=16):
        super().__init__()
        self.feat_sz = search_feat_size
        self.stride=stride
        self.img_sz =  self.feat_sz * self.stride
        self.in_channel = in_channel
        self.hidden_channel = hidden_channel

        # center predict
        self.conv1_ctr = _conv(t=t, in_channels=self.in_channel, out_channels=self.hidden_channel)
        self.conv2_ctr = _conv(t=t, in_channels=self.hidden_channel, out_channels=self.hidden_channel // 2)
        self.conv3_ctr = _conv(t=t, in_channels=self.hidden_channel // 2, out_channels=self.hidden_channel // 4)
        self.conv4_ctr = _conv(t=t, in_channels=self.hidden_channel // 4, out_channels=self.hidden_channel // 8)
        self.conv5_ctr = SeqToANNContainer(nn.Conv2d(in_channels=self.hidden_channel // 8,
                                                     out_channels=1, kernel_size=1))

        # offset regress
        self.conv1_offset = _conv(t=t, in_channels=self.in_channel, out_channels=self.hidden_channel)
        self.conv2_offset = _conv(t=t, in_channels=self.hidden_channel, out_channels=self.hidden_channel // 2)
        self.conv3_offset = _conv(t=t, in_channels=self.hidden_channel // 2, out_channels=self.hidden_channel // 4)
        self.conv4_offset = _conv(t=t, in_channels=self.hidden_channel // 4, out_channels=self.hidden_channel // 8)
        self.conv5_offset = SeqToANNContainer(nn.Conv2d(in_channels=self.hidden_channel // 8,
                                                        out_channels=2, kernel_size=1))

        # size regress
        self.conv1_size = _conv(t=t, in_channels=self.in_channel, out_channels=self.hidden_channel)
        self.conv2_size = _conv(t=t, in_channels=self.hidden_channel, out_channels=self.hidden_channel // 2)
        self.conv3_size = _conv(t=t, in_channels=self.hidden_channel // 2, out_channels=self.hidden_channel // 4)
        self.conv4_size = _conv(t=t, in_channels=self.hidden_channel // 4, out_channels=self.hidden_channel // 8)
        self.conv5_size = SeqToANNContainer(nn.Conv2d(in_channels=self.hidden_channel // 8,
                                                      out_channels=2, kernel_size=1))

        for p in self.parameters():  # 除了batchnorm, 把conv都初始化
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, feature, gt_score_map=None, return_mean_bbox=True, return_max_score=False):  #  (T, B, C, H, W)
        # (T, B, 1, H, W) (T, B, 2, H, W) (T, B, 2, H, W)
        score_map_ctr, offset_map, size_map = self.get_score_map(feature)

        if gt_score_map is None:
            bbox, max_score = self.cal_bbox(score_map_ctr, offset_map, size_map, return_max_score)  # (T, B, 4), 4:(x, y, h, w)
        else:
            bbox, max_score = self.cal_bbox(gt_score_map, offset_map, size_map, return_max_score)
        if return_mean_bbox:
            return bbox.mean(0), max_score, score_map_ctr, offset_map, size_map
        return bbox, max_score, score_map_ctr, offset_map, size_map

    def cal_bbox(self, score_map_ctr, offset_map, size_map,
                 return_max_score=False):  # (T, B, 1, H, W) (T, B, 2, H, W) (T, B, 2, H, W)
        max_score, idx = torch.max(score_map_ctr.flatten(2), dim=-1, keepdim=True)
        idx_y = idx // self.feat_sz
        idx_x = idx % self.feat_sz
        # idx (T, B, 1)
        idx = idx.unsqueeze(2).expand(idx.shape[0], idx.shape[1], 2, 1)  # (T, B, 2, 1)
        offset = offset_map.flatten(3).gather(dim=-1, index=idx)  # (T, B, 2, 1)
        size = size_map.flatten(3).gather(dim=-1, index=idx)  # (T, B, 2, 1)

        x = (idx_x.to(torch.float) + offset[..., 0, :])
        y = (idx_y.to(torch.float) + offset[..., 1, :])
        h = size[..., 0, :]
        w = size[..., 1, :]

        bbox = torch.cat([x, y, h, w], dim=-1)
        if return_max_score:
            return bbox, max_score
        return bbox, None  # (T, B, 4), 4:(x, y, h, w)

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
        # (T, B, 1, H, W) (T, B, 2, H, W) (T, B, 2, H, W)
        return _clamp_sigmoid(score_map_ctr), 2*_clamp_sigmoid(score_map_offset)-1, _clamp_sigmoid(score_map_size)

    def get_pred(self, score_map_ctr, size_map, offset_map):
        max_score, idx = torch.max(score_map_ctr.flatten(2), dim=-1, keepdim=True)
        idx_y = idx // self.feat_sz
        idx_x = idx % self.feat_sz
        # idx (T, B, 1)
        idx = idx.unsqueeze(2).expand(idx.shape[0], idx.shape[1], 2, 1)  # (T, B, 2, 1)
        offset = offset_map.flatten(3).gather(dim=-1, index=idx).squeeze(-1)  # (T, B, 2)
        size = size_map.flatten(3).gather(dim=-1, index=idx).squeeze(-1)  # (T, B, 2)

        ctr_x = ((idx_x.to(torch.float) + offset[..., 0]) / self.feat_sz)
        ctr_y = ((idx_y.to(torch.float) + offset[..., 1]) / self.feat_sz)
        offset_x = offset[..., 0]
        offset_y = offset[..., 1]
        size_h = size[..., 0] * self.feat_sz
        size_w = size[..., 1] * self.feat_sz
        return ctr_x, ctr_y, offset_x, offset_y, size_h, size_w

class _mlp(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, BN=False):
        super().__init__()
        self.num_layers = num_layers
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        if BN:
            self.layers = nn.ModuleList(nn.Sequential(nn.Linear(n, k), nn.BatchNorm1d(k))
                                        for n, k in zip([input_dim] + h, h + [output_dim]))
        else:
            self.layers = nn.ModuleList(nn.Linear(n, k)
                                        for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

def build_head(cfg, t, feat_dim):

    stride = cfg.MODEL.BACKBONE.STRIDE
    if cfg.MODEL.HEAD.TYPE == "MLP":
        mlp_head = _mlp(feat_dim, feat_dim, 4, 3)  # dim_in, dim_hidden, dim_out, 3 layers
        return mlp_head

    elif cfg.MODEL.HEAD.TYPE == "CENTER":
        in_channel = feat_dim
        out_channel = cfg.MODEL.HEAD.NUM_CHANNELS
        feat_sz = int(cfg.DATA.SEARCH.SIZE / stride)
        center_head = CenterPredictor(t=t,
                                      in_channel=in_channel,
                                      hidden_channel=out_channel,
                                      search_feat_size=feat_sz,
                                      stride=stride)
        return center_head
    else:
        raise ValueError(f"HEAD TYPE {cfg.MODEL.HEAD_TYPE} is not supported.")


if __name__ == '__main__':
    from lib.config.loader import load_from_yaml
    cfg = load_from_yaml('/home/yanjiezhang/Downloads/Dissertation/MainProj/experiments/fe108_sdtrack_tiny.yaml')
    head = build_head(cfg, t=5, feat_dim=256).to('cuda')
    inp = torch.randn(5, 8, 256, 24, 24).to('cuda')
    bbox, max_score, score_map_ctr, offset_map, size_map = head(inp, gt_score_map=None, return_mean_bbox=True, return_max_score=False)
    num_p = 0
    for p in head.parameters():
        num_p += p.numel()
    print(f'The number of parameters is {num_p:,}')
    print(f'The shape of bbox is {bbox.shape}')
    print(f'The shape of score_map is {score_map_ctr.shape}: T, B, 1, H, W')
    print(f'The shape of offset_map is {offset_map.shape}: T, B, 2, H, W')
    print(f'The shape of size_map is {size_map.shape}: T, B, 2, H, W')
