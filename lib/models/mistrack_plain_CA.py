import torch
from torch import nn
from pathlib import Path
from lib.models.heads.track_head import build_track_head
from lib.models.backbones.backbone_plain_CA import build_backbone_large, build_backbone_medium, build_backbone_small, build_backbone_tiny


class MISTrack(nn.Module):
    def __init__(self, backbone, track_head, head_type='CENTER'):
        super().__init__()
        self.backbone = backbone
        self.track_head = track_head
        self.head_type = head_type

        if head_type in ('CENTER'):
            # head期望的search输入尺寸和元素个数, 用来从backbone输出feature map中裁切出search
            self.search_feature_size = int(self.track_head.feat_sz)
            self.search_feature_len = int(self.search_feature_size ** 2)


    def forward(self, search: torch.Tensor, template: torch.Tensor):
        # 无论单步还是多步, backbone输入都是5D(T, B, C, H, W), 输出是3D(B, C, N) N=template tokens + search tokens
        features = self.backbone(search=search, template=template)
        search_feature = features[..., :self.search_feature_len]  # (T, B, C, HW)
        B, C, HW = search_feature.size()
        search_feature = search_feature.view(B, C, self.search_feature_size, self.search_feature_size)
        out_dict = self._forward_head(fused_feature=search_feature)
        out_dict['fused_feature'] = search_feature

        return out_dict

    def _forward_head(self, fused_feature: torch.Tensor):
        if self.head_type == 'CENTER':
            # (B, 4), (B, 2, H, W), (B, 2, H, W), (B, 1, H, W)
            pred_box, score_map_ctr, offset_map, size_map, idx = self.track_head(feature=fused_feature)
            out_dict = {'pred_boxes': pred_box, # (B, 4)
                        'score_map': score_map_ctr, # (B, 1, H, W)
                        'offset_map': offset_map, # (B, 2, H, W)
                        'size_map' : size_map, # (B, 2, H, W)
                        }
            return out_dict
        else:
            raise NotImplementedError(f'Selected head type is not implemented: {self.head_type}')
    def reset_neurons(self):
        # 重置backbone和multi_timescale_module的神经元状态
        self.backbone.reset_neurons()
        self.track_head.reset_neurons()

def _build_all_modules(cfg, t):
    # 根据模型type创建backbone, multi_timescale_module和head
    if cfg.MODEL.BACKBONE.TYPE == 'TINY':
        backbone = build_backbone_tiny(t=t)
        feat_dim = backbone.embed_dim
        track_head = build_track_head(cfg, feat_dim=feat_dim)
    elif cfg.MODEL.BACKBONE.TYPE == 'SMALL':
        backbone = build_backbone_small(t=t)
        feat_dim = backbone.embed_dim
        track_head = build_track_head(cfg, feat_dim=feat_dim)
    elif cfg.MODEL.BACKBONE.TYPE == 'MEDIUM':
        backbone = build_backbone_medium(t=t)
        feat_dim = backbone.embed_dim
        track_head = build_track_head(cfg, feat_dim=feat_dim)
    elif cfg.MODEL.BACKBONE.TYPE == 'LARGE':
        backbone = build_backbone_large(t=t)
        feat_dim = backbone.embed_dim
        track_head = build_track_head(cfg, feat_dim=feat_dim)
    else:
        raise NotImplementedError(f'Model Type {cfg.MODEL.BACKBONE.TYPE} not implemented')
    return backbone, track_head

# 计算参数总量
def _count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    backbone_params = sum(p.numel() for p in model.backbone.parameters())
    track_head_params = sum(p.numel() for p in model.track_head.parameters())
    return total_params, backbone_params, track_head_params

def build_model(cfg, training=True, load_ckpt=False):

    print(f'\033[93mstart building model:\033[0m MISTrack {cfg.MODEL.BACKBONE.TYPE} \033[93m>>>\033[0m')
    current_dir = Path(__file__).resolve().parent  # ./models
    pretrained_path = current_dir.parents[1] / 'pretrained_models'
    # print(str(current_dir))
    # print(str(pretrained_path))
    backbone, track_head = _build_all_modules(cfg, t=cfg.MODEL.T)

    model = MISTrack(
        backbone=backbone,
        track_head=track_head,
        head_type=cfg.MODEL.HEAD.TYPE,
    )

    # 加载checkpoint
    if load_ckpt:
        if not cfg.TEST.PRETRAINED_FILE_NAME:
            raise RuntimeError('PRETRAINED_FILE_NAME must be set in the configuration file when load_ckpt is True.')

        pretrained = pretrained_path / cfg.TEST.PRETRAINED_FILE_NAME
        if not pretrained.is_file():
            raise RuntimeError(f'Pretrained file not found: {pretrained}')
        ckpt = torch.load(pretrained, map_location='cpu', weights_only=False)

        if "model" in ckpt:
            model.load_state_dict(ckpt["model"], strict=True)
        elif "net" in ckpt:
            model.load_state_dict(ckpt["net"], strict=True)
        else:
            raise RuntimeError(f'Checkpoint does not contain "model" or "net" keys:{pretrained}')

        model.train(training)

    # 输出模型的总参数量
    # when t = 1, branch=4, layer=1 without gate
    # 19,289,414 with learnable pad.
    # 19,289,419 with learnable decay in multi_timescale_module, learnable pad.
    # 19,289,431 with learnable decay in head and multi_timescale_module, learnable pad
    # 19,289,499 with learnable decay in backbone and multi_timescale_module, learnable pad
    # 19,289,511 with learnable decay in all modules, learnable pad
    # the number of learnable decay parameters in backbone: 80, multi_timescale_module: 5, head: 12

    # when t = 1, branch=4, layer=1 with gate
    # 19,701,574 with learnable pad.
    # 19,701,579 with learnable decay in multi_timescale_module, learnable pad.
    # 19,701,597 with learnable decay in head and multi_timescale_module, learnable pad
    # 19,701,659 with learnable decay in backbone and multi_timescale_module, learnable pad
    # 19,701,671 with learnable decay in all modules, learnable pad

    total_params, backbone_params, track_head_params = _count_parameters(model)
    print(f"\033[92mTotal number of parameters:\033[0m {total_params:,}")
    print(f"\033[92mBackbone parameters:\033[0m {backbone_params:,}")
    print(f"\033[92mTrack head parameters:\033[0m {track_head_params:,}")

    return model

if __name__ == '__main__':
    from lib.config.loader import load_from_yaml
    cfg = load_from_yaml('/home/yanjiezhang/Downloads/Dissertation/MainProj/experiments/fe108_mistrack.yaml')
    net = build_model(cfg, training=False)
    net.to('cuda')

    dummy_search = torch.randn(1, 8, 3, 256, 256).to('cuda')
    dummy_template = torch.randn(1, 8, 3, 128, 128).to('cuda')
    import time
    with torch.inference_mode():
        start = time.time()
        y = net(search=dummy_search, template=dummy_template)
        print(time.time() - start)

    for key in y.keys():
        try:
            print(f'{key} shape: {y[key].shape}\n')
        except:
            print(f'{key} type: {type(y[key])}\n')