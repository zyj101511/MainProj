import torch
from torch import nn
from pathlib import Path
from lib.models.heads.track_head import build_head
from lib.models.backbones.backbone import build_backbone_large, build_backbone_medium, build_backbone_small, build_backbone_tiny
from lib.models.multi_timescale_module.multi_timescale_module import build_multi_timescale_module


class MASTrack(nn.Module):
    def __init__(self, backbone, multi_timescale_module, head, head_type='CENTER'):
        super().__init__()
        self.backbone = backbone
        self.multi_timescale_module = multi_timescale_module
        self.head = head
        self.head_type = head_type

        if head_type in ('CENTER'):
            # head期望的search输入尺寸和元素个数, 用来从backbone输出feature map中裁切出search
            self.search_feature_size = int(head.feat_sz)
            self.search_feature_len = int(self.search_feature_size ** 2)

    def forward(self, template: torch.Tensor, search: torch.Tensor, return_max_score=False):
        # 无论单步还是多步, backbone输入都是5D(T, B, C, H, W), 输出是4D(T, B, C, N) N=template tokens + search tokens
        features = self.backbone(template=template, search=search)
        fused_feature = self._forward_multi_timescale_module(feature=features)
        out_dict = self._forward_head(fused_feature=fused_feature, return_max_score=return_max_score)
        # 记录backbone和multi_timescale_module的输出特征
        out_dict['backbone_feature'] = features
        out_dict['fused_feature'] = fused_feature
        return out_dict

    def _forward_multi_timescale_module(self, feature: torch.Tensor):
        if feature.dim() != 4: raise RuntimeError(f'Input feature of head should be a 4D tensor(T, B, C, N): {feature.dim()}')
        if feature.size(-1) < self.search_feature_len: raise RuntimeError(f'Input feature map of head has token less than expected search feature len: {feature.size(-1)}')

        # feature(T, B, C, N)
        search_feature = feature[..., :self.search_feature_len]
        T, B, C, HW = search_feature.size()

        if HW != self.search_feature_size ** 2: raise RuntimeError('Input feature map can not align the expected search feature size')
        # search_feature(T, B, C, H, W)
        search_feature = search_feature.view(T, B, C, self.search_feature_size, self.search_feature_size)
        fused_feature = self.multi_timescale_module(search_feature)  # (B, C, H, W)
        return fused_feature

    def _forward_head(self, fused_feature: torch.Tensor, return_max_score):
        if self.head_type == 'CENTER':
            # (B, 4), (B, 2, H, W), (B, 2, H, W), (B, 1, H, W)
            pred_box, score_map_ctr, offset_map, size_map, max_score = self.head(feature=fused_feature,
                                                                                 return_max_score=return_max_score)
            out_dict = {'pred_boxes': pred_box,
                        'score_map': score_map_ctr,
                        'offset_map': offset_map,
                        'size_map' : size_map,
                        'max_score': max_score}
            return out_dict
        else:
            raise NotImplementedError(f'Selected head type is not implemented: {self.head_type}')

def _build_all_modules(cfg, t):
    # 根据模型type创建backbone, multi_timescale_module和head
    if cfg.MODEL.BACKBONE.TYPE == 'TINY':
        backbone = build_backbone_tiny(t=t)
        feat_dim = backbone.embed_dim
        multi_timescale_module = build_multi_timescale_module(t=t, in_channels=feat_dim,
                                                              num_branch=cfg.MODEL.MULTI_TIMESCALE_MODULE.NUM_BRANCHES,
                                                              num_layer=cfg.MODEL.MULTI_TIMESCALE_MODULE.NUM_LAYERS)
        head = build_head(cfg, feat_dim=feat_dim)
    elif cfg.MODEL.BACKBONE.TYPE == 'SMALL':
        backbone = build_backbone_small(t=t)
        feat_dim = backbone.embed_dim
        multi_timescale_module = build_multi_timescale_module(t=t, in_channels=feat_dim,
                                                              num_branch=cfg.MODEL.MULTI_TIMESCALE_MODULE.NUM_BRANCHES,
                                                              num_layer=cfg.MODEL.MULTI_TIMESCALE_MODULE.NUM_LAYERS)
        head = build_head(cfg, feat_dim=feat_dim)
    elif cfg.MODEL.BACKBONE.TYPE == 'MEDIUM':
        backbone = build_backbone_medium(t=t)
        feat_dim = backbone.embed_dim
        multi_timescale_module = build_multi_timescale_module(t=t, in_channels=feat_dim,
                                                              num_branch=cfg.MODEL.MULTI_TIMESCALE_MODULE.NUM_BRANCHES,
                                                              num_layer=cfg.MODEL.MULTI_TIMESCALE_MODULE.NUM_LAYERS)
        head = build_head(cfg, feat_dim=feat_dim)
    elif cfg.MODEL.BACKBONE.TYPE == 'LARGE':
        backbone = build_backbone_large(t=t)
        feat_dim = backbone.embed_dim
        multi_timescale_module = build_multi_timescale_module(t=t, in_channels=feat_dim,
                                                              num_branch=cfg.MODEL.MULTI_TIMESCALE_MODULE.NUM_BRANCHES,
                                                              num_layer=cfg.MODEL.MULTI_TIMESCALE_MODULE.NUM_LAYERS)
        head = build_head(cfg, feat_dim=feat_dim)
    else:
        raise NotImplementedError(f'Model Type {cfg.MODEL.BACKBONE.TYPE} not implemented')
    return backbone, multi_timescale_module, head

# 计算参数总量
def _count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    return total_params

def build_model(cfg, training=True):
    current_dir = Path(__file__).resolve().parent  # ./models
    pretrained_path = current_dir.parents[1] / 'pretrained_models'
    # print(str(current_dir))
    # print(str(pretrained_path))
    backbone, multi_timescale_module, head = _build_all_modules(cfg, t=cfg.MODEL.T)

    model = MASTrack(
        backbone=backbone,
        multi_timescale_module=multi_timescale_module,
        head=head,
        head_type=cfg.MODEL.HEAD.TYPE,
    )

    # 加载checkpoint, 训练时必须有
    if training:
        if not cfg.MODEL.PRETRAINED_FILE:
            raise RuntimeError('PRETRAINED_FILE must be set in the configuration file when training=True')

        pretrained = pretrained_path / cfg.MODEL.PRETRAINED_FILE
        if not pretrained.is_file():
            raise RuntimeError(f'Pretrained file not found: {pretrained}')
        ckpt = torch.load(pretrained, map_location='cpu')
        missing_keys, unexpected_keys = model.load_state_dict(ckpt["model"], strict=False)
        print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
        print('missing keys in checkpoint:')
        print(missing_keys)
        print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
        print('unexpected keys in checkpoint:')
        print(unexpected_keys)
        print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')

    # 输出模型的总参数量
    # when t = 1
    # 19,289,414 with learnable pad.
    # 19,289,419 with learnable decay in multi_timescale_module, learnable pad.
    # 19,289,431 with learnable decay in head and multi_timescale_module, learnable pad
    # 19,289,499 with learnable decay in backbone and multi_timescale_module, learnable pad
    # 19,289,511 with learnable decay in all modules, learnable pad
    # the number of learnable decay parameters in backbone: 80, multi_timescale_module: 5, head: 12
    print()
    print(f"Total number of parameters: {_count_parameters(model):,}")

    return model

if __name__ == '__main__':
    from lib.config.loader import load_from_yaml
    cfg = load_from_yaml('/home/yanjiezhang/Downloads/Dissertation/MainProj/experiments/fe108_mastrack.yaml')
    net = build_model(cfg, training=False)
    net.to('cuda')

    dummy_search = torch.randn(1, 8, 3, 256, 256).to('cuda')
    dummy_template = torch.randn(1, 8, 3, 128, 128).to('cuda')
    import time
    with torch.inference_mode():
        start = time.time()
        y = net(search=dummy_search, template=dummy_template, return_max_score=False)
        print(time.time() - start)

    for key in y.keys():
        try:
            print(f'{key} shape: {y[key].shape}\n')
        except:
            print(f'{key} type: {type(y[key])}\n')