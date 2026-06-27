import torch
from torch import nn
from pathlib import Path
from lib.models.heads.sdtrack_center import build_head
from lib.models.backbones.sdtrack_tiny import sdtrack_tiny


class SDTrack(nn.Module):
    def __init__(self, backbone, head, head_type='CENTER'):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.head_type = head_type

        if head_type in ('CENTER', 'CORNER'):
            # head期望的search输入尺寸和元素个数, 用来从backbone输出feature map中裁切出search
            self.search_feature_size = int(head.feat_sz)
            self.search_feature_len = int(self.search_feature_size ** 2)

    def forward(self, template: torch.Tensor, search: torch.Tensor, return_last=True, return_max_score=False):
        # 无论单步还是多步, backbone输入都是5D(T, B, C, H, W), 输出是4D(T, B, N, C) N=template tokens + search tokens
        features, aux_dict = self.backbone(template=template, search=search)
        last_feature = features[-1] if isinstance(features, list) else features  # 返回多层特征的backbone只用最后一层
        out_dict = self._forward_head(feature=last_feature, return_last=return_last, return_max_score=return_max_score)
        # 记录辅助输出和backbone的输出特征
        out_dict.update(aux_dict)
        out_dict['backbone_feature'] = features
        return out_dict

    def _forward_head(self, feature: torch.Tensor, return_last, return_max_score):
        if feature.dim() != 4: raise RuntimeError(f'Input feature of head should be a 4D tensor(T, B, N, C): {feature.dim()}')
        if feature.size(-2) < self.search_feature_len: raise RuntimeError(f'Input feature map of head has token less than expected search feature len: {feature.size(-2)}')

        # feature(T, B, N, C)
        search_feature = feature[..., -self.search_feature_len:, :]
        search_feature = search_feature.permute(0, 1, 3, 2).contiguous()
        T, B, C, HW = search_feature.size()

        if HW != self.search_feature_size ** 2: raise RuntimeError('Input feature map can not align the expected search feature size')
        # search_feature(T, B, C, H, W)
        search_feature = search_feature.view(T, B, C, self.search_feature_size, self.search_feature_size)

        if self.head_type == 'CENTER':
            # (B, 4), (B, 2, H, W), (B, 2, H, W), (B, 1, H, W)
            pred_box, score_map_ctr, offset_map, size_map, max_score = self.head(feature=search_feature,
                                                                                 return_last=return_last,
                                                                                 return_max_score=return_max_score)
            out_dict = {'pred_boxes': pred_box,
                        'score_map': score_map_ctr,
                        'offset_map': offset_map,
                        'size_map' : size_map,
                        'max_score': max_score}
            return out_dict
        else:
            raise NotImplementedError(f'Selected head type is not implemented: {self.head_type}')

def _build_backbone_head(cfg, t):
    # 根据模型type创建backbone和head
    if cfg.MODEL.BACKBONE.TYPE == 'TINY':
        backbone = sdtrack_tiny(t=t)
        feat_dim = backbone.embed_dim
        head = build_head(cfg, t=t, feat_dim=feat_dim)
    else:
        raise NotImplementedError(f'Model Type {cfg.MODEL.BACKBONE.TYPE} not implemented')
    return backbone, head

# 计算参数总量
def _count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    return total_params

def build_model(cfg, t, training=True):
    current_dir = Path(__file__).resolve().parent  # ./models
    pretrained_path = current_dir.parents[1] / 'pretrained_models'
    # print(str(current_dir))
    # print(str(pretrained_path))
    backbone, head = _build_backbone_head(cfg, t)

    model = SDTrack(
        backbone=backbone,
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
    # 18,257,654 with learnable decay and pad
    # 18,257,653 with learnable decay
    # 18,257,413 with nothing
    print()
    print(f"Total number of parameters: {_count_parameters(model):,}")

    return model

if __name__ == '__main__':
    from lib.config.loader import load_from_yaml
    cfg = load_from_yaml('/home/yanjiezhang/Downloads/Dissertation/MainProj/experiments/fe108_sdtrack_tiny.yaml')
    net = build_model(cfg, t=1, training=False)
    net.to('cuda')

    dummy_search = torch.randn(1, 32, 3, 256, 256).to('cuda')
    dummy_template = torch.randn(1, 32, 3, 128, 128).to('cuda')
    import time
    with torch.inference_mode():
        start = time.time()
        y = net(search=dummy_search, template=dummy_template, return_last=False, return_max_score=False)
        print(time.time() - start)

    for key in y.keys():
        try:
            print(f'{key} shape: {y[key].shape}\n')
        except:
            print(f'{key} type: {type(y[key])}\n')