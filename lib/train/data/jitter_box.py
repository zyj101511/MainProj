import torch


def get_jittered_box(bbox, center_factor, scale_factor, mode: str):
    """
    given bbox, return jittered crop area
    :param bbox: tensor(4,) / (B, 4)/ (T, B, 4): 左上x,左上y,w,h
    :param center_factor: template and search tensor(2,)/ (B, 2)/ (T, B, 2)
    :param scale_factor: template and search tensor(2,)/ (B, 2)/ (T, B, 2)
    :param mode: template or search
    """

    if not bbox.dim() == center_factor.dim() == scale_factor.dim():
        raise RuntimeError(f'Both bbox{bbox.shape}, center factor{center_factor.shape} '
                           f'and scale factor{scale_factor.shape} must have the same dim')
    if mode not in ['template', 'search']:
        raise RuntimeError(f'mode {mode} is not supported')

    if mode == 'template':
        mode_idx = 0
    else:
        mode_idx = 1
    device = bbox.device
    # 用exp保证缩放永远是正数, <1缩小, >1放大
    jittered_size = bbox[..., 2:] * torch.exp(torch.randn_like(bbox[..., 2:], device=device) * scale_factor[..., mode_idx].unsqueeze(-1))
    # prod(dim=-1)就变成了(T, B), 每个元素是H*W面积,开方是得到一个典型的长度尺度
    # max_offset限制中心最大的jitter像素
    max_offset = (jittered_size.prod(dim=-1).sqrt() * center_factor[..., mode_idx]).unsqueeze(-1)
    jittered_center = bbox[..., :2] + 0.5 * bbox[..., 2:] + max_offset * (torch.rand_like(bbox[..., :2], device=device) - 0.5)
    # 还原成(左上x, 左上角y, w, h)
    jittered_bbox = torch.cat([jittered_center-0.5*jittered_size, jittered_size], dim=-1)
    return jittered_bbox


if __name__ == "__main__":
    a = torch.ones(2, 5, 4)
    b = torch.ones_like(a[..., :2])
    factor = torch.ones(2, 5, 2)
    print(b.shape)
    c = torch.arange(10).reshape(2, 5, 1)
    jittered_size = b * c
    print(jittered_size.shape)
    j = jittered_size.prod(dim=-1)* factor[..., 1]
    print(j.shape)

