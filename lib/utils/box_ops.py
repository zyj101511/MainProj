import torch
from torchvision.ops.boxes import box_area
import numpy as np


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)

def box_xywh_to_cxcywh(x):
    x1, y1, w, h = x.unbind(-1)
    b = [x1 + 0.5 * w, y1 + 0.5 * h, w, h]
    return torch.stack(b, dim=-1)


def box_xywh_to_xyxy(x):
    x1, y1, w, h = x.unbind(-1)
    b = [x1, y1, x1 + w, y1 + h]
    return torch.stack(b, dim=-1)


def box_xyxy_to_xywh(x):
    x1, y1, x2, y2 = x.unbind(-1)
    b = [x1, y1, x2 - x1, y2 - y1]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


# modified from torchvision to also return the union
'''Note that this function only supports shape (N,4)'''


def box_iou(boxes1, boxes2):
    """

    :param boxes1: (N, 4) (x1,y1,x2,y2)
    :param boxes2: (N, 4) (x1,y1,x2,y2)
    :return:
    """
    area1 = box_area(boxes1) # (N,)
    area2 = box_area(boxes2) # (N,)

    lt = torch.max(boxes1[:, :2], boxes2[:, :2])  # (N,2)
    rb = torch.min(boxes1[:, 2:], boxes2[:, 2:])  # (N,2)

    wh = (rb - lt).clamp(min=0)  # (N,2)
    inter = wh[:, 0] * wh[:, 1]  # (N,)

    union = area1 + area2 - inter

    iou = inter / union
    return iou, union


'''Note that this implementation is different from DETR's'''


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    boxes1: (N, 4)
    boxes2: (N, 4)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    # try:
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2) # (N,)

    lt = torch.min(boxes1[:, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # (N,2)
    area = wh[:, 0] * wh[:, 1] # (N,)

    return iou - (area - union) / area, iou


def giou_loss(boxes1, boxes2):
    """

    :param boxes1: (N, 4) (x1,y1,x2,y2)
    :param boxes2: (N, 4) (x1,y1,x2,y2)
    :return:
    """
    giou, iou = generalized_box_iou(boxes1, boxes2)
    return (1 - giou).mean(), iou


def clip_box(box: list, H, W, margin=0):
    """input box: (x1, y1, w, h), output box: (x1, y1, w, h)"""
    x1, y1, w, h = box
    x2, y2 = x1 + w, y1 + h
    x1 = min(max(0, x1), W-margin)  # 左上x不能小于0, 不能大于W-margin
    x2 = min(max(margin, x2), W)  # 右下x不能小于margin(太靠左), 不能大于W
    y1 = min(max(0, y1), H-margin)  # 左上y不能小于0, 不能大于H-margin
    y2 = min(max(margin, y2), H)  # 右下y不能小于margin(太靠上), 不能大于H
    w = max(margin, x2-x1)  # 宽度不能小于margin
    h = max(margin, y2-y1)  # 高度不能小于margin
    return (x1, y1, w, h)

def clip_box_tensor(box, H, W, margin=0):
    # box: (4) (cx, cy, w, h)
    cx, cy, w, h = box.unbind(-1)

    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h

    x1 = torch.clamp(x1, min=0, max=W - margin)  # 左上x不能小于0, 不能大于W-margin
    x2 = torch.clamp(x2, min=margin, max=W)  # 右下x不能小于margin(太靠左), 不能大于W
    y1 = torch.clamp(y1, min=0, max=H - margin) # 左上y不能小于0, 不能大于H-margin
    y2 = torch.clamp(y2, min=margin, max=H) # 右下y不能小于margin(太靠上), 不能大于H

    w = torch.clamp(x2 - x1, min=margin) # 宽度不能小于margin
    h = torch.clamp(y2 - y1, min=margin) # 高度不能小于margin
    return torch.stack([x1, y1, w, h], dim=-1)

