import torch
import math
import cv2 as cv
import torch.nn.functional as F
import numpy as np
from lib.utils.box_ops import box_cxcywh_to_xyxy

def crop_template(img, gt_box, scale_factor:float, reshape_sz):
    """ Extracts a square crop centered at target_bb box, of area search_area_factor^2 times target_bb area

    args:
        img - TCHW tensor
        gt_box - ground truth box [x, y, w, h]
        scale_factor - Ratio of crop size to target size
        reshape_sz - (float) Size to which the extracted crop is resized (always square).

    returns:
        cv image - extracted crop
        float - the factor by which the crop has been resized to make the crop size equal output_size
    """
    if not isinstance(gt_box, list):
        x, y, w, h = gt_box.tolist()
    else:
        x, y, w, h = gt_box
    # Crop image
    crop_sz = math.ceil(math.sqrt(w * h) * scale_factor)

    if crop_sz < 1:
        raise Exception('Too small bounding box.')

    x1 = round(x + 0.5 * w - crop_sz * 0.5)
    x2 = x1 + crop_sz

    y1 = round(y + 0.5 * h - crop_sz * 0.5)
    y2 = y1 + crop_sz

    x1_pad = max(0, -x1)
    x2_pad = max(x2 - img.shape[3], 0)

    y1_pad = max(0, -y1)
    y2_pad = max(y2 - img.shape[2], 0)

    # Crop target
    im_crop = img[:, :, y1 + y1_pad:y2 - y2_pad, x1 + x1_pad:x2 - x2_pad]

    # Pad
    # im_crop: (T, C, H, W)
    im_crop = F.pad(
        im_crop,
        pad=(x1_pad, x2_pad, y1_pad, y2_pad),  # left, right, top, bottom
        mode="constant",
        value=0,
    )
    im_crop = F.interpolate(
        im_crop,
        size=(reshape_sz, reshape_sz),
        mode="bilinear",
        align_corners=False,
    )
    return im_crop


def get_search_box(image, pred_box, scale_factor:float):
    """ Extracts a square crop centered at target_bb box, of area search_area_factor^2 times target_bb area

    args:
        img - TCHW tensor
        pred_box - predicted box (4):(x, y, w, h) tensor
        scale_factor - Ratio of crop size to target size
        reshape_sz - (float) Size to which the extracted crop is resized (always square).
    """
    x, y, w, h = pred_box.tolist()

    T, C, h_origin, w_origin = image.shape

    # Crop image
    crop_sz = math.ceil(math.sqrt(w * h) * scale_factor)

    if crop_sz < 1:
        raise Exception('Too small bounding box.')

    x1 = round(x + 0.5 * w - crop_sz * 0.5)
    y1 = round(y + 0.5 * h - crop_sz * 0.5)

    x1 = max(0, min(x1, w_origin - crop_sz))
    y1 = max(0, min(y1, h_origin - crop_sz))
    x2 = x1 + crop_sz
    y2 = y1 + crop_sz
    return torch.tensor([x1, y1, x2, y2], dtype=torch.float32, device=image.device)


def get_search_box_plain(image, pred_box, scale_factor:float):
    """ Extracts a square crop centered at target_bb box, of area search_area_factor^2 times target_bb area

    args:
        img - TCHW tensor
        pred_box - predicted box (4):(x, y, w, h) tensor
        scale_factor - Ratio of crop size to target size
        reshape_sz - (float) Size to which the extracted crop is resized (always square).
    """

    x, y, w, h = pred_box.tolist()
    image_size = image.shape[2] * image.shape[3]
    cx = x + 0.5 * w
    cy = y + 0.5 * h
    if w * h > image_size * 0.4:
        crop_sz = math.ceil(math.sqrt(w * h))
    else:
        crop_sz = math.ceil(math.sqrt(w * h) * scale_factor)

    return box_cxcywh_to_xyxy(torch.tensor([cx, cy, crop_sz, crop_sz], dtype=torch.float32, device=image.device))


def crop_search(image, search_box, resize_sz):
    """ Extracts a square crop centered at target_bb box, of area search_area_factor^2 times target_bb area

    args:
        img - TCHW tensor
        search_box - (4):(x1, y1, x2, y2)tensor
        reshape_sz - (float) Size to which the extracted crop is resized (always square).

    """
    x1, y1, x2, y2 = search_box.tolist()

    x1, y1, x2, y2 = map(lambda v: int(round(v)), [x1, y1, x2, y2])

    im_crop = image[:, :, y1:y2 , x1:x2]
    H_scaler = (y2 - y1) / resize_sz
    W_scaler = (x2 - x1) / resize_sz

    im_crop = F.interpolate(
        im_crop,
        size=(resize_sz, resize_sz),
        mode="bilinear",
        align_corners=False,
    )

    return im_crop, H_scaler, W_scaler


def crop_search_plain(image, search_box, resize_sz):
    """ Extracts a square crop centered at target_bb box, of area search_area_factor^2 times target_bb area

    args:
        img - TCHW tensor
        search_box - (4):(x1, y1, x2, y2)tensor
        reshape_sz - (float) Size to which the extracted crop is resized (always square).

    """
    T, C, H, W = image.shape
    x1, y1, x2, y2 = search_box.tolist()

    x1, y1, x2, y2 = map(lambda v: int(round(v)), [x1, y1, x2, y2])

    # 计算和原图的相交区域
    src_x1 = max(0, x1)
    src_y1 = max(0, y1)
    src_x2 = min(W, x2)
    src_y2 = min(H, y2)

    crop_w = x2 - x1
    crop_h = y2 - y1

    # 先建一个带 padding 的 crop 画布tensor
    crop = torch.zeros((T, C, crop_h, crop_w), dtype=image.dtype, device=image.device)

    # 把原图有效区域拷进去
    dst_x1 = src_x1 - x1
    dst_y1 = src_y1 - y1
    dst_x2 = dst_x1 + (src_x2 - src_x1)
    dst_y2 = dst_y1 + (src_y2 - src_y1)

    crop[:, :, dst_y1:dst_y2, dst_x1:dst_x2] = image[:, :, src_y1:src_y2, src_x1:src_x2]

    H_scaler = (y2 - y1) / resize_sz
    W_scaler = (x2 - x1) / resize_sz

    im_crop = F.interpolate(
        crop,
        size=(resize_sz, resize_sz),
        mode="bilinear",
        align_corners=False,
    )

    return im_crop, H_scaler, W_scaler


def sample_target(image, target_bb, search_area_factor, output_sz):
    x, y, w, h = [float(v) for v in target_bb.tolist()]
    crop_sz = int(np.ceil(np.sqrt(max(w * h, 1e-6)) * search_area_factor))
    if crop_sz < 1:
        raise ValueError("Too small bounding box")

    x1 = round(x + 0.5 * w - crop_sz * 0.5)
    x2 = x1 + crop_sz
    y1 = round(y + 0.5 * h - crop_sz * 0.5)
    y2 = y1 + crop_sz

    x1_pad = max(0, -x1)
    x2_pad = max(x2 - image.shape[3] + 1, 0)
    y1_pad = max(0, -y1)
    y2_pad = max(y2 - image.shape[2] + 1, 0)

    img_crop = image[:, :, y1 + y1_pad:y2 - y2_pad, x1 + x1_pad:x2 - x2_pad]
    img_crop = torch.nn.functional.pad(
      img_crop,
      (x1_pad, x2_pad, y1_pad, y2_pad),
      mode="constant",
      value=0,
    )

    resize_factor = output_sz / crop_sz
    img_crop = torch.nn.functional.interpolate(
      img_crop,
      size=(output_sz, output_sz),
      mode="bilinear",
      align_corners=False,
    )

    crop_box = torch.tensor([x1, y1, crop_sz, crop_sz], dtype=torch.float32, device=image.device)
    return img_crop, resize_factor, crop_box


def _map_bbox_to_original(normalized_bbox, search_size, crop_box, resize_factor):
    cx = normalized_bbox[0] * search_size
    cy = normalized_bbox[1] * search_size
    w = normalized_bbox[2] * search_size
    h = normalized_bbox[3] * search_size

    crop_x, crop_y, _, _ = crop_box
    cx = cx / resize_factor + crop_x
    cy = cy / resize_factor + crop_y
    w = w / resize_factor
    h = h / resize_factor

    return torch.stack([cx, cy, w, h], dim=-1)

