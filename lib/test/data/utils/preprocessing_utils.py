import torch
import math
import cv2 as cv
import torch.nn.functional as F
import numpy as np

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
    x2 = x1 + crop_sz

    y1 = round(y + 0.5 * h - crop_sz * 0.5)
    y2 = y1 + crop_sz

    x1_limit = max(0, x1)
    x2_limit = min(x2, w_origin)

    y1_limit = max(0, y1)
    y2_limit = min(y2, h_origin)

    return torch.tensor([x1_limit, y1_limit, x2_limit, y2_limit], dtype=torch.float32, device=image.device)


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

