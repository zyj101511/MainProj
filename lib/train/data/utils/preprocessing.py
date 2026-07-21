import numpy as np
import cv2
from sympy.physics.units import sidereal_year


class Preprocessor():
    def __init__(self, search_out_sz, template_out_sz, scale_factor=1.2, scale_jitter_factor=0.1):
        self.search_out_sz = search_out_sz  # list (2), search and template
        self.template_out_sz = template_out_sz
        self.scale_factor = scale_factor
        self.scale_jitter_factor = scale_jitter_factor

    def _box_xyxy_to_cxcywh(self, x):
        x0, y0, x1, y1 = x
        b = np.array([(x0 + x1) / 2, (y0 + y1) / 2,
             (x1 - x0), (y1 - y0)], dtype=np.float32)
        return b

    def _get_outer_rect(self, search_anno_array):
        left_top_x = search_anno_array[:, 0]
        left_top_y = search_anno_array[:, 1]
        right_bottom_x = search_anno_array[:, 0] + search_anno_array[:, 2]
        right_bottom_y = search_anno_array[:, 1] + search_anno_array[:, 3]
        min_x = np.min(left_top_x)
        min_y = np.min(left_top_y)
        max_x = np.max(right_bottom_x)
        max_y = np.max(right_bottom_y)
        cx, cy, w, h = self._box_xyxy_to_cxcywh([min_x, min_y, max_x, max_y])

        side = np.sqrt(w * h)
        return np.array([cx, cy, side, side], dtype=np.float32)

    def _get_jittered_search_box_anno(self, search_anno_array):
        anno = search_anno_array.copy()
        original_box = self._get_outer_rect(anno)
        scale_factor = self.scale_factor + ((2*np.random.rand()-1)*self.scale_jitter_factor)
        scaled_size = original_box[2:4] * scale_factor
        new_w, new_h = scaled_size[0], scaled_size[1]
        new_x = original_box[0] - new_w/2
        new_y = original_box[1] - new_h/2
        jittered_box = np.array([new_x, new_y, new_w, new_h], dtype=np.float32)
        anno[:, 0] = anno[:, 0] - new_x
        anno[:, 1] = anno[:, 1] - new_y
        scale_x = self.search_out_sz / new_w
        scale_y = self.search_out_sz / new_h
        anno[:, 0] = anno[:, 0] * scale_x
        anno[:, 1] = anno[:, 1] * scale_y
        anno[:, 2] = anno[:, 2] * scale_x
        anno[:, 3] = anno[:, 3] * scale_y
        return jittered_box, anno  # (x,y,w,h)

    def _crop_resize_search(self, search_array, jittered_box):
        x, y, w, h = jittered_box

        x1 = int(np.floor(x))
        y1 = int(np.floor(y))
        x2 = int(np.ceil(x + w))
        y2 = int(np.ceil(y + h))

        L, T, C, H, W = search_array.shape
        crop_resized_search = np.zeros((L, T, C, self.search_out_sz, self.search_out_sz),
                       dtype=search_array.dtype)
        for l in range(L):
            for t in range(T):
                img = search_array[l, t]  # [C, H, W]
                img = img.transpose(1, 2, 0)  # [H, W, C]

                # 计算和原图的相交区域
                src_x1 = max(0, x1)
                src_y1 = max(0, y1)
                src_x2 = min(W, x2)
                src_y2 = min(H, y2)

                crop_w = x2 - x1
                crop_h = y2 - y1

                if crop_w <= 0 or crop_h <= 0:
                    continue

                # 先建一个带 padding 的 crop 画布
                crop = np.zeros((crop_h, crop_w, C), dtype=img.dtype)

                # 把原图有效区域拷进去
                dst_x1 = src_x1 - x1
                dst_y1 = src_y1 - y1
                dst_x2 = dst_x1 + (src_x2 - src_x1)
                dst_y2 = dst_y1 + (src_y2 - src_y1)
                if src_x2 <= src_x1 or src_y2 <= src_y1 or dst_x2 <= dst_x1 or dst_y2 <= dst_y1:
                    continue
                crop[dst_y1:dst_y2, dst_x1:dst_x2] = img[src_y1:src_y2,
                src_x1:src_x2]

                # resize 到固定大小
                crop = cv2.resize(crop, (self.search_out_sz, self.search_out_sz),
                                  interpolation=cv2.INTER_LINEAR)

                # 回到 CHW
                crop_resized_search[l, t] = crop.transpose(2, 0, 1)

        return crop_resized_search

    def _crop_resize_template(self, template_array, template_anno_array):
        anno = template_anno_array.copy()
        if anno.ndim == 2:
            anno = anno[0]  # [4]

        x, y, w, h = anno
        x1 = int(np.floor(x))
        y1 = int(np.floor(y))
        x2 = int(np.ceil(x + w))
        y2 = int(np.ceil(y + h))

        # 如果 T>1，只取最后一个子图，但保留 T 维
        if template_array.shape[1] > 1:
            template_array = template_array[:, -1:, ...]  # [1, 1, C, H, W]

        _, T, C, H, W = template_array.shape
        crop_resized_template = np.zeros((1, T, C, self.template_out_sz, self.template_out_sz),
                       dtype=template_array.dtype)

        for t in range(T):
            img = template_array[0, t]  # [C, H, W]
            img = img.transpose(1, 2, 0)  # [H, W, C]

            src_x1 = max(0, x1)
            src_y1 = max(0, y1)
            src_x2 = min(W, x2)
            src_y2 = min(H, y2)

            patch_w = x2 - x1
            patch_h = y2 - y1

            patch = np.zeros((patch_h, patch_w, C), dtype=img.dtype)

            dst_x1 = src_x1 - x1
            dst_y1 = src_y1 - y1
            dst_x2 = dst_x1 + (src_x2 - src_x1)
            dst_y2 = dst_y1 + (src_y2 - src_y1)

            if src_x2 <= src_x1 or src_y2 <= src_y1 or dst_x2 <= dst_x1 or dst_y2 <= dst_y1:
                continue

            patch[dst_y1:dst_y2, dst_x1:dst_x2] = img[src_y1:src_y2,src_x1:src_x2]

            patch = cv2.resize(
                patch,
                (self.template_out_sz, self.template_out_sz),
                interpolation=cv2.INTER_LINEAR
            )

            crop_resized_template[0, t] = patch.transpose(2, 0, 1)

        return crop_resized_template  #(L, 1, C, H, W)


    def __call__(self, search_array, search_anno_array, template_array, template_anno_array):
        # (L, T, C, H, W), (L, 4)
        search_crop_area, search_anno = self._get_jittered_search_box_anno(search_anno_array)
        crop_resized_search = self._crop_resize_search(search_array, search_crop_area)
        crop_resized_template = self._crop_resize_template(template_array, template_anno_array)
        return {"search": crop_resized_search,
                "search_anno": search_anno,
                "template": crop_resized_template}



class Preprocessor_plain():
    def __init__(self, search_out_sz, template_out_sz, scale_factor=4, scale_jitter_factor=0.5, ctr_jitter_factor=0.2):
        self.search_out_sz = search_out_sz  # list (2), search and template
        self.template_out_sz = template_out_sz
        self.scale_factor = scale_factor
        self.scale_jitter_factor = scale_jitter_factor
        self.ctr_jitter_factor = ctr_jitter_factor

    def _box_xyxy_to_cxcywh(self, x):
        x0, y0, x1, y1 = x
        b = np.array([(x0 + x1) / 2, (y0 + y1) / 2,
             (x1 - x0), (y1 - y0)], dtype=np.float32)
        return b

    def _get_jittered_search_box_anno(self, search_anno_array):
        anno = search_anno_array.copy()
        original_box = anno[0]  # xywh

        x, y, w, h = original_box
        cx = x + w / 2
        cy = y + h / 2

        scale_factor = self.scale_factor + ((2 * np.random.rand() - 1) *
                                            self.scale_jitter_factor)
        side = np.sqrt(w * h) * scale_factor

        max_ctr_jitter = self.ctr_jitter_factor * side
        jittered_cx = cx + np.random.uniform(-max_ctr_jitter, max_ctr_jitter)
        jittered_cy = cy + np.random.uniform(-max_ctr_jitter, max_ctr_jitter)

        jittered_x = jittered_cx - side / 2
        jittered_y = jittered_cy - side / 2

        jittered_box = np.array([jittered_x, jittered_y, side, side],
                                dtype=np.float32)

        # 把 GT 映射到 crop 坐标系
        anno[:, 0] = anno[:, 0] - jittered_x
        anno[:, 1] = anno[:, 1] - jittered_y

        scale = self.search_out_sz / side
        anno[:, :4] *= scale

        return jittered_box, anno

    def _crop_resize_search(self, search_array, jittered_box):
        x, y, w, h = jittered_box

        x1 = int(np.floor(x))
        y1 = int(np.floor(y))
        x2 = int(np.ceil(x + w))
        y2 = int(np.ceil(y + h))

        L, T, C, H, W = search_array.shape
        crop_resized_search = np.zeros((L, T, C, self.search_out_sz, self.search_out_sz),
                       dtype=search_array.dtype)
        for l in range(L):
            for t in range(T):
                img = search_array[l, t]  # [C, H, W]
                img = img.transpose(1, 2, 0)  # [H, W, C]

                # 计算和原图的相交区域
                src_x1 = max(0, x1)
                src_y1 = max(0, y1)
                src_x2 = min(W, x2)
                src_y2 = min(H, y2)

                crop_w = x2 - x1
                crop_h = y2 - y1

                if crop_w <= 0 or crop_h <= 0:
                    continue

                # 先建一个带 padding 的 crop 画布
                crop = np.zeros((crop_h, crop_w, C), dtype=img.dtype)

                # 把原图有效区域拷进去
                dst_x1 = src_x1 - x1
                dst_y1 = src_y1 - y1
                dst_x2 = dst_x1 + (src_x2 - src_x1)
                dst_y2 = dst_y1 + (src_y2 - src_y1)
                if src_x2 <= src_x1 or src_y2 <= src_y1 or dst_x2 <= dst_x1 or dst_y2 <= dst_y1:
                    continue
                crop[dst_y1:dst_y2, dst_x1:dst_x2] = img[src_y1:src_y2, src_x1:src_x2]

                # resize 到固定大小
                crop = cv2.resize(crop, (self.search_out_sz, self.search_out_sz),
                                  interpolation=cv2.INTER_LINEAR)

                # 回到 CHW
                crop_resized_search[l, t] = crop.transpose(2, 0, 1)

        return crop_resized_search

    def _crop_resize_template(self, template_array, template_anno_array):
        anno = template_anno_array.copy()
        if anno.ndim == 2:
            anno = anno[0]  # [4]

        x, y, w, h = anno
        x1 = int(np.floor(x))
        y1 = int(np.floor(y))
        x2 = int(np.ceil(x + w))
        y2 = int(np.ceil(y + h))

        # 如果 T>1，只取最后一个子图，但保留 T 维
        if template_array.shape[1] > 1:
            template_array = template_array[:, -1:, ...]  # [1, 1, C, H, W]

        _, T, C, H, W = template_array.shape
        crop_resized_template = np.zeros((1, T, C, self.template_out_sz, self.template_out_sz),
                       dtype=template_array.dtype)

        for t in range(T):
            img = template_array[0, t]  # [C, H, W]
            img = img.transpose(1, 2, 0)  # [H, W, C]

            src_x1 = max(0, x1)
            src_y1 = max(0, y1)
            src_x2 = min(W, x2)
            src_y2 = min(H, y2)

            patch_w = x2 - x1
            patch_h = y2 - y1

            patch = np.zeros((patch_h, patch_w, C), dtype=img.dtype)

            dst_x1 = src_x1 - x1
            dst_y1 = src_y1 - y1
            dst_x2 = dst_x1 + (src_x2 - src_x1)
            dst_y2 = dst_y1 + (src_y2 - src_y1)

            if src_x2 <= src_x1 or src_y2 <= src_y1 or dst_x2 <= dst_x1 or dst_y2 <= dst_y1:
                continue

            patch[dst_y1:dst_y2, dst_x1:dst_x2] = img[src_y1:src_y2,src_x1:src_x2]

            patch = cv2.resize(
                patch,
                (self.template_out_sz, self.template_out_sz),
                interpolation=cv2.INTER_LINEAR
            )

            crop_resized_template[0, t] = patch.transpose(2, 0, 1)

        return crop_resized_template  #(L, 1, C, H, W)


    def __call__(self, search_array, search_anno_array, template_array, template_anno_array):
        # (L, T, C, H, W), (L, 4)
        search_crop_area, search_anno = self._get_jittered_search_box_anno(search_anno_array)
        crop_resized_search = self._crop_resize_search(search_array, search_crop_area)
        crop_resized_template = self._crop_resize_template(template_array, template_anno_array)
        return {"search": crop_resized_search,
                "search_anno": search_anno,
                "template": crop_resized_template}

class Preprocessor_pp(Preprocessor_plain):

    def __init__(
        self,
        search_out_sz,
        template_out_sz,
        scale_factor=4,
        scale_jitter_factor=0.5,
        ctr_jitter_factor=0.2,
        template_scale_factor=2.0,
        template_center_jitter=0.0,
        template_scale_jitter=0.0,
    ):
        super().__init__(
            search_out_sz,
            template_out_sz,
            scale_factor=scale_factor,
            scale_jitter_factor=scale_jitter_factor,
            ctr_jitter_factor=ctr_jitter_factor,
        )
        self.search_area_factor = {
            'search': float(scale_factor),
            'template': float(template_scale_factor),
        }
        self.output_sz = {
            'search': int(search_out_sz),
            'template': int(template_out_sz),
        }
        self.center_jitter_factor = {
            'search': float(ctr_jitter_factor),
            'template': float(template_center_jitter),
        }
        self.scale_jitter_factor_dict = {
            'search': float(scale_jitter_factor),
            'template': float(template_scale_jitter),
        }

    def _get_jittered_box(self, box, mode):
        box = np.asarray(box, dtype=np.float32)
        jittered_size = box[2:4] * np.exp(np.random.randn(2).astype(np.float32) * self.scale_jitter_factor_dict[mode])
        max_offset = np.sqrt(max(jittered_size[0] * jittered_size[1], 1e-6)) * self.center_jitter_factor[mode]
        jittered_center = box[0:2] + 0.5 * box[2:4] + max_offset * (np.random.rand(2).astype(np.float32) - 0.5)
        return np.concatenate((jittered_center - 0.5 * jittered_size, jittered_size)).astype(np.float32)

    def _sample_target(self, img_chw, target_bb, search_area_factor, output_sz):
        img = img_chw.transpose(1, 2, 0)
        x, y, w, h = [float(v) for v in target_bb]
        crop_sz = int(np.ceil(np.sqrt(max(w * h, 1e-6)) * search_area_factor))
        if crop_sz < 1:
            raise ValueError('Too small bounding box')

        x1 = round(x + 0.5 * w - crop_sz * 0.5)
        x2 = x1 + crop_sz
        y1 = round(y + 0.5 * h - crop_sz * 0.5)
        y2 = y1 + crop_sz

        x1_pad = max(0, -x1)
        x2_pad = max(x2 - img.shape[1] + 1, 0)
        y1_pad = max(0, -y1)
        y2_pad = max(y2 - img.shape[0] + 1, 0)

        img_crop = img[y1 + y1_pad:y2 - y2_pad, x1 + x1_pad:x2 - x2_pad, :]
        img_crop_padded = cv2.copyMakeBorder(img_crop, y1_pad, y2_pad, x1_pad, x2_pad, cv2.BORDER_CONSTANT)

        resize_factor = output_sz / crop_sz
        img_crop_padded = cv2.resize(img_crop_padded, (output_sz, output_sz), interpolation=cv2.INTER_LINEAR)
        return img_crop_padded.transpose(2, 0, 1), resize_factor

    def _transform_image_to_crop(self, box_in, box_extract, resize_factor, crop_sz):
        box_in = np.asarray(box_in, dtype=np.float32)
        box_extract = np.asarray(box_extract, dtype=np.float32)
        crop_sz = float(crop_sz)
        box_extract_center = box_extract[0:2] + 0.5 * box_extract[2:4]
        box_in_center = box_in[0:2] + 0.5 * box_in[2:4]
        box_out_center = (crop_sz - 1.0) / 2.0 + (box_in_center - box_extract_center) * resize_factor
        box_out_wh = box_in[2:4] * resize_factor
        return np.concatenate((box_out_center - 0.5 * box_out_wh, box_out_wh)).astype(np.float32)

    def _jittered_center_crop(self, frame_array, gt_array, mode):
        box_extract = [self._get_jittered_box(gt_array[i], mode) for i in range(len(gt_array))]
        crops = np.zeros((frame_array.shape[0], frame_array.shape[1], frame_array.shape[2], self.output_sz[mode], self.output_sz[mode]), dtype=frame_array.dtype)
        box_crop = []

        for l in range(frame_array.shape[0]):
            resize_factor = None
            for t in range(frame_array.shape[1]):
                crop, resize_factor = self._sample_target(
                    frame_array[l, t], box_extract[l], self.search_area_factor[mode], self.output_sz[mode]
                )
                crops[l, t] = crop
            box_crop.append(self._transform_image_to_crop(gt_array[l], box_extract[l], resize_factor, self.output_sz[mode]))

        return crops, np.asarray(box_crop, dtype=np.float32)

    def _is_valid_sample(self, search_anno):
        x, y, w, h = [float(v) for v in search_anno[0]]
        crop_sz = self.output_sz['search']
        if w < 1.0 or h < 1.0:
            return False

        cx = x + 0.5 * w
        cy = y + 0.5 * h
        if not (0.0 <= cx <= crop_sz and 0.0 <= cy <= crop_sz):
            return False

        x1 = max(0.0, x)
        y1 = max(0.0, y)
        x2 = min(float(crop_sz), x + w)
        y2 = min(float(crop_sz), y + h)
        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        area = max(w * h, 1e-6)
        if inter / area < 0.25:
            return False
        return True

    def __call__(self, search_array, search_anno_array, template_array, template_anno_array):
        template_crop, template_box = self._jittered_center_crop(template_array, template_anno_array, 'template')
        search_crop, search_box = self._jittered_center_crop(search_array, search_anno_array, 'search')
        valid = self._is_valid_sample(search_box)
        return {
            'search': search_crop,
            'search_anno': search_box,
            'template': template_crop,
            'valid': valid,
        }
