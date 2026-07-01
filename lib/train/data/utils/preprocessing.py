import numpy as np
import cv2

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
        return self._box_xyxy_to_cxcywh([min_x, min_y, max_x, max_y])

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

                # 先建一个带 padding 的 crop 画布
                crop = np.zeros((crop_h, crop_w, C), dtype=img.dtype)

                # 把原图有效区域拷进去
                dst_x1 = src_x1 - x1
                dst_y1 = src_y1 - y1
                dst_x2 = dst_x1 + (src_x2 - src_x1)
                dst_y2 = dst_y1 + (src_y2 - src_y1)

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

            patch[dst_y1:dst_y2, dst_x1:dst_x2] = img[src_y1:src_y2,
            src_x1:src_x2]

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

