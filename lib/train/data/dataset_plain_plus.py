import cv2
import random
import numpy as np
import torch

from lib.train.data.utils.base_video_dataset import BaseSeqDataset
from lib.train.data.utils.preprocessing import Preprocessor_pp


class FE108DatasetPP(BaseSeqDataset):

    def __init__(
        self,
        root,
        split: str,
        search_out_sz=256,
        template_out_sz=128,
        scale_factor=4,
        scale_jitter_factor=0.5,
        ctr_jitter_factor=0.2,
    ):
        super().__init__(root, split)
        self.meta = self.json_loader(self.lmdb, f"{split}/meta.json")["videos"]
        self.preprocessor = Preprocessor_pp(
            search_out_sz,
            template_out_sz,
            scale_factor,
            scale_jitter_factor,
            ctr_jitter_factor,
        )

    def __len__(self):
        return self._get_num_seqs()

    def __getitem__(self, items: tuple):
        """
        items: (seq_id, template_frame_id, search_frame_id, T)
        """
        seq_id, template_frame_id, search_frame_id, T = items
        data = None

        for _ in range(20):
            search_array = self._get_frames(seq_id, [search_frame_id], T=T)
            search_anno_array = self._get_annos(seq_id, [search_frame_id])

            template_array = self._get_frames(seq_id, [template_frame_id], T=T)
            template_anno_array = self._get_annos(seq_id, [template_frame_id])

            data = self.preprocessor(
                search_array,
                search_anno_array,
                template_array,
                template_anno_array,
            )
            if data["valid"]:
                break

            seq_id, template_frame_id, search_frame_id = self._resample_item(seq_id)

        data["search"] = torch.from_numpy(data["search"]).float() / 255.0
        data["template"] = torch.from_numpy(data["template"]).float() / 255.0
        data["search_anno"] = torch.from_numpy(data["search_anno"]).float()
        data.pop("valid", None)
        return data

    def _sample_visible_id(self, visible, min_id=0, max_id=None):
        if max_id is None or max_id > len(visible):
            max_id = len(visible)
        if min_id >= max_id:
            return None
        valid_ids = [i for i in range(min_id, max_id) if visible[i]]
        if not valid_ids:
            return None
        return random.choice(valid_ids)

    def _resample_item(self, fallback_seq_id):
        seq_id = random.randint(0, self._get_num_seqs() - 1)
        visible = self._get_visible(seq_id)

        if visible.sum() < 2:
            seq_id = fallback_seq_id
            visible = self._get_visible(seq_id)

        template_frame_id = self._sample_visible_id(
            visible, min_id=0, max_id=len(visible) - 1)
        if template_frame_id is None:
            return fallback_seq_id, 0, 0

        gap_increase = 0
        search_frame_id = None
        while search_frame_id is None and gap_increase <= len(visible):
            search_frame_id = self._sample_visible_id(
                visible,
                min_id=template_frame_id + 1,
                max_id=template_frame_id + 201 + gap_increase,
            )
            gap_increase += 5

        if search_frame_id is None:
            search_frame_id = min(template_frame_id + 1, len(visible) - 1)

        return seq_id, template_frame_id, search_frame_id


    def _get_seq_name(self, seq_id):
        return self.meta[seq_id]["video_name"]

    def _get_num_seqs(self):
        return len(self.meta)

    def _get_num_frames(self, seq_id):
        return self.meta[seq_id]["num_frames"]

    def _get_frame_names(self, seq_id, frame_ids):
        return [self.meta[seq_id]["frame_names"][frame_id] for frame_id in frame_ids]

    def _get_frame_keys(self, seq_id, frame_ids, T=1):
        frame_names = self._get_frame_names(seq_id, frame_ids)
        seq_name = self._get_seq_name(seq_id)
        frame_keys = [f"{self.split}/{seq_name}/{frame_name}" for frame_name in frame_names]
        return self._get_sub_keys(frame_keys, T)

    def _get_sub_keys(self, frame_keys, T=1):
        if T == 1:
            return frame_keys

        all_keys = []
        for frame_key in frame_keys:
            for t in range(1, T + 1):
                all_keys.append(frame_key + f"_{t:02d}")
        return all_keys

    def _get_frames(self, seq_id, frame_ids, T=1):
        frame_keys = self._get_frame_keys(seq_id, frame_ids, T=T)
        frame_array = np.array(
            [self.image_loader(self.lmdb, frame_key).transpose(2, 0, 1) for frame_key in frame_keys]
        )
        return frame_array.reshape(
            -1, T, frame_array.shape[1], frame_array.shape[2], frame_array.shape[3]
        )

    def _get_annos(self, seq_id, frame_ids):
        gt = self.txt_loader(self.lmdb, f"{self.split}/{self._get_seq_name(seq_id)}/gt.txt")
        return gt[frame_ids]

    def _get_visible(self, seq_id):
        gt = self.txt_loader(self.lmdb, f"{self.split}/{self._get_seq_name(seq_id)}/gt.txt")
        return (gt[:, 2] > 0) & (gt[:, 3] > 0)


if __name__ == "__main__":
    dataset = FE108DatasetPP(
        root="/home/yanjiezhang/Downloads/Dissertation/dataset/FE108_nbinsGTP_lmdb",
        split="train",
        search_out_sz=256,
        template_out_sz=128,
        scale_factor=4,
        scale_jitter_factor=0.5,
        ctr_jitter_factor=0.2,
    )
    item = (13, 120, 155, 1)
    data = dataset[item]
    print(data.keys())
    print(data["search"].shape)
    print(data["template"].shape)
    print(data["search_anno"].shape)

    _, _, C, H, W = data["search"].shape
    img = data["search"][0, 0].permute(1, 2, 0).numpy()
    img = np.ascontiguousarray(img)
    gt = data["search_anno"][0]
    cv2.rectangle(
        img,
        (int(gt[0]), int(gt[1])),
        (int(gt[0]) + int(gt[2]), int(gt[1]) + int(gt[3])),
        (0, 255, 0),
        1,
    )
    cv2.imshow("search", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
