import cv2
import numpy as np
from lib.train.data.utils.base_video_dataset import BaseSeqDataset
from lib.train.data.utils.preprocessing import Preprocessor


class FE108Dataset(BaseSeqDataset):
    """ Base class for video datasets """

    def __init__(self, root, split: str, search_out_sz, template_out_sz, scale_factor, scale_jitter_factor):
        super().__init__(root, split)
        self.meta = self.json_loader(self.lmdb, f"{split}/meta.json")['videos']
        self.preprocessor = Preprocessor(search_out_sz, template_out_sz, scale_factor, scale_jitter_factor)

    def __len__(self):
        """
        Returns the number of videos in the dataset
        :return:
        """
        return self._get_num_seqs()

    def __getitem__(self, items: tuple):
        """
        Returns an entire sequence and ground truth or a clip of one sequence and ground truth
        items: (seq_id, frame_start_id, L, P, distance_factor, T)
                L is the length of the clip, P is the step of predictions
        """
        seq_id, frame_start_id, l, p, df, T = items
        if l < 1 or p*df < l:
            raise ValueError(f'clip length must be >= 1, and (prediction length * distance factor) must be >= clip length, but got {l} and {p*df}')
        if df < 1:
            raise ValueError(f"distance_factor should be >= 1, but got {df}")
        if frame_start_id + l + p > self._get_num_frames(seq_id) or frame_start_id + l + df * p > self._get_num_frames(seq_id):
            raise ValueError(f"(start+l+p) and (start+l+distance_factor*p) should be <= number of frames in the sequence, "
                             f"but got {frame_start_id + l + p}, {frame_start_id + l + df * p} and number of frames={self._get_num_frames(seq_id)}")
        search_array = self._get_frames(seq_id, list(range(frame_start_id, frame_start_id + l)), T=T)  # (L, T, 3, 260, 346)
        search_anno_array = self._get_annos(seq_id, list(range(frame_start_id, frame_start_id + (l+df*p))))
        if frame_start_id > 0:
            template_array = self._get_frames(seq_id, [frame_start_id-1], T=T)  # (L, T, 3, 260, 346)
            template_anno_array = self._get_annos(seq_id, [frame_start_id-1])
        else:
            template_array = self._get_frames(seq_id, [0], T=T)
            template_anno_array = self._get_annos(seq_id, [0])
        return self.preprocessor(search_array, search_anno_array, template_array, template_anno_array)

    def _get_seq_name(self, seq_id):
        """ Name of the sequence
        returns:
            string - Name of the sequence
        """
        return self.meta[seq_id]['video_name']

    def _get_num_seqs(self):
        """ Number of sequences in a dataset

        returns:
            int - number of sequences in the dataset."""
        return len(self.meta)

    def _get_num_frames(self, seq_id):
        """
        Total number of frames in a single sequence
        """
        return self.meta[seq_id]['num_frames']

    def _get_frame_names(self, seq_id, frame_ids):
        """
        get name of the frames
        :return:
        """
        frame_names = [self.meta[seq_id]['frame_names'][frame_id] for frame_id in frame_ids]
        return frame_names

    def _get_frame_keys(self, seq_id, frame_ids, T=1):
        frame_names = self._get_frame_names(seq_id, frame_ids)
        seq_name = self._get_seq_name(seq_id)
        frame_keys = [f'{self.split}/{seq_name}/{frame_name}' for frame_name in frame_names]
        all_keys = self._get_sub_keys(frame_keys, T)
        return all_keys

    def _get_sub_keys(self, frame_keys, T=1):
        all_keys = []
        if T == 1:
            return frame_keys
        for frame_key in frame_keys:
            for t in range(1, T+1):
                all_keys.append(frame_key + f'_{t:02d}')
        return all_keys

    def _get_frames(self, seq_id, frame_ids, T=1):
        frame_keys = self._get_frame_keys(seq_id, frame_ids, T=T)
        frame_array = np.array([self.image_loader(self.lmdb, frame_key).transpose(2, 0, 1) for frame_key in frame_keys])
        # (L, T, C, H, W)
        return frame_array.reshape(-1, T, frame_array.shape[1], frame_array.shape[2], frame_array.shape[3])

    def _get_annos(self, seq_id, frame_ids):
        gt = self.txt_loader(self.lmdb, f"{self.split}/{self._get_seq_name(seq_id)}/gt.txt")
        # (L+P*df, 4)
        return gt[frame_ids]

if __name__ == '__main__':
    dataset = FE108Dataset(root='/home/yanjiezhang/Downloads/Dissertation/dataset/FE108_nbinsGTP_lmdb',
                           split='train')
    seq_id = 0
    frame_ids = [0, 1, 2]
    print(len(dataset))
    print(dataset._get_seq_name(seq_id))
    print(dataset._get_num_frames(seq_id))
    print(dataset._get_frame_names(seq_id, frame_ids))
    print(dataset._get_frame_keys(seq_id, frame_ids))
    imgs = dataset._get_frames(seq_id, frame_ids)
    gt = dataset._get_annos(seq_id, frame_ids)
    for idx, img in enumerate(imgs):
        print(img.shape)
        img = img[0].transpose(1, 2, 0) # (C, H, W) -> (H, W, C)
        img = np.ascontiguousarray(img)
        cv2.rectangle(img, (int(gt[idx][0]), int(gt[idx][1])),
                      (int(gt[idx][0])+int(gt[idx][2]), int(gt[idx][1])+int(gt[idx][3])), (0, 255, 0), 1)
        cv2.imshow('img', img)
        cv2.waitKey(0)
    cv2.destroyAllWindows()
    item = (seq_id, 0, 5, 5, 4, 1)
    data = dataset[item]
    print(len(data['frames']))
    print(len(data['annos']))