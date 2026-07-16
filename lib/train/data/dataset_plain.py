import cv2
import random
import torch
import numpy as np
from lib.train.data.utils.base_video_dataset import BaseSeqDataset
from lib.train.data.utils.preprocessing import Preprocessor_plain


class FE108Dataset(BaseSeqDataset):
    """ Base class for video datasets """

    def __init__(self, root, split: str, search_out_sz=256, template_out_sz=128, scale_factor=4, scale_jitter_factor=0.5,
                 ctr_jitter_factor=0.2, sample_last_template=0.5):
        super().__init__(root, split)
        self.meta = self.json_loader(self.lmdb, f"{split}/meta.json")['videos']
        self.preprocessor = Preprocessor_plain(search_out_sz, template_out_sz, scale_factor, scale_jitter_factor, ctr_jitter_factor)
        self.sample_last_template = sample_last_template

    def __len__(self):
        """
        Returns the number of videos in the dataset
        :return:
        """
        return self._get_num_seqs()

    def __getitem__(self, items: tuple):
        """
        Returns an entire sequence and ground truth or a clip of one sequence and ground truth
        items: (seq_id, frame_start_id, T)
        """
        # print(f"Fetching item: {items}")
        seq_id, frame_start_id, T = items

        search_array = self._get_frames(seq_id, [frame_start_id], T=T)  # (1, T, 3, 260, 346)
        search_anno_array = self._get_annos(seq_id, [frame_start_id]) # (1, 4)
        if frame_start_id > 0:

            if random.random() < self.sample_last_template:
                template_id = frame_start_id - 1
            else:
                template_id = random.randint(0, frame_start_id - 1)
        else:
            template_id = 0

        template_array = self._get_frames(seq_id, [template_id], T=T)  # (1, T, 3, 260, 346)
        template_anno_array = self._get_annos(seq_id, [template_id])
        data = self.preprocessor(search_array, search_anno_array, template_array,
                                 template_anno_array)
        data['search'] = torch.from_numpy(data['search']).float() / 255.0
        data['template'] = torch.from_numpy(data['template']).float() / 255.0
        data['search_anno'] = torch.from_numpy(data['search_anno']).float()
        return data

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
        return gt[frame_ids]  #list of x, y, w, h

if __name__ == '__main__':
    dataset = FE108Dataset(root='/home/yanjiezhang/Downloads/Dissertation/dataset/FE108_nbinsGTP_lmdb',
                           split='train', search_out_sz=256, template_out_sz=128, scale_factor=4,
                           scale_jitter_factor=0.5, ctr_jitter_factor=0.2, sample_last_template=0.1)
    seq_id = 13
    frame_ids = [0, 1, 2]
    print(len(dataset))
    print(dataset._get_seq_name(seq_id))
    print(dataset._get_num_frames(seq_id))
    print(dataset._get_frame_names(seq_id, frame_ids))
    print(dataset._get_frame_keys(seq_id, frame_ids))
    imgs = dataset._get_frames(seq_id, frame_ids)
    gt = dataset._get_annos(seq_id, frame_ids)
    item = (seq_id, 155, 1)
    data = dataset[item]
    print(data.keys())
    print(len(data['template']))
    print(len(data['search_anno']))
    print(data['search'][0][0].shape)
    print(data['template'][0][0].shape)

    L, T, C, H, W = data['search'].shape
    for l, img in enumerate(data['search'].reshape(L * T, C, H, W)):
        img = img.permute(1, 2, 0) # (C, H, W) -> (H, W, C)
        img = np.ascontiguousarray(img)
        gt = data['search_anno'][l]
        cv2.rectangle(img, (int(gt[0]), int(gt[1])),
                      (int(gt[0])+int(gt[2]), int(gt[1])+int(gt[3])), (0, 255, 0), 1)
        for gt in data['search_anno'][l+1:l+5]:
            cv2.circle(img, (int(gt[0]+gt[2]/2), int(gt[1]+gt[3]/2)), 1, (255, 255, 255), -1)
        cv2.imshow('img', img)
        cv2.waitKey(0)