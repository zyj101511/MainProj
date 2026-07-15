import cv2
import numpy as np
import torch

from lib.test.data.utils.base_video_dataset import BaseSeqDataset



class FE108Dataset(BaseSeqDataset):
    """ Base class for video datasets """

    def __init__(self, root, split: str, T=1):
        super().__init__(root, split)
        self.meta = self.json_loader(self.lmdb, f"{split}/meta.json")['videos']
        self.T = T

    def __len__(self):
        """
        Returns the number of videos in the dataset
        :return:
        """
        return self._get_num_seqs()

    def __getitem__(self, seq_id):
        seq_len = self._get_num_frames(seq_id)

        search_array = self._get_frames(seq_id, list(range(seq_len)), T=self.T)
        search_anno_array = self._get_annos(seq_id, list(range(seq_len)))

        data = {
            'search': torch.from_numpy(search_array).float()/255.0,
            'search_anno': torch.from_numpy(search_anno_array).float()
        }
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
        # (L, 4)
        return gt[frame_ids]

if __name__ == '__main__':
    dataset = FE108Dataset(root='/home/yanjiezhang/Downloads/Dissertation/dataset/FE108_nbinsGTP_lmdb',
                           split='test', T=1)
    seq_id = 1
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
    item = 1
    data = dataset[item]
    print(data.keys())
    print(len(data['search_anno']))
    print(data['search'][0][0].shape)

    L, T, C, H, W = data['search'].shape
    for l, img in enumerate(data['search'].reshape(L * T, C, H, W)):
        img = (img.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        img = np.ascontiguousarray(img)

        gt = data['search_anno'][l].cpu().numpy()
        cv2.rectangle(img, (int(gt[0]), int(gt[1])),
                      (int(gt[0]) + int(gt[2]), int(gt[1]) + int(gt[3])), (0, 255, 0), 1)

        for gt_next in data['search_anno'][l + 1:l + 5]:
            gt_next = gt_next.cpu().numpy()
            cv2.circle(img, (int(gt_next[0] + gt_next[2] / 2), int(gt_next[1] + gt_next[3] / 2)),
                       1, (255, 255, 255), -1)

        cv2.imshow('img', img)
        cv2.waitKey(100)
