import random
from torch.utils.data import Sampler
import math
try:
    import torch.distributed as dist
except ImportError:
    dist = None


class TrackingPredSampler(Sampler):
    """
    A batch sampler returns list of [(seq_id, frame_start_id, L, P, distance_factor, T)]
    L is thelength of the clip sequence, P is the steps of predictions
    """
    def __init__(self, dataset, batch_size, samples_per_epoch, L_candidates, P, distance_factor, T=1):
        self.dataset = dataset
        self.num_seqs = self.dataset._get_num_seqs()
        self.batch_size = batch_size
        self.samples_per_epoch = samples_per_epoch
        if self.samples_per_epoch < self.batch_size:
            raise ValueError(f'samples_per_epoch {self.samples_per_epoch} must be >= batch_size{self.batch_size}')
        self.L_candidates = L_candidates
        self.P = P
        self.df = distance_factor
        self.T = T

    def __len__(self):
        return self.samples_per_epoch // self.batch_size

    def __iter__(self):
        num_batches = len(self)

        for _ in range(num_batches):
            # 同一个batch内, L必须一致
            L = random.choice(self.L_candidates)
            item_batch = []

            while len(item_batch) < self.batch_size:
                seq_id = random.randint(0, self.num_seqs - 1)
                num_frames = self.dataset._get_num_frames(seq_id)
                max_start = num_frames - (L + self.P * self.df)
                if max_start < 0:
                    continue
                frame_start_id = random.randint(0, max_start)

                item = (seq_id, frame_start_id, L, self.P, self.df, self.T)
                item_batch.append(item)
            yield item_batch



class DistributedTrackingPredSampler(Sampler):
    """
    A distributed batch sampler that returns
    [(seq_id, frame_start_id, L, P, distance_factor, T)] for each local batch.

    Same global step across all ranks uses the same L.
    """

    def __init__(
            self,
            dataset,
            batch_size,
            samples_per_epoch,
            L_candidates,
            P,
            distance_factor,
            T=1,
            num_replicas=None,
            rank=None,
            drop_last=True,
    ):
        self.dataset = dataset
        self.num_seqs = self.dataset._get_num_seqs()
        self.batch_size = batch_size
        self.samples_per_epoch = samples_per_epoch
        self.L_candidates = L_candidates
        self.P = P
        self.df = distance_factor
        self.T = T
        self.drop_last = drop_last
        self.epoch = 0

        if num_replicas is None:
            if dist is not None and dist.is_available() and dist.is_initialized():
                num_replicas = dist.get_world_size()
            else:
                num_replicas = 1

        if rank is None:
            if dist is not None and dist.is_available() and dist.is_initialized():
                rank = dist.get_rank()
            else:
                rank = 0

        self.num_replicas = num_replicas
        self.rank = rank

        if self.samples_per_epoch < self.batch_size * self.num_replicas:
            raise ValueError(
                f"samples_per_epoch {self.samples_per_epoch} must be >= "
                f"global batch size {self.batch_size * self.num_replicas}"
            )

        global_batch_size = self.batch_size * self.num_replicas
        if self.drop_last:
            self.num_global_steps = self.samples_per_epoch // global_batch_size
        else:
            self.num_global_steps = math.ceil(self.samples_per_epoch / global_batch_size)

    def __len__(self):
        return self.num_global_steps

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        rng = random.Random(self.epoch)

        # Pre-sample one L per global step.
        step_Ls = [rng.choice(self.L_candidates) for _ in range(self.num_global_steps)]

        for step_idx in range(self.num_global_steps):
            L = step_Ls[step_idx]
            item_batch = []

            # Step-specific RNG so all ranks use same L but sample different data.
            step_rng = random.Random(self.epoch * 100000 + step_idx * 1000 + self.rank)

            while len(item_batch) < self.batch_size:
                seq_id = step_rng.randint(0, self.num_seqs - 1)
                num_frames = self.dataset._get_num_frames(seq_id)
                max_start = num_frames - (L + self.P * self.df)
                if max_start < 0:
                    continue

                frame_start_id = step_rng.randint(0, max_start)
                item = (seq_id, frame_start_id, L, self.P, self.df, self.T)
                item_batch.append(item)

            yield item_batch

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from lib.train.data.dataset import FE108Dataset
    dataset = FE108Dataset(root='/home/yanjiezhang/Downloads/Dissertation/dataset/FE108_nbinsGTP_lmdb',
                           split='train', search_out_sz=256, template_out_sz=128,
                           scale_factor=1.2, scale_jitter_factor=0.3)

    L_candidates = [100]

    from torch.utils.data._utils.collate import default_collate
    def mas_collate(batch):
        batch = default_collate(batch)
        batch['search'] = batch['search'].permute(1, 2, 0, 4, 5, 3).contiguous()  # (L, T, B, H, W, C)
        batch['template'] = batch['template'].permute(1, 2, 0, 4, 5, 3).contiguous()  # (L, T, B, H, W, C)
        batch['search_anno'] = batch['search_anno'].permute(1, 0, 2).contiguous()  # (L, B, 4)
        return batch
    batch_sampler = TrackingPredSampler(dataset, 1, 100,
                                        L_candidates, P=10, distance_factor=20, T=1)
    train_loader = DataLoader(dataset=dataset, batch_sampler=batch_sampler, collate_fn=mas_collate)
    batch = next(iter(train_loader))
    print(batch.keys())
    print(len(batch))
    print(batch['search'].shape)  # (B, L, T, C, H, W)
    print(batch['search_anno'].shape)  # (B, L, 4)
    print(batch['template'].shape)

    # permuted_batch = batch['search'].permute(1, 2, 0, 4, 5, 3)  # (L, T, B, H, W, C)
    # permuted_anno = batch['search_anno'].permute(1, 0, 2)  # (L, B, 4)
    permuted_batch = batch['search']
    permuted_anno = batch['search_anno']
    import numpy as np
    L, T, B,  = permuted_batch.shape[:3]
    for l in range(L):
        for t in range(T):
            for b in range(B):
                import cv2
                img = permuted_batch[l, t, b].contiguous().numpy()
                print(np.max(img), np.min(img))
                from lib.utils.box_ops import box_xywh_to_xyxy
                anno = box_xywh_to_xyxy(permuted_anno[l, b])

                # 后面10个真实框中心点
                for k in range(1, 11):
                    future_anno = permuted_anno[l + k, b]  # xywh
                    cx = future_anno[0] + future_anno[2] / 2
                    cy = future_anno[1] + future_anno[3] / 2

                    cv2.circle(
                        img,
                        (int(cx), int(cy)),
                        1,
                        (255, 255, 255),
                        -1
                    )
                cv2.rectangle(img, (int(anno[0]), int(anno[1])),
                              (int(anno[2]), int(anno[3])), (0, 255, 0), 1)
                cv2.imshow(f'b', img)
                cv2.waitKey(100)
    cv2.destroyAllWindows()
