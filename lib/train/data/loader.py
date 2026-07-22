import torch
import importlib
import collections
from lib.utils.tensor_ds import TensorDict, TensorList

from torch.utils.data._utils.collate import default_collate


def mis_collate(batch):
    batch = default_collate(batch)
    batch['search'] = batch['search'].permute(1, 2, 0, 3, 4, 5).contiguous()
    batch['template'] = batch['template'].permute(1, 2, 0, 3, 4, 5).contiguous()  # (L, T, B, C, H, W)
    return batch

class MISLoader(torch.utils.data.dataloader.DataLoader):
    def __init__(self, name, dataset, training=True,batch_size=None, shuffle=False, sampler=None,
                 epoch_interval=1, batch_sampler=None, num_workers=0, collate_fn=mis_collate,
                 pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None, batch_dim=2):
        if batch_sampler is not None:
            super().__init__(
                dataset,
                batch_sampler=batch_sampler,
                num_workers=num_workers,
                collate_fn=collate_fn,
                pin_memory=pin_memory,
                timeout=timeout,
                worker_init_fn=worker_init_fn,
            )
        else:
            super().__init__(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                sampler=sampler,
                num_workers=num_workers,
                collate_fn=collate_fn,
                pin_memory=pin_memory,
                drop_last=drop_last,
                timeout=timeout,
                worker_init_fn=worker_init_fn,
            )

        self.name = name
        self.training = training
        self.epoch_interval = epoch_interval
        self.batch_dim = batch_dim







