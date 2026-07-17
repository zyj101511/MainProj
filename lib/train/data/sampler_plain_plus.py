import math
import random

from torch.utils.data import Sampler

try:
    import torch.distributed as dist
except ImportError:
    dist = None


class TrackingSamplerSD(Sampler):
    """
    SDTrack-style causal pair sampling adapted to MainProj's batch_sampler interface.

    Each sampled item is:
        (seq_id, template_frame_id, search_frame_id, T)
    """

    def __init__(self, dataset, batch_size, samples_per_epoch, max_gap, T=1):
        self.dataset = dataset
        self.num_seqs = self.dataset._get_num_seqs()
        self.batch_size = batch_size
        self.samples_per_epoch = samples_per_epoch
        self.max_gap = max_gap
        self.T = T

        if self.samples_per_epoch < self.batch_size:
            raise ValueError(
                f"samples_per_epoch {self.samples_per_epoch} must be >= batch_size {self.batch_size}"
            )

    def __len__(self):
        return self.samples_per_epoch // self.batch_size

    def _sample_visible_ids(
        self,
        visible,
        rng,
        num_ids=1,
        min_id=None,
        max_id=None,
    ):
        if num_ids == 0:
            return []

        if min_id is None or min_id < 0:
            min_id = 0
        if max_id is None or max_id > len(visible):
            max_id = len(visible)
        if min_id >= max_id:
            return None

        valid_ids = [i for i in range(min_id, max_id) if visible[i]]
        if not valid_ids:
            return None

        return rng.choices(valid_ids, k=num_ids)

    def _sample_pair(self, rng):
        while True:
            seq_id = rng.randint(0, self.num_seqs - 1)
            visible = self.dataset._get_visible(seq_id)
            num_frames = len(visible)

            if num_frames < 2:
                continue

            template_frame_id = None
            search_frame_id = None
            gap_increase = 0

            while search_frame_id is None:
                template_ids = self._sample_visible_ids(
                    visible,
                    rng,
                    num_ids=1,
                    min_id=0,
                    max_id=num_frames - 1,
                )
                if template_ids is None:
                    break

                template_frame_id = template_ids[0]
                search_ids = self._sample_visible_ids(
                    visible,
                    rng,
                    num_ids=1,
                    min_id=template_frame_id + 1,
                    max_id=template_frame_id + self.max_gap + gap_increase + 1,
                )
                if search_ids is None:
                    gap_increase += 5
                    if template_frame_id + 1 >= num_frames and gap_increase > self.max_gap:
                        break
                    if template_frame_id + self.max_gap + gap_increase >= num_frames and gap_increase > num_frames:
                        break
                    continue

                search_frame_id = search_ids[0]

            if template_frame_id is not None and search_frame_id is not None:
                return (seq_id, template_frame_id, search_frame_id, self.T)

    def __iter__(self):
        num_batches = len(self)
        rng = random.Random()

        for _ in range(num_batches):
            item_batch = []
            while len(item_batch) < self.batch_size:
                item_batch.append(self._sample_pair(rng))
            yield item_batch


class DistributedTrackingSamplerPP(Sampler):
    """
    Distributed SDTrack-style causal pair sampling adapted to MainProj's batch_sampler interface.
    """

    def __init__(
        self,
        dataset,
        batch_size,
        samples_per_epoch,
        max_gap=200,
        T=1,
        num_replicas=None,
        rank=None,
        drop_last=True,
    ):
        self.dataset = dataset
        self.num_seqs = self.dataset._get_num_seqs()
        self.batch_size = batch_size
        self.samples_per_epoch = samples_per_epoch
        self.max_gap = max_gap
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

    def _sample_visible_ids(
        self,
        visible,
        rng,
        num_ids=1,
        min_id=None,
        max_id=None,
    ):
        if num_ids == 0:
            return []

        if min_id is None or min_id < 0:
            min_id = 0
        if max_id is None or max_id > len(visible):
            max_id = len(visible)
        if min_id >= max_id:
            return None

        valid_ids = [i for i in range(min_id, max_id) if visible[i]]
        if not valid_ids:
            return None

        return rng.choices(valid_ids, k=num_ids)

    def _sample_pair(self, rng):
        while True:
            seq_id = rng.randint(0, self.num_seqs - 1)
            visible = self.dataset._get_visible(seq_id)
            num_frames = len(visible)

            if num_frames < 2:
                continue

            template_frame_id = None
            search_frame_id = None
            gap_increase = 0

            while search_frame_id is None:
                template_ids = self._sample_visible_ids(
                    visible,
                    rng,
                    num_ids=1,
                    min_id=0,
                    max_id=num_frames - 1,
                )
                if template_ids is None:
                    break

                template_frame_id = template_ids[0]
                search_ids = self._sample_visible_ids(
                    visible,
                    rng,
                    num_ids=1,
                    min_id=template_frame_id + 1,
                    max_id=template_frame_id + self.max_gap + gap_increase + 1,
                )
                if search_ids is None:
                    gap_increase += 5
                    if template_frame_id + 1 >= num_frames and gap_increase > self.max_gap:
                        break
                    if template_frame_id + self.max_gap + gap_increase >= num_frames and gap_increase > num_frames:
                        break
                    continue

                search_frame_id = search_ids[0]

            if template_frame_id is not None and search_frame_id is not None:
                return (seq_id, template_frame_id, search_frame_id, self.T)

    def __iter__(self):
        for step_idx in range(self.num_global_steps):
            step_rng = random.Random(self.epoch * 100000 + step_idx * 1000 + self.rank)
            item_batch = []

            while len(item_batch) < self.batch_size:
                item_batch.append(self._sample_pair(step_rng))

            yield item_batch
