import torch.nn as nn
# Here we use DistributedDataParallel(DDP) rather than DataParallel(DP) for multiple GPUs training


def is_multi_gpu(net):
    # 判断net是不是DDP模型
    return isinstance(net, (MultiGPU, nn.parallel.distributed.DistributedDataParallel))

'''
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

local_rank = int(os.environ["LOCAL_RANK"])

torch.cuda.set_device(local_rank)
dist.init_process_group(backend="nccl")

model = MyModel().to(local_rank)

model = DDP(
  model,
  device_ids=[local_rank],
  output_device=local_rank
)
再访问时模型实际再model.module中而不是直接model.
如果再使用MultiGPU包装,就可以直接使用model.
local_rank = int(os.environ["LOCAL_RANK"])

torch.cuda.set_device(local_rank)  # 设置当前进程默认gpu
dist.init_process_group(backend="nccl")

model = MyModel().to(local_rank)

model = MultiGPU(
  model,
  device_ids=[local_rank],
  output_device=local_rank
)
'''

class MultiGPU(nn.parallel.distributed.DistributedDataParallel):
    # 子类没有init, 会自动调用父类init
    def __getattr__(self, item):
        try:
            return super().__getattr__(item)
        except:
            return getattr(self.module, item)
