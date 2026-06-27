import os
from collections import OrderedDict
try:
    from torch.utils.tensorboard import SummaryWriter
except:
    print('WARNING: You are using tensorboardX instead sis you have a too old pytorch version.')
    from tensorboardX import SummaryWriter


class TensorboardWriter:
    def __init__(self, directory, loader_names):
        # 这里为不同的数据集划分创建不同的 TensorBoard writer。比如：loader_names = ["train", "val"]
        # 那么它会创建：directory/train,directory/val
        self.directory = directory
        self.writer = OrderedDict({name: SummaryWriter(os.path.join(self.directory, name))
                                   for name in loader_names})

    def write_info(self, script_name, description):
        # 是写一些文本信息到TensorBoard里
        tb_info_writer = SummaryWriter(os.path.join(self.directory, 'info'))
        tb_info_writer.add_text('Script_name', script_name)
        tb_info_writer.add_text('Description', description)
        tb_info_writer.close()

    # 是写一些文本信息到 TensorBoard 里
    def write_epoch(self, stats: OrderedDict, epoch: int, ind=-1):
        for loader_name, loader_stats in stats.items():
            if loader_stats is None:
                continue
            for var_name, val in loader_stats.items():
                if hasattr(val, 'history') and getattr(val, 'has_new_data', True):
                    # add_scalar(图的名字, y轴数值, x轴步数)
                    # stats里放的是stats.py实现的类的实例
                    # 比如有的metric或者val几个epoch才计算一次, 那中间不计算的时候has_new_data就是False, 不会写入tensorboard
                        self.writer[loader_name].add_scalar(var_name, val.history[ind], epoch)
                '''
                  stats = OrderedDict({
                                      "train": OrderedDict({
                                          "loss": AverageMeter(),
                                          "acc": AverageMeter(),
                                          "lr": StatValue(),
                                      }),
                                      "val": OrderedDict({
                                          "loss": AverageMeter(),
                                          "acc": AverageMeter(),
                                      })
                                  })
                '''