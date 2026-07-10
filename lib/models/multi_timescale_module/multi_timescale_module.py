from torch import nn
from functools import partial
from lib.models.neuron import MILIF
from lib.models.multi_timescale_module.blocks import Cross_Timescale_Fuse_Block, Multi_Timescale_Memory_Block

MILIF_layer = partial(MILIF,
                      min_v=0.,
                      max_v=4.0,
                      norm=None,
                      t=None,
                      decay=True,
                      decay_rate=0.2,
                      state_clip=(-0.5, 4),
                      learnable_decay=True,
                      mem=True,
                      infere_mode=False,
                      detach_reset=True,
                      store_v_seq=False,
                      reset_mode='soft')

class Multi_Timescale_Module(nn.Module):
    def __init__(self, t: int, in_channels=3, num_branch=4, num_layer=1, neuron_factory=MILIF_layer):
        super().__init__()

        self.multi_timescale_memory_block = Multi_Timescale_Memory_Block(t=t, neuron_factory=neuron_factory,
                                                                        in_channels=in_channels,
                                                                        num_branch=num_branch,
                                                                        num_layers=num_layer)
        self.cross_timescale_fuse_block = Cross_Timescale_Fuse_Block(in_channels=in_channels,
                                                                     num_branch=num_branch)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Conv3d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d)):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

    def forward(self, x):  # (T, B, C, H, W)
        branches_out_list = self.multi_timescale_memory_block(x)  # list of (1, B, C, H, W)
        fused_out = self.cross_timescale_fuse_block(branches_out_list)  # (1, B, C, H, W)
        return fused_out.squeeze(0)  # (B, C, H, W)

    def reset_neurons(self):
        for m in self.modules():
            if isinstance(m, MILIF):
                m.reset()

if __name__ == '__main__':
    import torch
    t = 4
    B = 2
    C = 3
    H = 8
    W = 8
    x = torch.rand(t, B, C, H, W)
    multi_timescale_module = build_multi_timescale_module(t=t, in_channels=C, num_branch=4, num_layer=1)
    out = multi_timescale_module(x)
    print(out.shape)  # (B, C, H, W)