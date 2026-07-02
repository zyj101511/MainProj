import torch
from torch import nn
from spikingjelly.activation_based.layer import SeqToANNContainer


class Memory_Branch(nn.Module):
    def __init__(self, t, neuron_factory, in_channels, num_layers=1):
        super().__init__()
        self.neuron_factory = neuron_factory
        self.pwconv_block = nn.ModuleList([
                SeqToANNContainer(
                    nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, bias=False),
                    nn.BatchNorm2d(in_channels)
                )
                for _ in range(num_layers)])
        self.spike_block = nn.ModuleList([
            self.neuron_factory(t=t)
            for _ in range(num_layers)])

    def forward(self, x):
        for conv, spike in zip(self.pwconv_block, self.spike_block):
            x = spike(conv(x)) + x
        return x

class Multi_Timescale_Memory_Block(nn.Module):
    def __init__(self, t, neuron_factory, in_channels, num_branch=4, num_layers=1):
        """
        Multi-Memory Dynamics Block
        :param t:
        :param neuron_factory:
        :param num_branch:
        :param num_layers:
        :param in_channels:
        """
        super().__init__()
        self.neuron_factory = neuron_factory
        # compress T to 1
        self.conv3d1 = nn.Conv3d(in_channels, in_channels, kernel_size=(t,1,1), padding=0, stride=1)  # (B,C,T,H,W)
        self.bn3d1 = nn.BatchNorm3d(in_channels)
        self.spike1 = self.neuron_factory(t=1)
        self.branches = nn.ModuleList([
            Memory_Branch(t=1, neuron_factory=neuron_factory,
                          in_channels=in_channels, num_layers=num_layers)
            for _ in range(num_branch)
        ])


    def forward(self, x):  # (T, B, C, H, W)
        x = x.permute(1, 2, 0, 3, 4)  # (B, C, T, H, W)
        x = self.conv3d1(x)  # (B, C, 1, H, W)
        x = self.bn3d1(x)
        x = x.permute(2, 0, 1, 3, 4)  # (1, B, C, H, W)

        x = self.spike1(x)

        branches_out_list = [branch(x) for branch in self.branches]
        return branches_out_list


class Cross_Timescale_Fuse_Block(nn.Module):
    def __init__(self, in_channels, num_branch=4):
        super().__init__()
        self.num_branch = num_branch

        # 每个分支生成score_map, (1, B, C, H, W) -> (1, B, 1, H, W)
        self.score_net_block = nn.ModuleList([
            SeqToANNContainer(
                nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.Conv2d(in_channels, 1, kernel_size=1, stride=1, bias=False)
            )
            for _ in range(num_branch)
        ])
        self.proj = SeqToANNContainer(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(in_channels)
        )

    def forward(self, branches_out_list):  # branches_out_list: list of (1, B, C, H, W)
        assert len(branches_out_list) == self.num_branch
        branch_scores = []
        for x, score_net in zip(branches_out_list, self.score_net_block):
            score = score_net(x)  # (1, B, 1, H, W)
            score = score.flatten(2)  # (1, B, H*W)
            score = score.mean(dim=-1, keepdim=True)  # (1, B, 1)
            branch_scores.append(score)
        scores = torch.stack(branch_scores, dim=2)  # (1, B, num_branch, 1)
        weights = torch.softmax(scores, dim=2)  # (1, B, num_branch, 1)

        fused_feature = torch.zeros_like(branches_out_list[0])
        for i, x in enumerate(branches_out_list):
            w = weights[:, :, i].unsqueeze(-1).unsqueeze(-1)  # (1, B, 1, 1, 1)
            fused_feature = fused_feature + x * w

        fused_feature = self.proj(fused_feature)
        return fused_feature   # (1, B, C, H, W)