import math
from lib.models.backbones.blocks import *
from timm.layers import trunc_normal_
from functools import partial
from lib.models.neuron import MILIF

class Spiking_vit_MetaFormer_Spike_SepConv(nn.Module):
    def __init__(
            self,
            t: int,
            in_channels=3,
            embed_dim:list =None,
            num_heads=8,
            mlp_ratio=4,
            lambda_ratio=4,
            learnable_pad=False,
            cross=False,
            neuron_factory=None
    ):
        super().__init__()
        if learnable_pad:
            self.pad = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.neuron_factory = neuron_factory
        self.embed_dim = embed_dim[-1]  # backbone输出特征维度

        self.downsample1_1 = MS_DownSampling(
            t=t,
            in_channels=in_channels,
            embed_dims=embed_dim[0] // 2,
            kernel_size=7,
            stride=2,
            padding=3,
            first_layer=True,
            neuron_mem=False,
            neuron_factory=self.neuron_factory
        )

        self.ConvBlock1_1 = nn.ModuleList(
            [MS_ConvBlock_spike_SepConv(t=t, dim=embed_dim[0] // 2, mlp_ratio=mlp_ratio, neuron_factory=self.neuron_factory)]
        )

        self.downsample1_2 = MS_DownSampling(
            t=t,
            in_channels=embed_dim[0] // 2,
            embed_dims=embed_dim[0],
            kernel_size=3,
            stride=2,
            padding=1,
            first_layer=False,
            neuron_mem=False,
            neuron_factory=self.neuron_factory
        )

        self.ConvBlock1_2 = nn.ModuleList(
            [MS_ConvBlock_spike_SepConv(t=t, dim=embed_dim[0], mlp_ratio=mlp_ratio, neuron_factory=self.neuron_factory)]
        )

        self.downsample2 = MS_DownSampling(
            t=t,
            in_channels=embed_dim[0],
            embed_dims=embed_dim[1],
            kernel_size=3,
            stride=2,
            padding=1,
            first_layer=False,
            neuron_mem=False,
            neuron_factory=self.neuron_factory
        )

        self.mas_attnmlp_1 = MAS_AttnMLP(
            t=t,
            dim=embed_dim[1],
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            bidirectional=False,
            lambda_ratio=lambda_ratio,
            neuron_factory=self.neuron_factory
        )

        self.ConvBlock2_1 = nn.ModuleList(
            [MS_ConvBlock_spike_SepConv(t=t, dim=embed_dim[1], mlp_ratio=mlp_ratio, neuron_factory=self.neuron_factory)]
        )

        self.ConvBlock2_2 = nn.ModuleList(
            [MS_ConvBlock_spike_SepConv(t=t, dim=embed_dim[1], mlp_ratio=mlp_ratio, neuron_factory=self.neuron_factory)]
        )

        self.downsample3 = MS_DownSampling(
            t=t,
            in_channels=embed_dim[1],
            embed_dims=embed_dim[2],
            kernel_size=3,
            stride=2,
            padding=1,
            first_layer=False,
            neuron_mem=False,
            neuron_factory=self.neuron_factory
        )

        self.mas_attnmlp_2 = MAS_AttnMLP(
            t=t,
            dim=embed_dim[2],
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            bidirectional=False,
            lambda_ratio=lambda_ratio,
            neuron_factory=self.neuron_factory
        )

        self.block3 = nn.ModuleList(
            [
                MS_Block_Spike_AttnMLP(
                    t=t,
                    dim=embed_dim[2],
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    cross=cross,
                    lambda_ratio=lambda_ratio,
                    neuron_factory=self.neuron_factory
                )
                for _ in range(6)
            ]
        )

        self.downsample4 = MS_DownSampling(
            t=t,
            in_channels=embed_dim[2],
            embed_dims=embed_dim[3],
            kernel_size=3,
            stride=1,
            padding=1,
            first_layer=False,
            neuron_mem=False,
            neuron_factory=self.neuron_factory
        )

        self.mas_attnmlp_3 = MS_Block_Spike_AttnMLP(
            t=t,
            dim=embed_dim[3],
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            cross=True,
            bidirectional=False,
            lambda_ratio=lambda_ratio,
            neuron_factory=self.neuron_factory
        )

        self.block4 = nn.ModuleList(
            [
                MS_Block_Spike_AttnMLP(
                    t=t,
                    dim=embed_dim[3],
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    cross=cross,
                    lambda_ratio=lambda_ratio,
                    neuron_factory=self.neuron_factory
                )
                for _ in range(2)
            ]
        )

        self.time_fuse = TimeFuse_Block(
            t=t,
            in_channels=embed_dim[3],
            out_channels=embed_dim[3],
            kernel_size=(t, 1, 1),
            stride=1,
            padding=0,
            bias=False,
            neuron_factory=self.neuron_factory
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

    def create_graph(self, search, template):
        # (Ts, B, C, Hs, Ws)
        # (Ts, B, C, Ht, Wt)
        Ts, B, C, searchH, searchW = search.shape
        Tt, B, C, templateH, templateW = template.shape
        if Tt == 1 and Ts > 1:
            template = template.expand(Ts, B, C, templateH, templateW)
        else:
            assert Tt == Ts, f"Template and search must have the same temporal dimension, but got {Tt} and {Ts}."

        H = searchH + templateH
        W = searchW + templateW

        if hasattr(self, "pad"):
            canvas = self.pad.view(1, 1, 1, 1, 1).expand(Ts, B, C, H, W).clone()
        else:
            canvas = torch.zeros((Ts, B, C, H, W),  device=search.device, dtype=search.dtype)
        canvas[..., :searchH, :searchW] = search
        start = searchH
        canvas[..., start:, start:] = template

        return canvas  # (Ts, B, C, H, W)

    def split_graph(self, cat_graph, search_size, template_size):
        return (cat_graph[..., :search_size, :search_size],
                cat_graph[..., -template_size:, -template_size:])

    def forward_features(self, x):  # conv feature extractor
        x = self.downsample1_1(x)
        for blk in self.ConvBlock1_1:
            x = blk(x)
        x = self.downsample1_2(x)
        for blk in self.ConvBlock1_2:
            x = blk(x)

        x = self.downsample2(x)
        x = self.mas_attnmlp_1(x)

        for blk in self.ConvBlock2_1:
            x = blk(x)
        for blk in self.ConvBlock2_2:
            x = blk(x)

        x = self.downsample3(x)
        x = self.mas_attnmlp_2(x)

        return x

    def forward_features_transformer(self, x):  # x[T, B, 256, 18, 18]

        for blk in self.block3:
            x = blk(x)  # output[T, B, 256, 18, 18]

        # 在这里应该拆分，先把最后四个token提取出来，再将template和search从对应位置还原，分别过downsample再拼接
        T, B, C, H, W = x.shape
        x_all = x.reshape(T, B, C, H * W)  # 包含search和template的整个特征图[T, B, 256, 324]

        search = x_all[..., :256]  # [T, B, 256, 256]
        _, _, _, Ns = search.shape
        search = search.reshape(T, B, C, math.isqrt(Ns), math.isqrt(Ns))  # # [T, B, 256, 16, 16]

        template = x_all[..., 256:320]  # [T, B, 256, 64]
        _, _, _, Nt = template.shape
        template = template.reshape(T, B, C, math.isqrt(Nt), math.isqrt(Nt))  # [T, B, 256, 8, 8]

        appendix_token = x_all[..., 320:]  # [T, B, 256, 4]
        _, _, _, N = appendix_token.shape
        appendix_token = appendix_token.reshape(T, B, C, math.isqrt(N), math.isqrt(N))

        ds_search = self.downsample4(search)  # [T, B, 320, 16, 16]
        ds_template = self.downsample4(template)  # [T, B, 320, 8, 8]
        ds_appendix_token = self.downsample4(appendix_token)  # [T, B, 320, 2, 2]

        search_token = ds_search.flatten(3)  # [T, B, 320, 256]
        template_token = ds_template.flatten(3)  # [T, B, 320, 64]
        appendix_token = ds_appendix_token.flatten(3)  # [T, B, 320, 4]

        cat_feat = torch.cat((search_token, template_token, appendix_token), dim=3)  # [T, B, 320, 324]

        T, B, C, N = cat_feat.shape

        cat_feat = cat_feat.reshape(T, B, C, math.isqrt(N), math.isqrt(N))

        cat_feat = self.mas_attnmlp_3(cat_feat)  # [T, B, 320, 18, 18]

        for blk in self.block4:
            cat_feat = blk(cat_feat)

        return cat_feat  # (T, B, C, H, W)

    def forward(self, search, template):
        # x [T, B, 3, 256, 256]            z [T, B, 3, 128, 128]

        cat_feat = self.create_graph(search, template)  # [T, B, 3, 384, 384]

        cat_feat = self.forward_features(cat_feat)  # [T, B, 256, 24, 24]

        search, template = self.split_graph(cat_feat, 16, 8)  # [T, B, 256, 16, 16]    [B, 256, 8, 8]

        search = search.flatten(3)  # [T, B, 256, 256]
        template = template.flatten(3)  # [T, B, 256, 64]

        canvas = torch.cat((search, template), dim=3)  # [T, B, 256, 320]
        T, B, C, HW = canvas.shape
        canvas = torch.cat([canvas, torch.zeros(T, B, C, 4, device=search.device, dtype=search.dtype)], dim=3)
        T, B, C, HW = canvas.shape
        canvas = canvas.reshape(T, B, C, math.isqrt(HW), math.isqrt(HW)) # [T, B, 256, 18, 18]

        x = self.forward_features_transformer(canvas)  # [T, B, 320, 18, 18]
        x = self.time_fuse(x)

        y = x.flatten(2)
        y = y[..., :320]

        return y  # [B, C, 320](B,C,N)

    def reset_neurons(self):
        for m in self.modules():
            if isinstance(m, MILIF):
                m.reset()

def build_backbone_tiny(t):
    # when t = 1
    # 2,197,497 with pad
    # 2,197,577 with learnable decay and pad
    model = Spiking_vit_MetaFormer_Spike_SepConv(
        t=t,
        in_channels=3,
        embed_dim=[24, 48, 96, 128],
        num_heads=8,
        mlp_ratio=4,
        lambda_ratio=4,
        learnable_pad = True,
        cross=False,
        neuron_factory=ILIF_layer,
    )
    return model

def build_backbone_small(t):
    # when t = 1
    # 4,192,225 with pad
    # 4,192,305 with learnable decay and pad
    model = Spiking_vit_MetaFormer_Spike_SepConv(
        t=t,
        in_channels=3,
        embed_dim=[32, 64, 128, 192],
        num_heads=8,
        mlp_ratio=4,
        lambda_ratio=4,
        learnable_pad = True,
        cross=False,
        neuron_factory=ILIF_layer,
    )
    return model

def build_backbone_medium(t):
    # when t = 1
    # 8,390,881 with pad
    # 8,390,961 with learnable decay and pad
    model = Spiking_vit_MetaFormer_Spike_SepConv(
        t=t,
        in_channels=3,
        embed_dim=[48, 96, 192, 240],
        num_heads=8,
        mlp_ratio=4,
        lambda_ratio=4,
        learnable_pad = True,
        cross=False,
        neuron_factory=ILIF_layer,
    )
    return model

def build_backbone_large(t):
    # when t = 1
    # 14,879,873 with pad
    # 14,879,953 with learnable decay and pad
    model = Spiking_vit_MetaFormer_Spike_SepConv(
        t=t,
        in_channels=3,
        embed_dim=[64, 128, 256, 360],
        num_heads=8,
        mlp_ratio=4,
        lambda_ratio=4,
        learnable_pad = True,
        cross=False,
        neuron_factory=ILIF_layer,
    )
    return model

ILIF_layer = partial(MILIF,
                     min_v=0.,
                     max_v=4.0,
                     norm=None,
                     t=None,
                     decay=False,
                     decay_rate=0.25,
                     state_clip=(-4, 4),
                     learnable_decay=True,
                     mem=False,
                     infere_mode=False,
                     detach_reset=True,
                     store_v_seq=False,
                     reset_mode='hard')

