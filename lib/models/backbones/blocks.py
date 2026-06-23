import torch
from torch import nn
from spikingjelly.activation_based.layer import SeqToANNContainer

class SepConv_Spike(nn.Module):
    """
    Modified Inverted separable convolution from MobileNetV2: https://arxiv.org/abs/1801.04381.
    Wrapped with SeqToANNContainer, inputs should have the temporal dimension (T, B, C, H, W).
    """

    def __init__(self, t: int, dim: int, expansion_ratio=2, bias=False, kernel_size=7, padding=3, neuron_factory=None):
        super().__init__()
        self.neuron_factory = neuron_factory
        med_channels = int(expansion_ratio * dim)
        self.spike1 = self.neuron_factory(t=t)
        self.pwconv1 = SeqToANNContainer(
            nn.Conv2d(dim, med_channels, kernel_size=1, stride=1, bias=bias),
            nn.BatchNorm2d(med_channels)
        )
        self.spike2 = self.neuron_factory(t=t)
        self.dwconv = SeqToANNContainer(
            nn.Conv2d(med_channels, med_channels, kernel_size=kernel_size, padding=padding, groups=med_channels,
                      bias=bias),
            nn.BatchNorm2d(med_channels)
        )
        self.spike3 = self.neuron_factory(t=t)
        self.pwconv2 = SeqToANNContainer(
            nn.Conv2d(med_channels, dim, kernel_size=1, stride=1, bias=bias),
            nn.BatchNorm2d(dim)
        )

    def forward(self, x):  # (T, B, C, H, W)
        x = self.spike1(x)

        x = self.pwconv1(x)

        x = self.spike2(x)

        x = self.dwconv(x)

        x = self.spike3(x)

        x = self.pwconv2(x)
        return x  # (T, B, C, H, W)


class MS_ConvBlock_spike_SepConv(nn.Module):
    def __init__(self, t: int, dim: int, mlp_ratio: int=4, neuron_factory=None):
        super().__init__()
        self.neuron_factory = neuron_factory
        self.Conv = SepConv_Spike(dim=dim, t=t, neuron_factory=self.neuron_factory)

        self.hidden_dim = int(dim * mlp_ratio)

        self.spike1 = self.neuron_factory(t=t)
        self.conv1 = SeqToANNContainer(
            nn.Conv2d(dim, self.hidden_dim, kernel_size=3, padding=1, groups=1, bias=False)
        )
        self.bn1 = SeqToANNContainer(
            nn.BatchNorm2d(self.hidden_dim)
        )
        self.spike2 = self.neuron_factory(t=t)
        self.conv2 = SeqToANNContainer(
            nn.Conv2d(self.hidden_dim, dim, kernel_size=3, padding=1, groups=1, bias=False)
        )
        self.bn2 = SeqToANNContainer(
            nn.BatchNorm2d(dim)
        )

    def forward(self, x): # (T, B, C, H, W)
        T, B, C, H, W = x.shape
        x = self.Conv(x) + x  # residual

        x_feat = x

        x = self.spike1(x)

        x = self.bn1(self.conv1(x))

        assert x.shape == (T, B, self.hidden_dim, H, W)

        x = self.spike2(x)

        x = self.bn2(self.conv2(x))

        assert x.shape == (T, B, C, H, W)

        x = x_feat + x  # residual

        return x


class MS_MLP(nn.Module):
    def __init__(self, t: int, in_features: int, hidden_features=None, out_features=None, neuron_factory=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.neuron_factory = neuron_factory

        self.fc1_conv =SeqToANNContainer(
            nn.Conv1d(in_features, hidden_features, kernel_size=1, stride=1)
        )
        self.fc1_bn = SeqToANNContainer(
            nn.BatchNorm1d(hidden_features)
        )
        self.fc1_spike =  self.neuron_factory(t=t)
        self.fc2_conv = SeqToANNContainer(
            nn.Conv1d(hidden_features, out_features, kernel_size=1, stride=1)
        )
        self.fc2_bn = SeqToANNContainer(
            nn.BatchNorm1d(out_features)
        )
        self.fc2_spike = self.neuron_factory(t=t)

        self.c_hidden = hidden_features
        self.c_output = out_features

    def forward(self, x):  # (T, B, C, H, W)
        T, B, C, H, W = x.shape
        N = H * W  # the number of tokens
        x = x.flatten(3)  # (T, B, C, N)
        x = self.fc1_spike(x)
        x = self.fc1_conv(x)
        x = self.fc1_bn(x)
        assert x.shape == (T, B, self.c_hidden, N)

        x = self.fc2_spike(x)
        x = self.fc2_conv(x)
        assert x.shape == (T, B, self.c_output, N)
        x = self.fc2_bn(x).reshape(T, B, self.c_output, H, W).contiguous()

        return x


class MS_Attention_linear(nn.Module):
    def __init__(self, t: int, dim: int, num_heads: int=8, lambda_ratio=1, neuron_factory=None):
        super().__init__()
        assert (dim % num_heads == 0), f"dim {dim} should be divided by num_heads {num_heads}."
        self.neuron_factory = neuron_factory
        self.dim = dim
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.lambda_ratio = lambda_ratio
        self.C_v = int(dim * self.lambda_ratio)
        assert self.C_v % self.num_heads == 0

        self.head_spike = self.neuron_factory(t=t)

        self.q_conv = SeqToANNContainer(
            nn.Conv2d(dim, dim, 1, 1, bias=False),
            nn.BatchNorm2d(dim)
        )

        self.q_spike = self.neuron_factory(t=t)

        self.k_conv = SeqToANNContainer(
            nn.Conv2d(dim, dim, 1, 1, bias=False),
            nn.BatchNorm2d(dim)
        )

        self.k_spike = self.neuron_factory(t=t)

        self.v_conv = SeqToANNContainer(
            nn.Conv2d(dim, self.C_v, 1, 1, bias=False),
            nn.BatchNorm2d(self.C_v)
        )

        self.v_spike = self.neuron_factory(t=t)

        self.attn_spike = self.neuron_factory(t=t)

        self.proj_conv = SeqToANNContainer(
            nn.Conv2d(self.C_v, dim, 1, 1, bias=False),
            nn.BatchNorm2d(dim)
        )

    def forward(self, x):
        T, B, C, H, W = x.shape
        assert C == self.dim

        N = H * W
        x = self.head_spike(x)

        q = self.q_conv(x)  # (T, B, C, H, W)
        k = self.k_conv(x)  # (T, B, C, H, W)
        v = self.v_conv(x)  # (T, B, C_v, H, W)

        q = self.q_spike(q)  # (T, B, C, H, W)
        q = q.flatten(3)  # (T, B, C, N), N is the number of token
        q = (
            q.transpose(-1, -2)  # (T, B, N, C)
            .reshape(T, B, N, self.num_heads, C // self.num_heads)  # (T, B, N, num_heads, head_dim_qv)
            .permute(0, 1, 3, 2, 4)  # (T, B, num_heads, N, head_dim_qv)
            .contiguous()
        )

        k = self.k_spike(k)
        k = k.flatten(3)
        k = (
            k.transpose(-1, -2)
            .reshape(T, B, N, self.num_heads, C // self.num_heads)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )

        v = self.v_spike(v)  # (T, B, C_v, H, W)
        v = v.flatten(3)  # (T, B, C_v, N)
        v = (
            v.transpose(-1, -2)  # (T, B, N, C_v)
            .reshape(T, B, N, self.num_heads, self.C_v // self.num_heads)  # (T, B, N, num_heads, head_dim_v)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )

        kv = k.transpose(-2, -1) @ v  # [T, B, heads, qk_dim, v_dim]
        x = (q @ kv) * (self.scale * 2)  # [T, B, heads, N, v_dim]

        # (T, B, num_heads, head_dim_v, N) -> (T, B, C_v, N)
        x = x.transpose(-2, -1).reshape(T, B, self.C_v, N).contiguous()
        x = self.attn_spike(x)  # (T, B, C_v, N)
        x = x.reshape(T, B, self.C_v, H, W)
        x = self.proj_conv(x)

        return x


class Cross_MS_Attention_linear(nn.Module):
    def __init__(self, t: int, dim: int, num_heads=8, lambda_ratio=1, neuron_factory=None):
        super().__init__()
        assert (dim % num_heads == 0), f"dim {dim} should be divided by num_heads {num_heads}."
        self.neuron_factory = neuron_factory
        self.dim = dim
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.lambda_ratio = lambda_ratio
        self.C_v = int(dim * self.lambda_ratio)
        assert self.C_v % self.num_heads == 0

        self.head_spike_search = self.neuron_factory(t=t)
        self.head_spike_template = self.neuron_factory(t=t)

        self.q_conv = SeqToANNContainer(
            nn.Conv2d(dim, dim, 1, 1, bias=False),
            nn.BatchNorm2d(dim)
        )

        self.q_spike_search = self.neuron_factory(t=t)
        self.q_spike_template = self.neuron_factory(t=t)

        self.k_conv = SeqToANNContainer(
            nn.Conv2d(dim, dim, 1, 1, bias=False),
            nn.BatchNorm2d(dim)
        )

        self.k_spike_search = self.neuron_factory(t=t)
        self.k_spike_template = self.neuron_factory(t=t)

        self.v_conv = SeqToANNContainer(
            nn.Conv2d(dim, int(dim * lambda_ratio), 1, 1, bias=False),
            nn.BatchNorm2d(int(dim * lambda_ratio))
        )

        self.v_spike_search = self.neuron_factory(t=t)
        self.v_spike_template = self.neuron_factory(t=t)

        self.attn_spike = self.neuron_factory(t=t)

        self.proj_conv = SeqToANNContainer(
            nn.Conv2d(dim * lambda_ratio, dim, 1, 1, bias=False),
            nn.BatchNorm2d(dim)
        )

    def forward(self, x):
        T, B, C, H, W = x.shape  # [T, B, 256, 18, 18]
        N = H * W

        x = x.flatten(3)  # [T, B, 256, 324]
        x = x[..., :320]  # [T, B, 256, 320]

        search = x[..., :256]  # [T, B, 256, 256]
        template = x[..., 256:320]  # [T, B, 256, 64]

        template = template.reshape(T, B, C, 8, 8)
        search = search.reshape(T, B, C, 16, 16)

        template = self.head_spike_template(template)  # [T, B, 256, 8, 8]
        search = self.head_spike_search(search)  # [T, B, 256, 16, 16]

        N_template = 64
        N_search = 256

        q_template = self.q_conv(template)
        k_template = self.k_conv(template)
        v_template = self.v_conv(template)

        q_template = self.q_spike_template(q_template)
        q_template = q_template.flatten(3)  # [T, B, 256, 64]
        q_template = (
            q_template.transpose(-1, -2)
            .reshape(T, B, N_template, self.num_heads, C // self.num_heads)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )

        k_template = self.k_spike_template(k_template)
        k_template = k_template.flatten(3)
        k_template = (
            k_template.transpose(-1, -2)
            .reshape(T, B, N_template, self.num_heads, C // self.num_heads)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )

        v_template = self.v_spike_template(v_template)
        v_template = v_template.flatten(3)
        v_template = (
            v_template.transpose(-1, -2)
            .reshape(T, B, N_template, self.num_heads, self.C_v // self.num_heads)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )

        q_search = self.q_conv(search)
        k_search = self.k_conv(search)
        v_search = self.v_conv(search)

        q_search = self.q_spike_search(q_search)
        q_search = q_search.flatten(3)
        q_search = (
            q_search.transpose(-1, -2)
            .reshape(T, B, N_search, self.num_heads, C // self.num_heads)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )
        k_search = self.k_spike_search(k_search)
        k_search = k_search.flatten(3)
        k_search = (
            k_search.transpose(-1, -2)
            .reshape(T, B, N_search, self.num_heads, C // self.num_heads)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )

        v_search = self.v_spike_search(v_search)
        v_search = v_search.flatten(3)
        v_search = (
            v_search.transpose(-1, -2)
            .reshape(T, B, N_search, self.num_heads, self.C_v // self.num_heads)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )
        search_kv = k_search.transpose(-2, -1) @ v_search
        template = q_template @ search_kv * (self.scale * 2)

        template_kv = k_template.transpose(-2, -1) @ v_template
        search = q_search @ template_kv * (self.scale * 2)

        x = torch.cat((template, search), dim=3)
        _, _, _, _, C_padding = x.shape
        padding = torch.zeros((T, B, self.num_heads, 4, C_padding), device=x.device, dtype=x.dtype)
        x = torch.cat((x, padding), dim=3)

        x = x.transpose(2, 3).reshape(T, B, self.C_v, N).contiguous()
        x = self.attn_spike(x)
        x = x.reshape(T, B, self.C_v, H, W)
        x = self.proj_conv(x).reshape(T, B, C, H, W)

        return x


class MS_Block_Spike_AttnMLP(nn.Module):
    def __init__(self, t: int, dim: int, num_heads, mlp_ratio=4, lambda_ratio=4, cross=False, neuron_factory=None):
        super().__init__()
        self.neuron_factory = neuron_factory
        # self.conv = SepConv_Spike(dim=dim, kernel_size=3, padding=1)
        if cross == False:
            self.attn = MS_Attention_linear(
                t=t,
                dim=dim,
                num_heads=num_heads,
                lambda_ratio=lambda_ratio,
                neuron_factory=self.neuron_factory
            )
        else:
            self.attn = Cross_MS_Attention_linear(
                t=t,
                dim=dim,
                num_heads=num_heads,
                lambda_ratio=lambda_ratio,
                neuron_factory=self.neuron_factory
            )

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MS_MLP(t=t, in_features=dim, hidden_features=mlp_hidden_dim, neuron_factory=self.neuron_factory)

    def forward(self, x):  # (T, B, C, H, W)
        # x = x + self.conv(x)

        x = x + self.attn(x)
        x = x + self.mlp(x)

        return x


class MS_DownSampling(nn.Module):
    def __init__(
            self,
            t,
            in_channels=2,
            embed_dims=256,
            kernel_size=3,
            stride=2,
            padding=1,
            first_layer=True,
            neuron_mem = True,
            neuron_factory = None
    ):
        super().__init__()
        self.neuron_factory = neuron_factory
        self.encode_conv = SeqToANNContainer(
            nn.Conv2d(in_channels, embed_dims, kernel_size=kernel_size,
                      stride=stride, padding=padding)
        )

        self.encode_bn = SeqToANNContainer(
            nn.BatchNorm2d(embed_dims)
        )
        self.first_layer = first_layer
        if not first_layer:
            self.encode_spike = self.neuron_factory(t=t, mem=neuron_mem)

    def forward(self, x):
        if hasattr(self, "encode_spike"):
            x = self.encode_spike(x)
        x = self.encode_conv(x)
        x = self.encode_bn(x)

        return x