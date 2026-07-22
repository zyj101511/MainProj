import torch
from numpy.ma.core import identity
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
    def __init__(self, t: int, dim: int, num_heads=8, lambda_ratio=1, bidirectional=False, neuron_factory=None):
        super().__init__()
        assert (dim % num_heads == 0), f"dim {dim} should be divided by num_heads {num_heads}."
        self.neuron_factory = neuron_factory
        self.dim = dim
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.lambda_ratio = lambda_ratio
        self.bidirectional = bidirectional
        self.C_v = int(dim * self.lambda_ratio)
        assert self.C_v % self.num_heads == 0

        self.head_spike_search = self.neuron_factory(t=t)
        self.head_spike_template = self.neuron_factory(t=t)

        self.q_conv = SeqToANNContainer(
            nn.Conv2d(dim, dim, 1, 1, bias=False),
            nn.BatchNorm2d(dim)
        )
        self.q_spike_search = self.neuron_factory(t=t)

        self.k_conv = SeqToANNContainer(
            nn.Conv2d(dim, dim, 1, 1, bias=False),
            nn.BatchNorm2d(dim)
        )
        self.v_conv = SeqToANNContainer(
            nn.Conv2d(dim, int(dim * lambda_ratio), 1, 1, bias=False),
            nn.BatchNorm2d(int(dim * lambda_ratio))
        )

        self.k_spike_template = self.neuron_factory(t=t)
        self.v_spike_template = self.neuron_factory(t=t)

        if self.bidirectional:
            self.q_spike_template = self.neuron_factory(t=t)
            self.k_spike_search = self.neuron_factory(t=t)
            self.v_spike_search = self.neuron_factory(t=t)

        self.search_attn_spike = self.neuron_factory(t=t)
        self.template_attn_spike = self.neuron_factory(t=t)

        self.search_proj_conv = SeqToANNContainer(
            nn.Conv2d(dim * lambda_ratio, dim, 1, 1, bias=False),
            nn.BatchNorm2d(dim)
        )
        self.template_proj_conv = SeqToANNContainer(
            nn.Conv2d(dim * lambda_ratio, dim, 1, 1, bias=False),
            nn.BatchNorm2d(dim)
        )

    def forward(self, x):
        T, B, C, H, W = x.shape  # [T, B, C, 18, 18]
        N = H * W

        x_flat = x.flatten(3)  # [T, B, C, 324]
        x_main = x_flat[..., :320]  # [T, B, C, 320]

        search = x_main[..., :256]  # [T, B, C, 256]
        template = x_main[..., 256:320]  # [T, B, C, 64]

        search = search.reshape(T, B, C, 16, 16)
        template = template.reshape(T, B, C, 8, 8)

        # Build T x T temporal pairs:
        # search_t paired with template_tau for all t, tau
        template = (
            template.unsqueeze(0)
            .expand(T, T, B, C, 8, 8)
            .contiguous()
            .view(T * T, B, C, 8, 8)
        )
        search = torch.repeat_interleave(search, repeats=T, dim=0)  # [T*T, B, C, 16, 16]

        template = self.head_spike_template(template)
        search = self.head_spike_search(search)

        N_template = 64
        N_search = 256

        k_template = self.k_conv(template)
        v_template = self.v_conv(template)

        k_template = self.k_spike_template(k_template)
        k_template = k_template.flatten(3)
        k_template = (
            k_template.transpose(-1, -2)
            .reshape(T * T, B, N_template, self.num_heads, C // self.num_heads)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )

        v_template = self.v_spike_template(v_template)
        v_template = v_template.flatten(3)
        v_template = (
            v_template.transpose(-1, -2)
            .reshape(T * T, B, N_template, self.num_heads, self.C_v // self.num_heads)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )

        q_search = self.q_conv(search)
        q_search = self.q_spike_search(q_search)
        q_search = q_search.flatten(3)
        q_search = (
            q_search.transpose(-1, -2)
            .reshape(T * T, B, N_search, self.num_heads, C // self.num_heads)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )

        if self.bidirectional:
            q_template = self.q_conv(template)
            q_template = self.q_spike_template(q_template)
            q_template = q_template.flatten(3)
            q_template = (
                q_template.transpose(-1, -2)
                .reshape(T * T, B, N_template, self.num_heads, C // self.num_heads)
                .permute(0, 1, 3, 2, 4)
                .contiguous()
            )

            k_search = self.k_conv(search)
            v_search = self.v_conv(search)

            k_search = self.k_spike_search(k_search)
            k_search = k_search.flatten(3)
            k_search = (
                k_search.transpose(-1, -2)
                .reshape(T * T, B, N_search, self.num_heads, C // self.num_heads)
                .permute(0, 1, 3, 2, 4)
                .contiguous()
            )

            v_search = self.v_spike_search(v_search)
            v_search = v_search.flatten(3)
            v_search = (
                v_search.transpose(-1, -2)
                .reshape(T * T, B, N_search, self.num_heads, self.C_v // self.num_heads)
                .permute(0, 1, 3, 2, 4)
                .contiguous()
            )

            search_kv = k_search.transpose(-2, -1) @ v_search
            template_out = q_template @ search_kv * (self.scale * 2)
        else:
            template_out = v_template

        template_kv = k_template.transpose(-2, -1) @ v_template
        search_out = q_search @ template_kv * (self.scale * 2)

        # Temporal aggregation over all template times for each search time
        search_out = search_out.reshape(T, T, B, self.num_heads, N_search, self.C_v // self.num_heads)
        search_logits = search_out.mean(dim=(-1, -2, -3))  # [T, T, B]
        search_weights = torch.softmax(search_logits, dim=1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        search_out = (search_out * search_weights).sum(dim=1)  # [T, B, heads, N_search, Cv//heads]

        template_out = template_out.reshape(T, T, B, self.num_heads, N_template, self.C_v // self.num_heads)
        if self.bidirectional:
            template_logits = template_out.mean(dim=(-1, -2, -3))
            template_weights = torch.softmax(template_logits, dim=0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            template_out = (template_out * template_weights).sum(dim=0)
        else:
            # keep diagonal correspondence for template branch
            template_out = template_out[0]

        search_patch = search_out.permute(0, 1, 4, 2, 3).reshape(T, B, self.C_v, 16, 16)
        template_patch = template_out.permute(0, 1, 4, 2, 3).reshape(T, B, self.C_v, 8, 8)

        search_patch = self.search_attn_spike(search_patch)
        template_patch = self.template_attn_spike(template_patch)

        search_patch = self.search_proj_conv(search_patch)
        template_patch = self.template_proj_conv(template_patch)

        search_token = search_patch.flatten(3)  # [T, B, C, 256]
        template_token = template_patch.flatten(3)  # [T, B, C, 64]

        out_main = torch.cat([search_token, template_token], dim=3)  # [T, B, C, 320]

        pad_token = x_flat[..., 320:]  # preserve the last 4 appendix tokens
        out = torch.cat([out_main, pad_token], dim=3)  # [T, B, C, 324]
        out = out.reshape(T, B, C, H, W)

        return out


class MAS_Cross_Attention_linear(nn.Module):
    def __init__(self, t: int, dim: int, num_heads=8, lambda_ratio=1, bidirectional=False, neuron_factory=None):
        super().__init__()
        assert (dim % num_heads == 0), f"dim {dim} should be divided by num_heads {num_heads}."
        self.neuron_factory = neuron_factory
        self.dim = dim
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.lambda_ratio = lambda_ratio
        self.bidirectional = bidirectional
        self.C_v = int(dim * self.lambda_ratio)
        assert self.C_v % self.num_heads == 0

        self.head_spike_search = self.neuron_factory(t=t)
        self.head_spike_template = self.neuron_factory(t=t)

        self.q_conv = SeqToANNContainer(
            nn.Conv2d(dim, dim, 1, 1, bias=False),
            nn.BatchNorm2d(dim)
        )

        self.q_spike_search = self.neuron_factory(t=t)

        self.k_conv = SeqToANNContainer(
            nn.Conv2d(dim, dim, 1, 1, bias=False),
            nn.BatchNorm2d(dim)
        )

        self.v_conv = SeqToANNContainer(
            nn.Conv2d(dim, int(dim * lambda_ratio), 1, 1, bias=False),
            nn.BatchNorm2d(int(dim * lambda_ratio))
        )

        if bidirectional:
            self.q_spike_template = self.neuron_factory(t=t)

            self.k_spike_search = self.neuron_factory(t=t)
            self.v_spike_search = self.neuron_factory(t=t)

        self.v_spike_template = self.neuron_factory(t=t)
        self.k_spike_template = self.neuron_factory(t=t)

        self.search_attn_spike = self.neuron_factory(t=t)
        self.template_attn_spike = self.neuron_factory(t=t)

        self.search_proj_conv = SeqToANNContainer(
            nn.Conv2d(dim * lambda_ratio, dim, 1, 1, bias=False),
            nn.BatchNorm2d(dim)
        )

        self.template_proj_conv = SeqToANNContainer(
            nn.Conv2d(dim * lambda_ratio, dim, 1, 1, bias=False),
            nn.BatchNorm2d(dim)
        )

    def forward(self, x):
        T, B, C, H, W = x.shape  # [T, B, C, H, W]
        assert H == W, f"Expected feature map to be square, got H={H}, W={W}"
        assert H % 3 == 0, f"Expected feature side divisible by 3, got H={H}"

        template_side = H // 3
        search_side = 2 * H // 3

        template = x[..., -int(template_side):, -int(template_side):]  # [T, B, C, template_side, template_side]
        template = template.unsqueeze(0).expand(T, T, B, C, template_side, template_side).contiguous().view(T * T, B, C, template_side, template_side)
        search = x[..., :int(search_side), :int(search_side)]  # [T * T, B, C, search_side, search_side]
        search = torch.repeat_interleave(search, repeats=T, dim=0)

        template = self.head_spike_template(template)  # [T * T, B, C, template_side, template_side]
        search = self.head_spike_search(search)  # [T * T, B, C, search_side, search_side]

        N_template = template_side * template_side
        N_search = search_side * search_side

        k_template = self.k_conv(template)
        v_template = self.v_conv(template)

        if self.bidirectional:
            q_template = self.q_conv(template)
            q_template = self.q_spike_template(q_template)
            q_template = q_template.flatten(3)  # [T * T, B, 256, 64]
            q_template = (
                q_template.transpose(-1, -2)  # [T * T, B, N, C]
                .reshape(T * T, B, N_template, self.num_heads, C // self.num_heads)
                .permute(0, 1, 3, 2, 4)
                .contiguous()
            )

        k_template = self.k_spike_template(k_template)
        k_template = k_template.flatten(3)
        k_template = (
            k_template.transpose(-1, -2)
            .reshape(T * T, B, N_template, self.num_heads, C // self.num_heads)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )

        v_template = self.v_spike_template(v_template)
        v_template = v_template.flatten(3)
        v_template = (
            v_template.transpose(-1, -2)
            .reshape(T * T, B, N_template, self.num_heads, self.C_v // self.num_heads)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )

        q_search = self.q_conv(search)

        q_search = self.q_spike_search(q_search)
        q_search = q_search.flatten(3)
        q_search = (
            q_search.transpose(-1, -2)
            .reshape(T * T, B, N_search, self.num_heads, C // self.num_heads)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )

        if self.bidirectional:
            k_search = self.k_conv(search)
            v_search = self.v_conv(search)

            k_search = self.k_spike_search(k_search)
            k_search = k_search.flatten(3)
            k_search = (
                k_search.transpose(-1, -2)
                .reshape(T * T, B, N_search, self.num_heads, C // self.num_heads)
                .permute(0, 1, 3, 2, 4)
                .contiguous()
            )

            v_search = self.v_spike_search(v_search)
            v_search = v_search.flatten(3)
            v_search = (
                v_search.transpose(-1, -2)
                .reshape(T * T, B, N_search, self.num_heads, self.C_v // self.num_heads)
                .permute(0, 1, 3, 2, 4)
                .contiguous()
            )
            search_kv = k_search.transpose(-2, -1) @ v_search
            template = q_template @ search_kv * (self.scale * 2)  # (T * T, B, num_head, N, Cv//heads)
        else:
            template = v_template  # (T * T, B, num_head, N, Cv//heads)

        template_kv = k_template.transpose(-2, -1) @ v_template
        search = q_search @ template_kv * (self.scale * 2)  # (T * T, B, num_head, N, Cv//heads)

        # softmax
        search = search.reshape(T, T, B, self.num_heads, N_search, self.C_v // self.num_heads)
        search_logits = search.mean(dim=(-1, -2, -3))  # (T, T, B)
        search_weights = torch.softmax(search_logits, dim=1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # 沿 template 维 softmax, shape  (T, T, B, 1, 1, 1)
        search = (search * search_weights).sum(dim=1)  # (T, B, heads, N_search, Cv//heads)

        template = template.reshape(T, T, B, self.num_heads, N_template, self.C_v // self.num_heads)

        if self.bidirectional:
            template_logits = template.mean(dim=(-1, -2, -3))  # (T, T, B)
            template_weights = torch.softmax(template_logits, dim=0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # 沿 search 维 softmax, shape  (T, T, B, 1, 1, 1)
            template = (template * template_weights).sum(dim=0)  # (T, B, heads, N_template, Cv//heads)
        else:
            template = template[0]  # (T, B, heads, N_template, Cv//heads)

        # place the two patches back into their original positions
        search_patch = search.permute(0, 1, 4, 2, 3).reshape(T, B, self.C_v, search_side, search_side)
        template_patch = template.permute(0, 1, 4, 2, 3).reshape(T, B, self.C_v, template_side, template_side)

        search_patch = self.search_attn_spike(search_patch)
        template_patch = self.template_attn_spike(template_patch)

        search_patch = self.search_proj_conv(search_patch)
        template_patch = self.template_proj_conv(template_patch)

        x_out = x.new_zeros(T, B, C, H, W)
        x_out[..., :search_side, :search_side] = search_patch
        x_out[..., -template_side:, -template_side:] = template_patch

        x_out[..., search_side:, :search_side] = x[..., search_side:, :search_side]
        x_out[..., :search_side, search_side:] = x[..., :search_side, search_side:]

        return x_out


class MS_Block_Spike_AttnMLP(nn.Module):
    def __init__(self, t: int, dim: int, num_heads, mlp_ratio=4, lambda_ratio=4, cross=False, bidirectional=False, neuron_factory=None):
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
                bidirectional=bidirectional,
                neuron_factory=self.neuron_factory
            )

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MS_MLP(t=t, in_features=dim, hidden_features=mlp_hidden_dim, neuron_factory=self.neuron_factory)

    def forward(self, x):  # (T, B, C, H, W)
        # x = x + self.conv(x)

        x = x + self.attn(x)
        x = x + self.mlp(x)

        return x  # (T, B, C, H, W)

class MAS_AttnMLP(nn.Module):
    def __init__(self, t: int, dim: int, num_heads, mlp_ratio=4, lambda_ratio=4, bidirectional=False, neuron_factory=None):
        super().__init__()
        self.neuron_factory = neuron_factory
        # self.conv = SepConv_Spike(dim=dim, kernel_size=3, padding=1)

        self.attn = MAS_Cross_Attention_linear(
            t=t,
            dim=dim,
            num_heads=num_heads,
            lambda_ratio=lambda_ratio,
            bidirectional=bidirectional,
            neuron_factory=self.neuron_factory
        )

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MS_MLP(t=t, in_features=dim, hidden_features=mlp_hidden_dim, neuron_factory=self.neuron_factory)

    def forward(self, x):  # (T, B, C, H, W)
        # x = x + self.conv(x)

        x = x + self.attn(x)
        x = x + self.mlp(x)

        return x  # (T, B, C, H, W)


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
        if not first_layer:  # 第一层不先过脉冲
            if not neuron_mem:
                self.encode_spike = self.neuron_factory(t=t, mem=False)
            else:
                self.encode_spike = self.neuron_factory(t=t)

    def forward(self, x):
        if hasattr(self, "encode_spike"):
            x = self.encode_spike(x)
        x = self.encode_conv(x)
        x = self.encode_bn(x)

        return x

class TimeFuse_Block(nn.Module):
    def __init__(self, t, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False, neuron_factory=None):
        super().__init__()
        self.conv3d = nn.Conv3d(
            in_channels=in_channels,
            out_channels=2 * out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias
        )
        self.conv2d = nn.Conv2d(
            in_channels=2 * out_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias
        )
        self.neuron_factory = neuron_factory
        self.bn1 = nn.BatchNorm2d(2 * out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.timefuse_spike1 = self.neuron_factory(t=t, mem=False)

    def forward(self, x):
        x = self.timefuse_spike1(x)
        x = x.permute(1, 2, 0, 3, 4)  # (B, C, T, H, W)
        x = self.conv3d(x)
        x = x.squeeze(2)
        x = self.bn1(x)
        x = self.conv2d(x)
        x = self.bn2(x)
        return x





