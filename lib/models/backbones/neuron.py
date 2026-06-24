import torch
import math
from torch import nn
from spikingjelly.activation_based import neuron


def init_sigmoid_param(p):
    return math.log((p) / (1 - p))

class Quant(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type='cuda')
    def forward(ctx, i, min_value, max_value):
        ctx.min = min_value
        ctx.max = max_value
        ctx.save_for_backward(i)
        return torch.round(torch.clamp(i, min=min_value, max=max_value))

    @staticmethod
    @torch.amp.custom_bwd(device_type='cuda')
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        i, = ctx.saved_tensors
        grad_input[i < ctx.min] = 0
        grad_input[i > ctx.max] = 0
        return grad_input, None, None

class MILIF(neuron.BaseNode):
    def __init__(self, min_v:float=0., max_v:float=4, norm:float|None=None,
                 t:int | None=None, decay:bool=False, decay_rate: float|None=None, state_clip:tuple|None=None,
                 learnable_decay=False, mem=False, infere_mode=False, store_v_seq=False, detach_reset=True, step_mode:str='m'):
        """
        ILIF neuron with memory state and decay
        :param min_v: the lower boundary of discrete stages
        :param max_v: the upper boundary of discrete stages, also known as D
        :param norm: the normalization factor, usually should be max_v - min_v
        :param t: the time steps of input data
        :param decay: whether to decay or not
        :param decay_rate: the decay rate for decay
        :param state_clip: clip memory state in the specific range, usually should be (min_v, max_v)
        :param learnable_decay: whether the decay should be learnable or not
        :param detach_reset: whether to detach the computation graph of the spike in training stage
        :param mem: whether to inherit memory state across forward
        :param infere_mode: whether to spread discrete values into binary spikes
        """

        super().__init__(v_threshold=1.,
                         v_reset=None,
                         detach_reset=detach_reset,
                         step_mode=step_mode)
        if step_mode == 'm' and t < 1:
            raise RuntimeError('t should be >= 1')
        if max_v <= 0 or min_v >= max_v or min_v < 0:
            raise RuntimeError('max_v and min_v should not less than 0, and max_v should be >= min_v')
        if norm is None:
            self.norm = max_v
        else:
            self.norm = norm
        if self.norm == 0:
            raise RuntimeError('norm should not be 0')
        if mem and not decay:
            raise RuntimeError('mem=True has no effect when decay=False')

        self.mem = mem
        self.min_v = min_v
        self.max_v = max_v
        self.T = t
        self.decay = decay
        self.learnable_decay = learnable_decay
        self.cur_ts = 0
        self.D = int(max_v)
        self.infere_mode = infere_mode
        self.state_clip = state_clip
        self.store_v_seq = store_v_seq

        if self.decay:
            if decay_rate is None or not (0 < decay_rate < 1):
                raise RuntimeError('decay_rate should be in (0, 1)')
            if t > 1:
                decay_logit = torch.full((t - 1,), init_sigmoid_param(decay_rate))
            else:
                decay_logit = torch.tensor([init_sigmoid_param(decay_rate)])
            # 如果是可学学习,注册成参数,不可学习时存进buffer(model.to(device)时会和模型一起)
            if learnable_decay:
                self.decay_rate = nn.Parameter(decay_logit)
            else:
                self.register_buffer('decay_rate', decay_logit)
        else:
            self.decay_rate = None

    def neuronal_charge(self, x):
        if not self.decay:
            self.v = x
        else:
            if self.T == 1:
                alpha = self.decay_rate[0].sigmoid()
                self.v = self.v * alpha + x if self.mem else x
            else:
                if self.cur_ts >= 1:
                    self.v = self.v * nn.functional.sigmoid(self.decay_rate[self.cur_ts-1]) + x
                else:
                    # 跨 forward 保留的状态，也先衰减一次
                    alpha = self.decay_rate[0].sigmoid()
                    self.v = self.v * alpha + x if self.mem else x

    def neuronal_fire(self):
        spike_count = Quant.apply(self.v, self.min_v, self.max_v)
        return spike_count / self.norm

    def neuronal_reset(self, spike):
        spike_d = spike.detach() if self.detach_reset else spike
        self.v = self.v - spike_d
        if self.state_clip is not None:
            self.v = torch.clamp(self.v, self.state_clip[0], self.state_clip[1])

    def single_step_forward(self, x: torch.Tensor):
        self.v_float_to_tensor(x)
        self.neuronal_charge(x)
        spike = self.neuronal_fire()
        self.neuronal_reset(spike)
        if self.infere_mode:
            return self.expand_spike_count(spike, self.D)
        return spike

    def multi_step_forward(self, x_seq: torch.Tensor):
        if x_seq.shape[0] != self.T:
            raise RuntimeError(f'input batch should have the same time length as defined T: {x_seq.shape}')

        if self.store_v_seq:
            v_seq = []
        if not self.mem:
            self.v = 0.

        y0 = None
        y_seq = None
        for t in range(self.T):
            self.cur_ts = t
            y = self.single_step_forward(x_seq[t])
            if y0 is None:
                y0 = y
                y_seq = torch.empty((self.T,) + tuple(y.shape), device=y.device, dtype=y.dtype)
            y_seq[t] = y
            if self.store_v_seq:
                v_seq.append(self.v)

        if self.store_v_seq:
            self.v_seq = torch.stack(v_seq)

        return y_seq

    def expand_spike_count(self, quant, D):
        # quant: values in [0, 1], shape(T, B, C, H, W)
        count = quant * self.norm
        levels = torch.arange(1, D+1, device=count.device)
        shape = (D,) + (1,) * count.dim()
        levels = levels.view(shape)
        return (count.unsqueeze(0) >= levels).to(quant)


if __name__ == '__main__':
    # (T, B, C, H, W)
    n = MILIF(min_v=0, max_v=4,
                  t = 1, decay=True,
                  decay_rate=0.25, state_clip=(0, 4),
                  learnable_decay=False, mem=True, store_v_seq=True)
    # the first forward
    dummy = torch.ones(1, 5, 3, 18, 18)
    y = n(dummy)
    print(n.v.mean())
    print(y.shape)  # (T, B, C, H, W)

    # the second forward with zero input
    dummy2 = torch.zeros(1, 5, 3, 18, 18)
    y2 = n(dummy2)
    print(y2.shape)
    print(n.v.mean())  # mem == True means memory across forward, so it should not be all 0

    dummy3 = torch.zeros(1, 5, 3, 18, 18)
    y3 = n(dummy3)
    print(y3.shape)
    print(n.v.mean())

    # spread the discrete output into spike trains
    # (T, D, B, C, H, W)
    n.infere_mode = True
    dummy4 = torch.zeros(1, 5, 3, 18, 18)
    y4 = n(dummy4)
    print(y4.shape)  # (T, D, B, C, H, W)