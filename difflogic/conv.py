import torch
from torch import nn, Tensor
import torch.nn.functional as F
import einops

logic_gates = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

def apply_logic_gate(a: Tensor, b: Tensor, logic_gate: int):
    return {
        0:  torch.zeros_like(a),
        1:  a * b,
        2:  a - a * b,
        3:  a,
        4:  b - a * b,
        5:  b,
        6:  a + b - 2 * a * b,
        7:  a + b - a * b,
        8:  1 - (a + b - a * b),
        9:  1 - (a + b - 2 * a * b),
        10:  1 - b,
        11:  1 - b + a * b,
        12:  1 - a,
        13:  1 - a + a * b,
        14:  1 - a * b,
        15:  torch.ones_like(a),
    }[logic_gate]


class Logic(nn.Module):
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 initialization_type: str = 'residual',
                 ):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.initialization_type = initialization_type

        a, b = self.get_connections()
        self.register_buffer('a', a)
        self.register_buffer('b', b)

        weights = torch.randn(out_dim, len(logic_gates))
        if self.initialization_type == 'residual':
            weights[:, :] = 0
            weights[:, 3] = 5

        self.weights = torch.nn.parameter.Parameter(weights)

    def forward(self, x: Tensor):
        a, b = x[:, self.a, ...], x[:, self.b, ...]

        if self.training:
            normalized_weights = torch.nn.functional.softmax(self.weights, dim=-1).to(x.dtype)

            r = torch.zeros_like(a).to(x.dtype)

            for logic_gate in logic_gates:
                if len(a.shape) > 2:
                    nw = einops.repeat(normalized_weights[..., logic_gate], 'weights -> weights depth', depth=a.shape[-1])
                else:
                    nw = normalized_weights[..., logic_gate]
                r = r + nw * apply_logic_gate(a, b, logic_gate)
            return r
        else:
            one_hot_weights = torch.nn.functional.one_hot(self.weights.argmax(-1), len(logic_gates)).to(torch.float32)

            with torch.no_grad():
                r = torch.zeros_like(a).to(x.dtype)

                for logic_gate in logic_gates:
                    if len(a.shape) > 2:
                        ohw = einops.repeat(one_hot_weights[..., logic_gate], 'weights -> weights depth', depth=a.shape[-1])
                    else:
                        ohw = one_hot_weights[..., logic_gate]

                    r = r + ohw * apply_logic_gate(a, b, logic_gate)
                return r

    def get_connections(self):
        connections = torch.randperm(2 * self.out_dim) % self.in_dim
        connections = torch.randperm(self.in_dim)[connections]
        connections = connections.reshape(2, self.out_dim)
        a, b = connections[0], connections[1]
        a, b = a.to(torch.int64), b.to(torch.int64)
        return a, b



class LogicTree(nn.Module):
    def __init__(self,
                 in_dim: int,
                 depth: int = 3,
                 initialization_type: str = 'residual',
                 ):
        super().__init__()

        self.tree = nn.Sequential(
            Logic(in_dim, int(2 ** (depth - 1)), initialization_type=initialization_type),
            *[
                Logic(int(2 ** (depth - 1 - i)), int(2 ** (depth - 1 - i - 1)), initialization_type=initialization_type) for i in range(0, depth - 1, 1)
            ]
        )

    def forward(self, x: Tensor):
        return self.tree(x)




class Conv(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 depth: int = 3,
                 kernel_size: int = 3,
                 stride: int = 1,
                 padding: int = 1,
                 initialization_type: str = 'residual',
                 ):
        super().__init__()

        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.filters = nn.ModuleList([LogicTree(in_dim=kernel_size ** 2 * in_channels, depth=depth, initialization_type=initialization_type) for _ in range(out_channels)])

        self.stride = stride
        self.padding = padding

    def forward(self, x: Tensor) -> Tensor:
        batch_size, _, height, width = x.shape
        x = F.pad(x, (self.padding, self.padding, self.padding, self.padding), mode='constant', value=0)

        out_height = (height + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_width = (width + 2 * self.padding - self.kernel_size) // self.stride + 1

        patches = F.unfold(x, kernel_size=self.kernel_size, stride=self.stride)

        outputs = []
        for filter in self.filters:
            out = filter(patches)  # Shape: (batch_size, 1, out_height * out_width)
            out = einops.rearrange(out, 'b 1 (h w) -> b (h w)', h=out_height, w=out_width)
            outputs.append(out)

        output_tensor = torch.stack(outputs, dim=1)  # Shape: (batch_size, out_channels, out_height * out_width)
        output_tensor = einops.rearrange(output_tensor, 'b c (h w) -> b c h w', h=out_height, w=out_width)

        return output_tensor
