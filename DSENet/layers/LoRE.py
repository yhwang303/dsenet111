import torch
import torch.nn as nn
import torch.nn.functional as F


class ReluSquared(nn.Module):
    def forward(self, x):
        return F.relu(x) ** 2


def exists(val): return val is not None
def default(val, default_val): return default_val if val is None else val
def init_zero_(layer):
    nn.init.constant_(layer.weight, 0.0)
    if exists(layer.bias):
        nn.init.constant_(layer.bias, 0.0)


class FeedForward(nn.Module):
    def __init__(self, dim: int, dim_out: int = None, mult=4, dropout: float = 0.0,
                 swish=True, relu_squared=False, no_bias=False, zero_init_output=False):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)

        if relu_squared:
            activation = ReluSquared()
        elif swish:
            activation = nn.SiLU()
        else:
            activation = nn.GELU()

        self.ff = nn.Sequential(
            nn.Linear(dim, inner_dim, bias=not no_bias),
            activation,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out, bias=not no_bias)
        )

        if zero_init_output:
            init_zero_(self.ff[-1])

    def forward(self, x):
        return self.ff(x)

class Dynamic_conv1d(nn.Module):
    def __init__(self, patch_len, out_planes=1, kernel_size=3, ratio=0.25, stride=1,
                 padding=1, K=4, temperature=34):
        super().__init__()
        self.K = K
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.padding = padding
        self.temperature = temperature

        hidden_planes = int(patch_len * ratio) + 1
        self.fc1 = nn.Conv1d(patch_len, hidden_planes, 1, bias=False)
        self.fc2 = nn.Conv1d(hidden_planes, K, 1, bias=True)

        self.weight = nn.Parameter(torch.randn(K, out_planes, 1, kernel_size))
        self.bias = nn.Parameter(torch.zeros(K, out_planes))
        self.gelu = nn.GELU()

    def forward(self, x):  # [B, N, P, L]
        B, N, P, L = x.shape
        total = B * N * P

        att = F.gelu(self.fc1(x.reshape(total, L).unsqueeze(-1)))
        att = self.fc2(att).view(total, -1)
        softmax_attention = F.softmax(att / self.temperature, dim=1)  # [total, K]

        weight = self.weight.view(self.K, -1)
        aggregate_weight = torch.mm(softmax_attention, weight).view(total, self.out_planes, 1, self.kernel_size)
        bias = torch.mm(softmax_attention, self.bias).view(total * self.out_planes)

        x_conv = x.reshape(1, total, L)  # [1, total, L]
        w = aggregate_weight.view(total * self.out_planes, 1, self.kernel_size)
        out = F.conv1d(x_conv, w, bias=bias, padding=self.padding, groups=total)
        out = out.view(B, N, P, -1)
        out = self.gelu(out)
        return out


class LoRE(nn.Module):
    def __init__(self, patch_len: int, dropout: float = 0.0, use_dynamic: bool = True):
        super().__init__()
        self.use_dynamic = use_dynamic
        self.patch_len = patch_len

        self.ffn = FeedForward(dim=patch_len, dim_out=patch_len, dropout=dropout)
        self.norm = nn.LayerNorm(patch_len)
        if self.use_dynamic:
            self.dynamic_conv = Dynamic_conv1d(patch_len=patch_len, out_planes=1, kernel_size=3, padding=1)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.ffn(x)
        x = self.norm(x + residual)
        if self.use_dynamic:
            x = self.dynamic_conv(x)
        x = self.dropout(x)
        return x
