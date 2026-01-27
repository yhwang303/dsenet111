import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba


class GateMamba(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=3, expand=1, dropout=0.):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.z_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):  # [B, W, D]
        residual = x
        h = self.norm(x)
        y = self.mamba(h)
        z = self.z_proj(h)
        gate_b = torch.sigmoid(z)
        gate_f = 1 - gate_b
        y_plus = y * F.silu(z) + h * gate_f
        y_plus = self.dropout(y_plus)
        return residual + y_plus


class DFBMambaLayerLayer(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=3, expand=1, dropout=0., ffn_ratio=4):
        super().__init__()
        self.fwd = GateMamba(d_model, d_state, d_conv, expand, dropout)
        self.bwd = GateMamba(d_model, d_state, d_conv, expand, dropout)

        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 2),
            nn.Softmax(dim=-1)
        )

        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_ratio * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_ratio * d_model, d_model),
            nn.Dropout(dropout)
        )
        self.norm = nn.LayerNorm(d_model)
    def forward(self, x):  # x: [B, W, D]
        re = x
        y_f = self.fwd(x)
        y_b = torch.flip(self.bwd(torch.flip(x, dims=[1])), dims=[1])

        gate_input = x.mean(dim=1)           # [B, D]
        weights = self.gate(gate_input)      # [B, 2]
        alpha, beta = weights[:, 0], weights[:, 1]  # [B]

        alpha = alpha.view(-1, 1, 1)
        beta = beta.view(-1, 1, 1)

        y = alpha * y_f + beta * y_b
        y = y + self.ffn(self.ffn_norm(y))
        
        return y


class Global_Stream_DFBMamba(nn.Module):
    def __init__(self,
                 c_in,
                 long_context_window,
                 target_window,
                 m_patch_len,
                 m_stride,
                 m_layers,
                 d_model,
                 d_ff,
                 norm='BatchNorm',
                 dropout=0.,
                 act='gelu',
                 pre_norm=False,
                 d_state=16,
                 d_conv=3,
                 fc_dropout=0.,
                 head_dropout=0.,
                 padding_patch='end',
                 pretrain_head=False,
                 head_type='flatten',
                 individual=False,
                 revin=False,
                 affine=False,
                 subtract_last=False,
                 verbose=False,
                 **kwargs):
        super().__init__()
        self.c_in = c_in
        self.d_model = d_model
        self.m_patch_len = m_patch_len
        self.m_stride = m_stride
        self.padding_patch = padding_patch

        self.m_patch_num = int((long_context_window - m_patch_len) / m_stride + 1)
        if padding_patch == 'end':
            self.m_patch_num += 1

        self.in_proj = nn.ModuleList([nn.Linear(m_patch_len, d_model) for _ in range(c_in)])

        ffn_ratio = max(1, d_ff // d_model)
        self.layers = nn.ModuleList([
            DFBMambaLayerLayer(d_model, d_state, d_conv, 1, dropout, ffn_ratio)
            for _ in range(m_layers)
        ])

    def _patchify_1d(self, x1c):
        B, L = x1c.shape
        P, S = self.m_patch_len, self.m_stride
        if self.padding_patch == 'end':
            need_len = (self.m_patch_num - 1) * S + P
            pad_right = max(0, need_len - L)
            if pad_right > 0:
                x1c = F.pad(x1c, (0, pad_right))
        return x1c.unfold(dimension=1, size=P, step=S)  # [B, Np, P]

    def forward(self, x):  # x: [B, C, L]
        B, C, L = x.shape
        outs = []
        for ci in range(C):
            patches = self._patchify_1d(x[:, ci, :])
            if patches.shape[1] != self.m_patch_num:
                patches = patches[:, :self.m_patch_num, :]
            h = self.in_proj[ci](patches)
            for layer in self.layers:
                h = layer(h)
            outs.append(h.transpose(1, 2))  # [B, D, Np]
        z = torch.stack(outs, dim=1)         # [B, C, D, Np]
        return z
