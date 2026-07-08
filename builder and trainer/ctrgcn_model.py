# ctrgcn_model_coco17.py — COCO-17 only, ROLE-AWARE M=2
# Compatible with build_adjacency(skeleton_layout="coco17")

import torch
import torch.nn as nn
import numpy as np

COCO17_PARENT = np.array([
    -1,  0,  0,  1,  2,
     0,  0,  5,  6,  7,  8,
     5,  6, 11, 12, 13, 14
], dtype=int)

def build_adjacency(
    add_self_loops: bool = True,
    normalize: bool = True,
    parent: np.ndarray = COCO17_PARENT,
    skeleton_layout: str = "coco17",   # <- ignore but accept for compatibility
    **kwargs,                            # <- ignore extra args safely
):
    parent = np.asarray(parent, dtype=int)
    V = int(parent.shape[0])
    A = np.zeros((V, V), dtype=np.float32)
    for v, p in enumerate(parent):
        if p >= 0:
            A[v, p] = 1.0
            A[p, v] = 1.0
    if add_self_loops:
        A += np.eye(V, dtype=np.float32)
    if normalize:
        D = A.sum(axis=1)
        D_inv_sqrt = np.diag(1.0 / np.sqrt(np.clip(D, 1e-6, None)))
        A = D_inv_sqrt @ A @ D_inv_sqrt
    return torch.tensor(A, dtype=torch.float32)

def masked_global_avg_pool(x, lengths):
    # x: (N, C, T, V)
    N, C, T, V = x.shape
    device = x.device
    mask = torch.zeros((N, 1, T, 1), dtype=x.dtype, device=device)
    Lc = lengths.clamp(min=0, max=T).detach()
    for i in range(N):
        Li = int(Lc[i].item())
        if Li > 0:
            mask[i, 0, :Li, 0] = 1.0
    x_sum = (x * mask).sum(dim=(2, 3))
    denom = (mask.sum(dim=(2, 3)).clamp_min(1.0) * V)
    return x_sum / denom

class CTRGC(nn.Module):
    def __init__(self, in_channels, out_channels, A, dropout=0.3):
        super().__init__()
        V = A.size(0)
        self.register_buffer("A", A)
        self.edge_param = nn.Parameter(torch.zeros(out_channels, V, V))
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        nn.init.constant_(self.edge_param, 0.0)

    def forward(self, x):
        x = self.conv(x)  # (N, C_out, T, V)
        A_eff = self.A.unsqueeze(0) + self.edge_param  # (C_out, V, V)
        x = torch.einsum("nctw,cvw->nctv", x, A_eff)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x

class TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1, dropout=0.3):
        super().__init__()
        pad = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, (kernel_size, 1),
                              stride=(stride, 1), padding=(pad, 0), bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x

class CTRGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, dropout=0.3):
        super().__init__()
        self.ctrgc = CTRGC(in_channels, out_channels, A, dropout=dropout)
        self.tcn = TemporalConv(out_channels, out_channels, stride=stride, dropout=dropout)
        if in_channels != out_channels or stride != 1:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1), bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.residual = nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        res = self.residual(x)
        x = self.ctrgc(x)
        x = self.tcn(x)
        x = x + res
        return self.relu(x)

class MultiStreamCTRGCN(nn.Module):
    def __init__(self, num_class, num_joints=17, adjacency_matrix=None,
                 dropout=0.3, in_channels=3, max_person=1):
        super().__init__()
        if int(num_joints) != 17:
            raise ValueError("COCO-17 only: num_joints must be 17.")
        A = adjacency_matrix if adjacency_matrix is not None else build_adjacency()
        self.max_person = int(max_person)

        self.joint_stream  = self._make_backbone(in_channels, A, dropout)
        self.bone_stream   = self._make_backbone(in_channels, A, dropout)
        self.motion_stream = self._make_backbone(in_channels, A, dropout)

        self.bn_joint  = nn.BatchNorm1d(in_channels * num_joints)
        self.bn_bone   = nn.BatchNorm1d(in_channels * num_joints)
        self.bn_motion = nn.BatchNorm1d(in_channels * num_joints)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(256 * self.max_person * 3, num_class)

    def _make_backbone(self, in_channels, A, dropout):
        return nn.ModuleList([
            CTRGCNBlock(in_channels,  64, A, stride=1, dropout=dropout),
            CTRGCNBlock(64,           64, A, stride=1, dropout=dropout),
            CTRGCNBlock(64,          128, A, stride=2, dropout=dropout),
            CTRGCNBlock(128,         256, A, stride=2, dropout=dropout),
        ])

    def _forward_stream_roleaware(self, x, layers, bn, lengths):
        # x: (N, C, T, V=17, M=2)
        N, C, T, V, M = x.shape
        if M != self.max_person:
            raise ValueError(f"M mismatch: input {M} vs model {self.max_person}")
        x = x.permute(0, 4, 1, 2, 3).contiguous().view(N * M, C, T, V)
        lengths_rep = lengths.repeat_interleave(M)

        x = x.view(N * M, C * V, T)
        x = bn(x)
        x = x.view(N * M, C, T, V)

        for layer in layers:
            x = layer(x)

        feat = masked_global_avg_pool(x, lengths_rep)  # (N*M,256)
        feat = feat.view(N, M, 256).reshape(N, M * 256)
        return feat

    def forward(self, joint, bone, motion, lengths=None):
        N = joint.shape[0]
        if lengths is None:
            T = joint.shape[2]
            lengths = torch.full((N,), T, dtype=torch.long, device=joint.device)

        j = self._forward_stream_roleaware(joint,  self.joint_stream,  self.bn_joint,  lengths)
        b = self._forward_stream_roleaware(bone,   self.bone_stream,   self.bn_bone,   lengths)
        m = self._forward_stream_roleaware(motion, self.motion_stream, self.bn_motion, lengths)

        x = torch.cat([j, b, m], dim=1)
        x = self.dropout(x)
        return self.fc(x)
