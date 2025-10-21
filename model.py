import torch
import torch.nn as nn
import torch.nn.functional as F

# ===================== 工具函数 =====================
def apply_adj(A, X):
    if A is None:
        return X
    if A.is_sparse:
        return torch.sparse.mm(A, X)
    return torch.matmul(A, X)

def add_self_loop(A):
    if A is None:
        return None
    if A.is_sparse:
        i = torch.arange(A.size(0), device=A.device)
        self_loop = torch.sparse_coo_tensor(
            torch.stack([i, i]), torch.ones_like(i, dtype=A.dtype), A.shape
        )
        return (A + self_loop).coalesce()
    else:
        I = torch.eye(A.size(0), device=A.device, dtype=A.dtype)
        return A + I

def drop_edge(A, p):
    if p <= 0 or A is None:
        return A
    if A.is_sparse:
        idx = A.coalesce().indices()
        val = A.coalesce().values()
        keep = torch.rand_like(val) > p
        idx, val = idx[:, keep], val[keep]
        return torch.sparse_coo_tensor(idx, val, A.size(), device=A.device).coalesce()
    else:
        mask = (torch.rand_like(A) > p).to(A.dtype)
        return A * mask

# ===================== GCN 层 =====================
class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.2, activation=nn.ReLU(), bias=True, prenorm=True, residual=True):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=bias)
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.prenorm = nn.LayerNorm(in_dim) if prenorm else None
        self.residual = residual and (in_dim == out_dim)

    def forward(self, X, A_norm):
        H = X
        if self.prenorm is not None:
            H = self.prenorm(H)
        H = self.dropout(H)
        H = self.lin(H)
        H = apply_adj(A_norm, H)
        if self.activation is not None:
            H = self.activation(H)
        if self.residual:
            H = H + X
        return H

# ===================== GraphSAGE 层 =====================
class GraphSAGELayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.2, activation=nn.ReLU(), bias=True, prenorm=True, residual=True):
        super().__init__()
        self.lin = nn.Linear(in_dim * 2, out_dim, bias=bias)
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.prenorm = nn.LayerNorm(in_dim) if prenorm else None
        self.residual = residual and (in_dim == out_dim)

    def forward(self, X, A_norm):
        H_in = X
        H = X
        if self.prenorm is not None:
            H = self.prenorm(H)
        H = self.dropout(H)
        neigh = apply_adj(A_norm, H) if A_norm is not None else torch.zeros_like(H)
        cat = torch.cat([H, neigh], dim=1)
        H = self.lin(cat)
        if self.activation is not None:
            H = self.activation(H)
        if self.residual:
            H = H + H_in
        return H

# ===================== Softmax 门控融合 =====================
class SoftmaxGatedFusion(nn.Module):
    def __init__(self, in_dim, num_channels=2, temperature=1.0):
        super().__init__()
        self.num_channels = num_channels
        self.temperature = temperature
        self.proj = nn.ModuleList([nn.Linear(in_dim, in_dim) for _ in range(num_channels)])
        self.gate = nn.ModuleList([nn.Linear(in_dim, in_dim) for _ in range(num_channels)])
        self.ln = nn.LayerNorm(in_dim)

    def forward(self, H_list):
        Z = [self.proj[i](H_list[i]) for i in range(self.num_channels)]
        logits = torch.stack([self.gate[i](Z[i]) for i in range(self.num_channels)], dim=1)
        attn = torch.softmax(logits / self.temperature, dim=1)
        H = (attn * torch.stack(Z, dim=1)).sum(dim=1)
        return self.ln(H)

# ===================== 图卷积型解码器 =====================
class GraphDecoder(nn.Module):
    def __init__(self, hidden_size, dropout=0.2):
        super().__init__()
        self.gcn1 = GCNLayer(hidden_size, hidden_size, dropout=dropout, residual=True)
        self.gcn2 = GCNLayer(hidden_size, hidden_size, dropout=dropout, activation=None, residual=True)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, H, A_norm, pairs):
        H = self.gcn2(self.gcn1(H, A_norm), A_norm)
        h_u = H[pairs[:, 0]]
        h_v = H[pairs[:, 1]]
        edge_feat = torch.cat([h_u, h_v], dim=1)
        logits = self.mlp(edge_feat).squeeze(1)
        return logits

# ===================== DualGNN =====================
class DualGNN(nn.Module):
    def __init__(self, num_mi, num_dis, embed_size=128, hidden_size=128,
                 dropout=0.2, use_residual=True, pretrained_emb=None,
                 dropedge_p=0.1):
        super().__init__()
        self.num_mi = num_mi
        self.num_dis = num_dis
        self.N = num_mi + num_dis
        self.use_residual = use_residual
        self.dropedge_p = dropedge_p

        # 嵌入
        self.node_emb = nn.Embedding(self.N, embed_size)
        self.type_emb = nn.Embedding(2, embed_size)
        if pretrained_emb is not None:
            self.node_emb.weight.data.copy_(torch.tensor(pretrained_emb, dtype=torch.float32))
        else:
            nn.init.uniform_(self.node_emb.weight, -0.1, 0.1)
        nn.init.uniform_(self.type_emb.weight, -0.05, 0.05)

        # 编码器：GCN 分支
        self.gcn1 = GCNLayer(embed_size, hidden_size, dropout=dropout, residual=use_residual)
        self.gcn2 = GCNLayer(hidden_size, hidden_size, dropout=dropout, activation=None, residual=use_residual)

        # 编码器：GraphSAGE 分支
        self.sage1 = GraphSAGELayer(embed_size, hidden_size, dropout=dropout, residual=use_residual)
        self.sage2 = GraphSAGELayer(hidden_size, hidden_size, dropout=dropout, activation=None, residual=use_residual)

        # 融合
        self.fusion = SoftmaxGatedFusion(hidden_size, num_channels=2, temperature=0.7)
        self.out_norm = nn.LayerNorm(hidden_size)

        # 解码器
        self.decoder = GraphDecoder(hidden_size, dropout=dropout)

    def _encode(self, A_norm, H_norm=None):
        device = A_norm.device if A_norm is not None else (H_norm.device if H_norm is not None else torch.device('cpu'))
        ids = torch.arange(self.N, device=device, dtype=torch.long)
        X = self.node_emb(ids)

        type_ids = torch.cat([
            torch.zeros(self.num_mi, dtype=torch.long, device=device),
            torch.ones(self.num_dis, dtype=torch.long, device=device)
        ], dim=0)
        X = X + self.type_emb(type_ids)

        # ---------- 编码前超图传播 ----------
        if H_norm is None:
            H_norm = add_self_loop(A_norm)  # 默认使用普通邻接
        X = apply_adj(H_norm, X)

        # ---------- 邻接矩阵用于图卷积 ----------
        A_use = add_self_loop(A_norm)
        A_use = drop_edge(A_use, self.dropedge_p if self.training else 0.0)

        # ---------- 双通道编码 ----------
        H_gcn = self.gcn2(self.gcn1(X, A_use), A_use)
        H_sage = self.sage2(self.sage1(X, A_use), A_use)
        H = self.fusion([H_gcn, H_sage])
        H = self.out_norm(H)

        return H, A_use

    def extract_pair_features(self, pairs, A_norm, H_norm=None):
        H, A_use = self._encode(A_norm, H_norm)
        return H, A_use

    def forward(self, pairs, A_norm, H_norm=None):
        H, A_use = self.extract_pair_features(pairs, A_norm, H_norm)
        # 转换 disease 索引为全局索引
        full_pairs = torch.stack([pairs[:, 0].long(), pairs[:, 1].long() + self.num_mi], dim=1).to(H.device)
        logits = self.decoder(H, A_use, full_pairs)
        return logits

# ===================== Focal Loss =====================
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha, self.gamma, self.reduction = alpha, gamma, reduction

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce)
        loss = self.alpha * (1 - pt) ** self.gamma * bce
        return loss.mean() if self.reduction == 'mean' else loss.sum()
