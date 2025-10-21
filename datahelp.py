import torch
from torch.utils.data import Dataset
import numpy as np
import os

# ================= 疾病相似度数据集 =================
class DisDataset(Dataset):
    def __init__(self, alpha=0.5):
        from data import dis_sim, dis_sim_gaussian
        sim = dis_sim * alpha + dis_sim_gaussian * (1 - alpha)
        x, y = [], []
        for i in range(sim.shape[0]):
            for j in range(sim.shape[1]):
                tmp = sim[i][j]
                if i != j and tmp != 0:
                    x.append([i, j])
                    y.append(tmp)
        self.x_data = torch.tensor(x, dtype=torch.long)
        self.y_data = torch.tensor(y, dtype=torch.float)
        self.len = len(y)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


# ================= miRNA相似度数据集 =================
class MiDataset(Dataset):
    def __init__(self, beta=0.5):
        from data import mi_sim, mi_sim_gaussian
        sim = mi_sim * beta + mi_sim_gaussian * (1 - beta)
        x, y = [], []
        for i in range(sim.shape[0]):
            for j in range(sim.shape[1]):
                tmp = sim[i][j]
                if i != j and tmp != 0:
                    x.append([i, j])
                    y.append(tmp)
        self.x_data = torch.tensor(x, dtype=torch.long)
        self.y_data = torch.tensor(y, dtype=torch.float)
        self.len = len(y)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


# ================= 已知疾病-miRNA 关系数据集 =================
class KnownDisMiDataset(Dataset):
    def __init__(self, train=True, test_ratio=0.2, random_state=42, dir_path='../data/'):
        # 载入 circRNA-miRNA 关系矩阵 (行: 疾病, 列: miRNA)
        dis_mi = np.loadtxt(os.path.join(dir_path, 'circRNA_miRNA_matrix.txt'), dtype=np.int32)

        self.num_dis, self.num_mi = dis_mi.shape
        self.dis_mi_matrix = dis_mi  # 保存原始矩阵

        # 正样本
        pos_pairs = np.argwhere(dis_mi == 1)
        pos_labels = np.ones(len(pos_pairs), dtype=np.int32)

        # 负样本 (随机采样)
        neg_pairs = np.argwhere(dis_mi == 0)
        np.random.shuffle(neg_pairs)
        neg_pairs = neg_pairs[:len(pos_pairs)]
        neg_labels = np.zeros(len(neg_pairs), dtype=np.int32)

        # 合并正负样本
        all_pairs = np.vstack([pos_pairs, neg_pairs])
        all_labels = np.concatenate([pos_labels, neg_labels])

        # 转换 (miRNA_idx, disease_idx)
        all_pairs = all_pairs[:, [1, 0]]

        # 划分训练/测试
        np.random.seed(random_state)
        indices = np.arange(len(all_pairs))
        np.random.shuffle(indices)

        split = int(len(all_pairs) * (1 - test_ratio))
        if train:
            self.pairs = all_pairs[indices[:split]]
            self.labels = all_labels[indices[:split]]
        else:
            self.pairs = all_pairs[indices[split:]]
            self.labels = all_labels[indices[split:]]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        pair = torch.tensor(self.pairs[index], dtype=torch.long)
        label = torch.tensor(self.labels[index], dtype=torch.float)
        return pair, label

    # ========== 普通邻接矩阵 ==========
    def get_adjacency(self):
        N = self.num_dis + self.num_mi
        A = np.zeros((N, N), dtype=np.float32)

        for d in range(self.num_dis):
            for m in range(self.num_mi):
                if self.dis_mi_matrix[d, m] == 1:
                    A[d, self.num_dis + m] = 1.0
                    A[self.num_dis + m, d] = 1.0
        return A

    # ========== 超图邻接矩阵 ==========
    def get_hypergraph_adjacency(self):
        """
        构造超图邻接矩阵 H (N x E)，
        - N: 节点数 (疾病 + miRNA)
        - E: 超边数 (这里用每个疾病和 miRNA 关联来构造超边)
        """
        N = self.num_dis + self.num_mi
        # 每条正样本 (d,m) 对应一个超边
        pos_edges = np.argwhere(self.dis_mi_matrix == 1)
        E = len(pos_edges)

        H = np.zeros((N, E), dtype=np.float32)

        for e, (d, m) in enumerate(pos_edges):
            d_idx = d
            m_idx = self.num_dis + m
            H[d_idx, e] = 1.0
            H[m_idx, e] = 1.0

        # 转换为超图邻接矩阵 A_hyper = H * H^T
        A_hyper = H @ H.T
        np.fill_diagonal(A_hyper, 0)  # 去掉自环

        return A_hyper
