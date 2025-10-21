import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    accuracy_score, f1_score, recall_score, precision_score,
    roc_curve, precision_recall_curve
)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

from datahelp import KnownDisMiDataset
from model import DualGNN  # GCN + GraphSAGE + Hypergraph + Decoder

# ===================== 配置 =====================
dir_path = '../data/'
batch_size = 128
embed_size = 128
hidden_size = 128
dropout = 0.2
seeds = [42]
k_folds = 5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
save_dir = './figures'
os.makedirs(save_dir, exist_ok=True)

# ===================== 加载数据 =====================
full_dataset = KnownDisMiDataset(train=True, test_ratio=0.0, random_state=42, dir_path=dir_path)
num_samples = len(full_dataset)
num_mi = full_dataset.num_mi
num_dis = full_dataset.num_dis

# 普通图邻接矩阵
A = torch.tensor(full_dataset.get_adjacency(), dtype=torch.float32).to(device)
# 超图邻接矩阵
A_hyper = torch.tensor(full_dataset.get_hypergraph_adjacency(), dtype=torch.float32).to(device)

# ===================== KFold =====================
kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

metrics_all = {'AUC': [], 'AUPR': [], 'Acc': [], 'F1': [], 'Recall': [], 'Precision': []}
roc_data, pr_data = [], []

for seed in seeds:
    np.random.seed(seed)
    torch.manual_seed(seed)
    print(f"==== Random Seed: {seed} ====")

    fold_idx = 1
    for train_idx, test_idx in kf.split(np.arange(num_samples)):
        print(f"---- Fold {fold_idx} ----")
        train_dataset = Subset(full_dataset, train_idx)
        test_dataset = Subset(full_dataset, test_idx)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # ===================== 模型 =====================
        model = DualGNN(
            num_mi=num_mi,
            num_dis=num_dis,
            embed_size=embed_size,
            hidden_size=hidden_size,
            dropout=dropout
        ).to(device)
        model.eval()

        # ===================== 特征提取 =====================
        def extract_features(loader):
            feats_list, labels_list = [], []
            with torch.no_grad():
                for pairs, labels in loader:
                    pairs = pairs.to(device)
                    # 使用超图传播
                    H, A_use = model.extract_pair_features(pairs, A, H_norm=A_hyper)
                    # 经过解码器的两层 GCN
                    H_updated = model.decoder.gcn2(
                        model.decoder.gcn1(H, A_use), A_use
                    )
                    h_m = H_updated[pairs[:, 0]]
                    h_d = H_updated[pairs[:, 1]]
                    feats = torch.cat([h_m, h_d], dim=1)
                    feats_list.append(feats.cpu().numpy())
                    labels_list.append(labels.numpy())
            return np.vstack(feats_list), np.hstack(labels_list)

        train_feats, train_labels = extract_features(train_loader)
        test_feats, test_labels = extract_features(test_loader)

        # 特征归一化
        scaler = StandardScaler()
        train_feats = scaler.fit_transform(train_feats)
        test_feats = scaler.transform(test_feats)

        # ===================== XGBoost 分类器 =====================
        clf = XGBClassifier(
            n_estimators=110,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        clf.fit(train_feats, train_labels)

        probs = clf.predict_proba(test_feats)[:, 1]
        pred = (probs >= 0.5).astype(int)

        # ===================== 评估 =====================
        auc = roc_auc_score(test_labels, probs)
        aupr = average_precision_score(test_labels, probs)
        acc = accuracy_score(test_labels, pred)
        f1 = f1_score(test_labels, pred)
        recall = recall_score(test_labels, pred)
        precision = precision_score(test_labels, pred)

        print(f"AUC: {auc:.4f} | AUPR: {aupr:.4f} | "
              f"Acc: {acc:.4f} | F1: {f1:.4f} | "
              f"Recall: {recall:.4f} | Precision: {precision:.4f}")

        metrics_all['AUC'].append(auc)
        metrics_all['AUPR'].append(aupr)
        metrics_all['Acc'].append(acc)
        metrics_all['F1'].append(f1)
        metrics_all['Recall'].append(recall)
        metrics_all['Precision'].append(precision)

        # ROC/PR 数据
        fpr, tpr, _ = roc_curve(test_labels, probs)
        precision_vals, recall_vals, _ = precision_recall_curve(test_labels, probs)
        roc_data.append((fpr, tpr, auc, fold_idx))
        pr_data.append((recall_vals, precision_vals, aupr, fold_idx))

        fold_idx += 1

# ===================== 平均结果 =====================
print("==== Overall Average Metrics ====")
for key, vals in metrics_all.items():
    print(f"{key}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

# ===================== 绘制 ROC 曲线 =====================
plt.figure(figsize=(7, 6))
for fpr, tpr, auc_val, idx in roc_data:
    plt.plot(fpr, tpr, label=f'Fold {idx} (AUC={auc_val:.4f})')
all_fpr = np.linspace(0, 1, 100)
mean_tpr = np.zeros_like(all_fpr)
for fpr, tpr, _, _ in roc_data:
    mean_tpr += np.interp(all_fpr, fpr, tpr)
mean_tpr /= len(roc_data)
plt.plot(all_fpr, mean_tpr, color='black', linestyle='--',
         label=f"Mean (AUC={np.mean(metrics_all['AUC']):.4f})")
plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
plt.xlabel('False Positive Rate', fontsize=18)
plt.ylabel('True Positive Rate', fontsize=18)
plt.title('ROC CMI - 20208', fontsize=20)
plt.legend(fontsize=18)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'ROC_all_folds.png'))
plt.show()

# ===================== 绘制 PR 曲线 =====================
plt.figure(figsize=(7, 6))
for recall_vals, precision_vals, aupr_val, idx in pr_data:
    plt.plot(recall_vals, precision_vals, label=f'Fold {idx} (AUPR={aupr_val:.4f})')
all_recalls = np.linspace(0, 1, 100)
mean_precisions = np.zeros_like(all_recalls)
for recall_vals, precision_vals, _, _ in pr_data:
    mean_precisions += np.interp(all_recalls, recall_vals[::-1], precision_vals[::-1])
mean_precisions /= len(pr_data)
mean_aupr_val = np.mean(metrics_all['AUPR'])
plt.plot(all_recalls, mean_precisions, color='black', linestyle='--',
         label=f"Mean (AUPR={mean_aupr_val:.4f})")
plt.xlabel('Recall', fontsize=18)
plt.ylabel('Precision', fontsize=18)
plt.title('PR CMI - 20208', fontsize=20)
plt.legend(fontsize=18)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'PR_all_folds.png'))
plt.show()
