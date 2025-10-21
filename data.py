import os
import numpy as np
import torch
import pandas as pd

dir_path = '../data/'

# 1. load all disease name
df_circ = pd.read_excel(os.path.join(dir_path, 'circ.xlsx'))
circ = [str(dis).lower() for dis in df_circ.iloc[:, 1]]  # 假设疾病名称在第二列

# 2. load all miRNA name
df_mi = pd.read_excel(os.path.join(dir_path, 'mi.xlsx'))
mirna_name = [str(mi).lower() for mi in df_mi.iloc[:, 1]]  # 假设 miRNA 名称在第二列

# 3. load disease semantic similarity
dis_sim = np.loadtxt(os.path.join(dir_path, 'circ_str_similarity.txt'), dtype=np.float32)
dis_sim = torch.from_numpy(dis_sim)

# 4. load miRNA functional similarity
mi_sim = np.loadtxt(os.path.join(dir_path, 'mi_str_similarity.txt'), dtype=np.float32)
mi_sim = torch.from_numpy(mi_sim)

# 5. load gaussian disease similarity
dis_sim_gaussian = np.loadtxt(os.path.join(dir_path, 'circ_seq_similarity.txt'), dtype=np.float32)
dis_sim_gaussian = torch.from_numpy(dis_sim_gaussian)

# 6. load gaussian miRNA similarity
mi_sim_gaussian = np.loadtxt(os.path.join(dir_path, 'mi_seq_similarity.txt'), dtype=np.float32)
mi_sim_gaussian = torch.from_numpy(mi_sim_gaussian)

# 7. load known dis-miRNA interaction
dis_mi = np.loadtxt(os.path.join(dir_path, 'circRNA_miRNA_matrix.txt'), dtype=np.int32)
dis_mi = torch.from_numpy(dis_mi).long()
