[README.md](https://github.com/user-attachments/files/23048305/README.md)
# AF-DGNN

AF-DGNN is a hybrid model combining Dual Graph Neural Networks (GCN + GraphSAGE) for circRNA–miRNA association prediction.

# Requirements
  * Python 3.8 or higher
  * PyTorch 1.12 or higher
  * scikit-learn
  * GPU (default)

# Data
  * circ_str_similarity.txt
  * mi_str_similarity.txt
  * circ_seq_similarity.txt
  * mi_seq_similarity.txt
  * circRNA_miRNA_matrix.tx

# Running  the Code
  * Execute ```python main.py``` to run the code.
  * The model will perform 5-fold cross validation by default and output


