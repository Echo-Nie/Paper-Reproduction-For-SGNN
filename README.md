# DeeperGATGNN

Github repository for our paper - **"Scalable Deeper Graph Neural Networks for High-performance Materials Property Prediction"** [PDF](https://www.cell.com/patterns/pdfExtended/S2666-3899(22)00076-9). 

The repository mainly analyzes the source code and does not make any improvements yet, the purpose of the reproduction is mainly to find innovations.

# Necessary Installations

```bash
pip install torch==2.4.1
```

```bash
export TORCH=2.4.1
export CUDA=cu118
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-geometric
```

Replace the ```${TORCH}``` with your PyTorch version and ```${CUDA}``` with your cuda version without the '.' and a prefix 'cu'. For example, if your PyTorch version is 1.9.0 and your CUDA version is 10.2, then replace ```${TORCH}``` with 1.9.0 and ```${CUDA}``` with cu102.

```bash
pip install -r requirements.txt
```

Requirements are as follows：

```bash
ase==3.24.0
joblib==1.4.2
matplotlib==3.9.2
numpy==1.26.4
ray==2.41.0
scikit-learn==1.5.2
scipy==1.15.1
tensorboard==2.17.1
```

# Model Structure

```bash
Input Data (data.x, data.edge_index, data.edge_attr, data.batch, data.glob_feat)
  |
  |→ Preprocessing Layers (pre_lin_list_N, pre_lin_list_E)
  ↓
Node Features (out_x) + Edge Features (out_e)
  |
  |→ Graph Attention Layers (GATGNN_AGAT_LAYER × gc_count)
  ↓
Updated Node Features
  |
  |→ Global Attention Module
  ↓
Attention-Weighted Node Features
  |
  |→ Pooling Layer (global_add_pool, global_mean_pool, etc.)
  ↓
Pooled Features
  |
  |→ Postprocessing Layers (post_lin_list)
  ↓
Output Features
  |
  |→ Output Layer (lin_out)
  ↓
Predictions
```

**输入层**

**预处理层**：通过全连接层将节点和边特征映射到更高维空间

**图注意力层**：通过多头注意力机制和消息传递机制更新节点特征

**全局注意力模块**：计算节点的全局注意力权重，并应用到节点特征上

**池化层**：将节点特征聚合为图级特征

**后处理层**：通过全连接层对图级特征进行进一步加工

**输出层**



