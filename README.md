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

Requirements are as follows

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

