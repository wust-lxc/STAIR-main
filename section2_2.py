import warnings
warnings.filterwarnings("ignore")

import os
import matplotlib
# 服务器端设置
matplotlib.use('Agg') 
import scanpy as sc
import torch

# 导入模块
from STAIR.emb_alignment import Emb_Align
from STAIR.utils import *

# 辅助函数：创建文件夹 (对应您代码中的 construct_folder)
def construct_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    return folder_name

# ==============================================================================
# 1. 加载数据 (衔接上一步)
# ==============================================================================
# 假设数据路径与之前一致
adata = sc.read('./data/hypothalamic_preoptic.h5ad')
# ==============================================================================
# 2. 您提供的代码逻辑
# ==============================================================================
result_path = construct_folder('hypothalamic')
keys_use = ['A0', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11']

# Construct the model
# 初始化 STAIR 模型
emb_align = Emb_Align(adata, batch_key='batch', result_path=result_path)

# Prepare data for AE
# 准备数据，进行归一化等处理
emb_align.prepare()

# Preprocessing
# 训练 Autoencoder (AE) 提取特征
emb_align.preprocess()

# Extract Latent features
# 提取隐空间特征存入 adata.obsm['latent']
emb_align.latent()