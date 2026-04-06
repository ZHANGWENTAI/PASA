import os
import random

import numpy as np
import torch


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 确保所有GPU的随机种子都设置
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # 禁用TF32以确保不同GPU型号上的一致性（TF32在不同GPU上可能有不同的行为）
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # 设置CUBLAS工作空间配置以确保确定性
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    # 尝试使用确定性算法（如果可用）
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        # 如果某些操作不支持确定性算法，忽略错误
        pass
