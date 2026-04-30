# code/loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss 用于解决类别不平衡问题，降低易分类样本的权重。
    FL(pt) = -alpha * (1 - pt)^gamma * log(pt)
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: [B, 1] (logits)
        # targets: [B, 1] (0 or 1)
        
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss) # pt = exp(-BCE)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class OrthogonalLoss(nn.Module):
    """
    自定义的总损失函数，包含 Focal Loss (替代 BCE) 和 Orthogonal Loss。
    L_total = L_Focal + λ * L_orth
    """
    def __init__(self, lambda_orth=0.1, alpha=0.25, gamma=2.0):
        super().__init__()
        self.lambda_orth = lambda_orth
        # 替换 BCE Loss 为 Focal Loss
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma)

    def forward(self, outputs, labels):
        # 从模型输出的字典中获取所需的值
        logits, F_s, F_f = outputs["logits"], outputs["F_s"], outputs["F_f"]
        
        # 1. 计算 Focal Loss
        # 将标签转为 float 类型并匹配 logits 的形状 [B, 1]
        labels = labels.float().unsqueeze(1)
        l_main = self.focal_loss(logits, labels)
        
        # 2. 计算正交损失 (Orthogonal Loss)
        # 首先对特征向量进行 L2 归一化
        F_s_norm = F.normalize(F_s, p=2, dim=1)
        F_f_norm = F.normalize(F_f, p=2, dim=1)
        
        # 计算归一化后向量的点积，即余弦相似度
        cosine_sim = (F_s_norm * F_f_norm).sum(dim=1)
        
        # 正交损失是余弦相似度平方的均值
        l_orth = torch.mean(cosine_sim ** 2)
        
        # 3. 合并总损失
        total_loss = l_main + self.lambda_orth * l_orth
        
        return total_loss, l_main, l_orth
