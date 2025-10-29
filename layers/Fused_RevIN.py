import torch
import torch.nn as nn
import torch.nn.functional as F

class FusedRevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False):
        super(FusedRevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(1, 1, num_features))
            self.affine_bias = nn.Parameter(torch.zeros(1, 1, num_features))
        
        # 使用更紧凑的缓存
        self.register_buffer('stats', torch.zeros(2, 1, 1, num_features))  # [2, 1, 1, C]

    def forward(self, x, mode: str):
        if mode == 'norm':
            return self._fused_normalize(x)
        elif mode == 'denorm':
            return self._fused_denormalize(x)
        else:
            raise NotImplementedError

    def _fused_normalize(self, x):
        if self.subtract_last:
            last_val = x[:, -1:, :]
            x_normalized = x - last_val
            self.stats[0] = last_val.mean(dim=0, keepdim=True)  # 简化存储
        else:
            # 一次性计算均值和标准差
            mean_val = x.mean(dim=1, keepdim=True)
            x_centered = x - mean_val
            # 使用更稳定的方差计算
            stdev_val = torch.sqrt(torch.mean(x_centered * x_centered, dim=1, keepdim=True) + self.eps)
            
            x_normalized = x_centered / stdev_val
            
            # 合并存储统计量
            self.stats[0] = mean_val.mean(dim=0, keepdim=True)
            self.stats[1] = stdev_val.mean(dim=0, keepdim=True)

        if self.affine:
            x_normalized = x_normalized * self.affine_weight + self.affine_bias
        
        return x_normalized

    def _fused_denormalize(self, x):
        if self.affine:
            x = (x - self.affine_bias) / (self.affine_weight + self.eps)
        
        if self.subtract_last:
            x = x + self.stats[0]
        else:
            x = x * self.stats[1] + self.stats[0]
            
        return x