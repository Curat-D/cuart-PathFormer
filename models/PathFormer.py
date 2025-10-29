import math
import torch
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import numpy as np
from layers.AMS import AMS
from layers.Layer import WeightGenerator, CustomLinear
#from layers.Fused_RevIN import FusedRevIN as RevIN
from layers.RevIN import RevIN
#from layers.Optimized_AMS import OptimizedAMS as AMS
#from layers.Optimized_Layer import WeightGenerator, CustomLinear
from functools import reduce
from operator import mul
import nvtx


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.layer_nums = configs.layer_nums  # 设置pathway的层数
        self.num_nodes = configs.num_nodes
        self.pre_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.k = configs.k
        self.num_experts_list = configs.num_experts_list
        self.patch_size_list = configs.patch_size_list
        self.d_model = configs.d_model
        self.d_ff = configs.d_ff
        self.residual_connection = configs.residual_connection
        self.revin = configs.revin
        if self.revin:
            self.revin_layer = RevIN(num_features=configs.num_nodes, affine=False, subtract_last=False)

        self.start_fc = nn.Linear(in_features=1, out_features=self.d_model)
        self.AMS_lists = nn.ModuleList()
        self.device = torch.device('cuda:{}'.format(configs.gpu))
        self.batch_norm = configs.batch_norm

        for num in range(self.layer_nums):
            self.AMS_lists.append(
                AMS(self.seq_len, self.seq_len, self.num_experts_list[num], self.device, k=self.k,
                    num_nodes=self.num_nodes, patch_size=self.patch_size_list[num], noisy_gating=True,
                    d_model=self.d_model, d_ff=self.d_ff, layer_number=num + 1, residual_connection=self.residual_connection, batch_norm=self.batch_norm))
        self.projections = nn.Sequential(
            nn.Linear(self.seq_len * self.d_model, self.pre_len)
        )

    def forward(self, x):
        balance_loss = 0
        
        # norm
        nvtx.push_range("RevIN_Normalize")
        if self.revin:
            x = self.revin_layer(x, 'norm')
        nvtx.pop_range()  # RevIN_Normalize
        
        nvtx.push_range("Start_Linear")
        out = self.start_fc(x.unsqueeze(-1))
        nvtx.pop_range()  # Start_Linear

        batch_size = x.shape[0]

        # AMS层循环
        for i, layer in enumerate(self.AMS_lists):
            nvtx.push_range(f"AMS_Layer_{i}")
            out, aux_loss = layer(out)
            balance_loss += aux_loss
            nvtx.pop_range()  # AMS_Layer_{i}

        nvtx.push_range("Reshape_Operations")
        out = out.permute(0,2,1,3).reshape(batch_size, self.num_nodes, -1)
        out = self.projections(out).transpose(2, 1)
        nvtx.pop_range()  # Reshape_Operations

        # denorm
        nvtx.push_range("RevIN_Denormalize")
        if self.revin:
            out = self.revin_layer(out, 'denorm')
        nvtx.pop_range()  # RevIN_Denormalize

        return out, balance_loss
