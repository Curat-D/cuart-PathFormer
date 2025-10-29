import os
import torch
import torch.nn as nn

class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError

    def _acquire_device(self):
        if self.args.use_gpu:
            # 检查是否是 DDP 模式
            if hasattr(self.args, 'use_ddp') and self.args.use_ddp:
                # DDP 模式下，每个进程使用自己的 GPU
                device = torch.device(f'cuda:{self.args.rank}')
                print(f'Use GPU: cuda:{self.args.rank} for training')
            # 检查是否是传统的多GPU模式
            elif hasattr(self.args, 'use_multi_gpu') and self.args.use_multi_gpu:
                device = torch.device('cuda:{}'.format(self.args.gpu))
                print('Use GPU: cuda:{} for training'.format(self.args.gpu))
            # 单GPU模式
            else:
                device = torch.device('cuda:{}'.format(self.args.gpu))
                print('Use GPU: cuda:{} for training'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU for training')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass