# run.py:
import torch
import numpy as np
import random
from exp.exp_main import Exp_Main
import argparse
import time
import os
import torch.distributed as dist
import torch.multiprocessing as mp

fix_seed = 1024
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

def setup(rank, world_size):
    """设置分布式训练环境"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # 初始化进程组
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    # 设置当前GPU
    torch.cuda.set_device(rank)

def cleanup():
    """清理分布式训练环境"""
    dist.destroy_process_group()

def run_training(rank, world_size, args):
    """在单个GPU上运行训练"""
    setup(rank, world_size)
    
    # 为每个进程设置不同的随机种子
    torch.manual_seed(fix_seed + rank)
    np.random.seed(fix_seed + rank)
    random.seed(fix_seed + rank)
    
    print(f"Running DDP on rank {rank}.")
    
    # 修改设备设置
    args.use_gpu = True
    args.gpu = rank
    args.rank = rank
    args.world_size = world_size
    
    # 处理patch_size_list
    args.patch_size_list = np.array(args.patch_size_list).reshape(args.layer_nums, -1).tolist()

    print(f'Args in experiment (rank {rank}):')
    print(args)

    Exp = Exp_Main

    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            setting = '{}_{}_ft{}_sl{}_pl{}_{}'.format(
                args.model_id,
                args.model,
                args.data_path[:-4],
                args.features,
                args.seq_len,
                args.pred_len, ii)

            exp = Exp(args)  # set experiments

            print(f'>>>>>>>start training : {setting} on rank {rank}>>>>>>>>>>>>>>>>>>>>>>>>>>')
            exp.train(setting)

            # 只在rank 0上执行测试和预测
            if rank == 0:
                time_now = time.time()
                print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.test(setting)
                print('Inference time: ', time.time() - time_now)

                if args.do_predict:
                    print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                    exp.predict(setting, True)

            # 等待所有进程完成
            dist.barrier()
            torch.cuda.empty_cache()
    
    cleanup()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multivariate Time Series Forecasting')

    # basic config
    parser.add_argument('--is_training', type=int, default=1, help='status')
    parser.add_argument('--model', type=str, default='PathFormer',
                        help='model name, options: [PathFormer]')
    parser.add_argument('--model_id', type=str, default="ETT.sh")

    # data loader
    parser.add_argument('--data', type=str, default='custom', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./dataset/weather', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='weather.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S]; M:multivariate predict multivariate, S:univariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--individual', action='store_true', default=False,
                        help='DLinear: a linear layer for each variate(channel) individually')

    # model
    parser.add_argument('--d_model', type=int, default=16)
    parser.add_argument('--d_ff', type=int, default=64)
    parser.add_argument('--num_nodes', type=int, default=21)
    parser.add_argument('--layer_nums', type=int, default=3)
    parser.add_argument('--k', type=int, default=2, help='choose the Top K patch size at the every layer ')
    parser.add_argument('--num_experts_list', type=list, default=[4, 4, 4])
    parser.add_argument('--patch_size_list', nargs='+', type=int, default=[16,12,8,32,12,8,6,4,8,6,4,2])
    parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')
    parser.add_argument('--revin', type=int, default=1, help='whether to apply RevIN')
    parser.add_argument('--drop', type=float, default=0.1, help='dropout ratio')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--residual_connection', type=int, default=0)
    parser.add_argument('--metric', type=str, default='mae')
    parser.add_argument('--batch_norm', type=int, default=0)

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=20, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='optimizer learning rate')
    parser.add_argument('--lradj', type=str, default='TST', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    parser.add_argument('--pct_start', type=float, default=0.4, help='pct_start')

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_ddp', action='store_true', help='use DDP for distributed training', default=False)
    parser.add_argument('--devices', type=str, default='0,1', help='device ids for DDP training')
    parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')

    # compile
    parser.add_argument('--use_compile', action='store_true', help='use torch.compile to optimize model')
    parser.add_argument('--compile_backend', type=str, default='inductor', help='backend for torch.compile')
    parser.add_argument('--compile_mode', type=str, default='default', help='mode for torch.compile')

    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_ddp and args.use_gpu:
        # 使用DDP时，设置world_size为GPU数量
        world_size = torch.cuda.device_count()
        print(f"Using DDP with {world_size} GPUs")
        
        # 启动多进程训练
        mp.spawn(run_training,
                 args=(world_size, args),
                 nprocs=world_size,
                 join=True)
    else:
        # 单GPU训练
        if args.use_gpu:
            torch.cuda.set_device(args.gpu)
        
        args.patch_size_list = np.array(args.patch_size_list).reshape(args.layer_nums, -1).tolist()

        print('Args in experiment:')
        print(args)

        Exp = Exp_Main

        if args.is_training:
            for ii in range(args.itr):
                setting = '{}_{}_ft{}_sl{}_pl{}_{}'.format(
                    args.model_id,
                    args.model,
                    args.data_path[:-4],
                    args.features,
                    args.seq_len,
                    args.pred_len, ii)

                exp = Exp(args)
                print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
                exp.train(setting)

                time_now = time.time()
                print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.test(setting)
                print('Inference time: ', time.time() - time_now)

                if args.do_predict:
                    print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                    exp.predict(setting, True)

                torch.cuda.empty_cache()
        else:
            ii = 0
            setting = '{}_{}_ft{}_sl{}_pl{}_{}'.format(
                args.model_id,
                args.model,
                args.data_path[:-4],
                args.features,
                args.seq_len,
                args.pred_len, ii)

            exp = Exp(args)
            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting, test=1)
            torch.cuda.empty_cache()