from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred,Dataset_Pretrain
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler
import torch
import nvtx
data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
    'pretrain': Dataset_Pretrain,
}

def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    # 基本参数设置
    if flag == 'test':
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size
        freq = args.freq
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = args.freq
        Data = Dataset_Pred
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq

    
    # 先创建 data_set，所有 rank 都执行
    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq
    )

    # DDP模式下的特殊处理
    if hasattr(args, 'use_ddp') and args.use_ddp:
        if flag == 'train':
            sampler = DistributedSampler(
                data_set, 
                num_replicas=args.world_size,
                rank=args.rank,
                shuffle=shuffle_flag,
                drop_last=drop_last
            )
            shuffle_flag = False
            print(f"[Rank {args.rank}] Train sampler: {len(sampler)} samples out of {len(data_set)}")
        else:
            if args.rank == 0:
                sampler = None
                print(f"[Rank {args.rank}] {flag} dataset: using full dataset on rank 0")
            else:
                # 现在 data_set 已定义，可安全使用
                from torch.utils.data import TensorDataset
                empty_data = TensorDataset(
                    torch.zeros(0, args.seq_len, data_set.data_x.shape[1]),
                    torch.zeros(0, args.pred_len, data_set.data_y.shape[1]),
                    torch.zeros(0, args.seq_len, data_set.data_stamp.shape[1]),
                    torch.zeros(0, args.pred_len, data_set.data_stamp.shape[1])
                )
                data_loader = DataLoader(
                    empty_data,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=0
                )
                return empty_data, data_loader
    else:
        sampler = None

    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        prefetch_factor=2,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False,
        sampler=sampler,
        drop_last=drop_last
    )
    
    return data_set, data_loader