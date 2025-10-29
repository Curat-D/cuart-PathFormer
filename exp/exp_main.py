from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import PathFormer
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from utils.metrics import metric

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np
import nvtx
warnings.filterwarnings('ignore')
import torch._dynamo as dynamo
dynamo.config.suppress_errors = True  # 抑制错误

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'PathFormer': PathFormer,
        }
        model = model_dict[self.args.model].Model(self.args).float()

        # 使用torch.compile优化模型
        if self.args.use_compile:
            # 检查模型是否包含 FFT 操作
            def has_fft_operations(model):
                for name, module in model.named_modules():
                    if any('fft' in str(module).lower() for module in [module]):
                        return True
                return False
            
            if has_fft_operations(model):
                print("Model contains FFT operations, using compatible compilation...")
                import torch._dynamo as dynamo
                dynamo.config.suppress_errors = True
                
                model = torch.compile(
                    model,
                    backend='aot_eager',
                    mode='reduce-overhead',
                    dynamic=False,
                    fullgraph=False,
                )
            else:
                print("Model doesn't contain FFT operations, using full optimization...")
                model = torch.compile(
                    model,
                    backend='inductor',
                    mode='max-autotune',
                    dynamic=False,
                )

        # 使用DDP
        if hasattr(self.args, 'use_ddp') and self.args.use_ddp and self.args.use_gpu:
            print(f"Using DDP on rank {self.args.rank}")
            model = model.to(self.device)
            model = DDP(model, device_ids=[self.args.rank], output_device=self.args.rank)
        elif self.args.use_gpu:
            model = model.to(self.device)

        return model

    def _get_data(self, flag):
        start_time = time.time()
        
        # 在DDP模式下，为每个进程设置不同的随机种子
        if hasattr(self.args, 'use_ddp') and self.args.use_ddp:
            # 为每个rank设置不同的随机种子，确保数据shuffle不同
            seed = self.args.seed if hasattr(self.args, 'seed') else 1024
            torch.manual_seed(seed + self.args.rank)
        
        data_set, data_loader = data_provider(self.args, flag)
        load_time = time.time() - start_time
        if self.args.rank == 0:  # 只在主进程打印
            print(f"Data loading time for {flag}: {load_time:.4f} seconds")
        return data_set, data_loader, load_time

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.L1Loss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.model=='PathFormer':
                            if hasattr(self.args, 'use_ddp') and self.args.use_ddp:
                                outputs, balance_loss = self.model.module(batch_x)
                            else:
                                outputs, balance_loss = self.model(batch_x)
                        else:
                            if hasattr(self.args, 'use_ddp') and self.args.use_ddp:
                                outputs = self.model.module(batch_x)
                            else:
                                outputs = self.model(batch_x)
                else:
                    if self.args.model=='PathFormer':
                        if hasattr(self.args, 'use_ddp') and self.args.use_ddp:
                            outputs, balance_loss = self.model.module(batch_x)
                        else:
                            outputs, balance_loss = self.model(batch_x)
                    else:
                        if hasattr(self.args, 'use_ddp') and self.args.use_ddp:
                            outputs = self.model.module(batch_x)
                        else:
                            outputs = self.model(batch_x)
                
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)
                total_loss.append(loss)

        total_loss = np.average(total_loss)
        
        # 在DDP模式下，收集所有进程的loss
        if hasattr(self.args, 'use_ddp') and self.args.use_ddp:
            # 将所有进程的loss收集到rank 0
            loss_tensor = torch.tensor(total_loss).to(self.device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            total_loss = loss_tensor.item() / self.args.world_size
        
        self.model.train()
        return total_loss

    def train(self, setting):
        # 统计数据加载时间
        load_start_time = time.time()
        
        with nvtx.annotate("Data Loading", color="red"):
            train_data, train_loader, train_load_time = self._get_data(flag='train')
            vali_data, vali_loader, vali_load_time = self._get_data(flag='val')
            test_data, test_loader, test_load_time = self._get_data(flag='test')
        
        # 只在主进程打印加载时间
        if self.args.rank == 0:
            total_load_time = time.time() - load_start_time
            print(f"Total data loading time: {total_load_time:.4f} seconds")
            print(f"Train data loading time: {train_load_time:.4f} seconds")
            print(f"Validation data loading time: {vali_load_time:.4f} seconds")
            print(f"Test data loading time: {test_load_time:.4f} seconds")

        # 只在主进程创建checkpoint目录
        if self.args.rank == 0:
            path = os.path.join(self.args.checkpoints, setting)
            if not os.path.exists(path):
                os.makedirs(path)
        else:
            path = os.path.join(self.args.checkpoints, setting)

        # 在训练开始前进行模型编译warmup
        if self.args.use_compile and self.args.rank == 0:
            print("Warming up compiled model...")
            with nvtx.annotate("Model_Compile_Warmup", color="magenta"):
                with torch.cuda.amp.autocast(enabled=self.args.use_amp):
                    dummy_batch = torch.randn(
                        self.args.batch_size, 
                        self.args.seq_len, 
                        self.args.num_nodes if self.args.features == 'M' else 1,
                        device=self.device
                    )
                    if hasattr(self.args, 'use_ddp') and self.args.use_ddp:
                        self.model.module(dummy_batch)
                    else:
                        self.model(dummy_batch)
            print("Model compilation warmup completed")

        # 只在主进程打印参数数量
        if self.args.rank == 0:
            total_num = sum(p.numel() for p in self.model.parameters())
            print(f"Total parameters: {total_num}")

        time_now = time.time()
        train_steps = len(train_loader)
        
        # 只在主进程设置early stopping
        if self.args.rank == 0:
            early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        # 调整学习率调度器以适应DDP
        if hasattr(self.args, 'use_ddp') and self.args.use_ddp:
            # 在DDP模式下，每个epoch的steps需要根据实际数据调整
            actual_steps = len(train_loader)
        else:
            actual_steps = train_steps

        scheduler = lr_scheduler.OneCycleLR(
            optimizer=model_optim,
            steps_per_epoch=actual_steps,
            pct_start=self.args.pct_start,
            epochs=self.args.train_epochs,
            max_lr=self.args.learning_rate
        )

        # 创建专用的传输stream
        transfer_stream = torch.cuda.Stream()
        
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            
            # 设置epoch的随机种子，确保所有进程有相同的shuffle行为
            # 只有在使用 DistributedSampler 时才调用 set_epoch
            if hasattr(self.args, 'use_ddp') and self.args.use_ddp:
                if hasattr(train_loader, 'sampler') and hasattr(train_loader.sampler, 'set_epoch'):
                    train_loader.sampler.set_epoch(epoch)
            
            with nvtx.annotate(f"Epoch_{epoch}_Train", color="blue"):
                self.model.train()
                epoch_time = time.time()
                
                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                    iter_count += 1
                    
                    with nvtx.annotate("Optimizer Zero Grad", color="yellow"):
                        model_optim.zero_grad()

                    # 数据加载到GPU
                    with nvtx.annotate("Data to GPU", color="green"):
                        with torch.cuda.stream(transfer_stream):
                            batch_x = batch_x.float().to(self.device, non_blocking=True)
                            batch_y = batch_y.float().to(self.device, non_blocking=True)
                            batch_x_mark = batch_x_mark.float().to(self.device, non_blocking=True)
                            batch_y_mark = batch_y_mark.float().to(self.device, non_blocking=True)
                        
                        torch.cuda.current_stream().wait_stream(transfer_stream)

                    # 前向传播
                    with nvtx.annotate("Forward Pass", color="purple"):
                        if self.args.use_amp:
                            with torch.cuda.amp.autocast():
                                if self.args.model=='PathFormer':
                                    if hasattr(self.args, 'use_ddp') and self.args.use_ddp:
                                        outputs, balance_loss = self.model.module(batch_x)
                                    else:
                                        outputs, balance_loss = self.model(batch_x)
                                else:
                                    if hasattr(self.args, 'use_ddp') and self.args.use_ddp:
                                        outputs = self.model.module(batch_x)
                                    else:
                                        outputs = self.model(batch_x)

                                f_dim = -1 if self.args.features == 'MS' else 0
                                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                                loss = criterion(outputs, batch_y)
                                if self.args.model=="PathFormer":
                                    loss = loss + balance_loss
                                train_loss.append(loss.item())
                        else:
                            if self.args.model == 'PathFormer':
                                if hasattr(self.args, 'use_ddp') and self.args.use_ddp:
                                    outputs, balance_loss = self.model.module(batch_x)
                                else:
                                    outputs, balance_loss = self.model(batch_x)
                            else:
                                if hasattr(self.args, 'use_ddp') and self.args.use_ddp:
                                    outputs = self.model.module(batch_x)
                                else:
                                    outputs = self.model(batch_x)
                            f_dim = -1 if self.args.features == 'MS' else 0
                            outputs = outputs[:, -self.args.pred_len:, f_dim:]
                            batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                            loss = criterion(outputs, batch_y)
                            if self.args.model=="PathFormer":
                                loss = loss + balance_loss
                            train_loss.append(loss.item())

                    # 只在主进程打印日志
                    if self.args.rank == 0 and (i + 1) % 100 == 0:
                        print("\titers: {0}, epoch: {1} | loss: {2:.7f} ".format(
                            i + 1, epoch + 1, loss.item()))
                        speed = (time.time() - time_now) / iter_count
                        left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                        print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                        iter_count = 0
                        time_now = time.time()

                    # 反向传播
                    with nvtx.annotate("Backward Pass", color="orange"):
                        if self.args.use_amp:
                            scaler.scale(loss).backward()
                            scaler.step(model_optim)
                            scaler.update()
                        else:
                            loss.backward()
                            model_optim.step()

                    # 学习率调整
                    with nvtx.annotate("Learning Rate Adjustment", color="pink"):
                        if self.args.lradj == 'TST':
                            adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                            scheduler.step()

            # 计算平均训练loss并同步所有进程
            avg_train_loss = np.average(train_loss)
            if hasattr(self.args, 'use_ddp') and self.args.use_ddp:
                train_loss_tensor = torch.tensor(avg_train_loss).to(self.device)
                dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.SUM)
                avg_train_loss = train_loss_tensor.item() / self.args.world_size

            # 只在主进程打印epoch信息
            if self.args.rank == 0:
                print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))

            # 验证阶段
            with nvtx.annotate(f"Epoch_{epoch}_Validation", color="cyan"):
                vali_loss = self.vali(vali_data, vali_loader, criterion)
                test_loss = self.vali(test_data, test_loader, criterion)

            # 只在主进程打印验证结果和early stopping
            if self.args.rank == 0:
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                    epoch + 1, train_steps, avg_train_loss, vali_loss, test_loss))
                
                with nvtx.annotate("Early Stopping Check", color="brown"):
                    early_stopping(vali_loss, self.model.module if hasattr(self.args, 'use_ddp') and self.args.use_ddp else self.model, path)
                    if early_stopping.early_stop:
                        print("Early stopping")
                        break

            # 学习率调整
            with nvtx.annotate("Epoch End LR Adjust", color="pink"):
                if self.args.lradj != 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
                elif self.args.rank == 0:
                    print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

            # 同步所有进程
            if hasattr(self.args, 'use_ddp') and self.args.use_ddp:
                dist.barrier()

        # 只在主进程加载最佳模型
        if self.args.rank == 0:
            with nvtx.annotate("Load Best Model", color="gray"):
                best_model_path = path + '/' + 'checkpoint.pth'
                if hasattr(self.args, 'use_ddp') and self.args.use_ddp:
                    self.model.module.load_state_dict(torch.load(best_model_path))
                else:
                    self.model.load_state_dict(torch.load(best_model_path))
        
        return self.model
    # test和predict方法保持类似修改，确保只在主进程执行
    def test(self, setting, test=0):
        # 只在主进程执行测试
        if hasattr(self.args, 'use_ddp') and self.args.use_ddp and self.args.rank != 0:
            return
            
        test_data, test_loader, test_load_time = self._get_data(flag='test')
        print(f"Test data loading time: {test_load_time:.4f} seconds")

        if test:
            print('loading model')
            model_path = os.path.join('./checkpoints/' + setting, 'checkpoint.pth')
            if hasattr(self.args, 'use_ddp') and self.args.use_ddp:
                self.model.module.load_state_dict(torch.load(model_path))
            else:
                self.model.load_state_dict(torch.load(model_path))

        # ... 其余test代码保持不变，但确保使用self.model.module访问DDP模型
        preds = []
        trues = []
        inputx = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.model=='PathFormer':
                            if hasattr(self.args, 'use_ddp') and self.args.use_ddp:
                                outputs, balance_loss = self.model.module(batch_x)
                            else:
                                outputs, balance_loss = self.model(batch_x)
                        else:
                            if hasattr(self.args, 'use_ddp') and self.args.use_ddp:
                                outputs = self.model.module(batch_x)
                            else:
                                outputs = self.model(batch_x)
                else:
                    if self.args.model == 'PathFormer':
                        if hasattr(self.args, 'use_ddp') and self.args.use_ddp:
                            outputs, balance_loss = self.model.module(batch_x)
                        else:
                            outputs, balance_loss = self.model(batch_x)
                    else:
                        if hasattr(self.args, 'use_ddp') and self.args.use_ddp:
                            outputs = self.model.module(batch_x)
                        else:
                            outputs = self.model(batch_x)
                
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                inputx.append(batch_x.detach().cpu().numpy())

                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        if self.args.test_flop:
            test_params_flop((batch_x.shape[1], batch_x.shape[2]))
            exit()
            
        preds = np.concatenate(preds, axis=0)  # 沿 batch 维度拼接
        trues = np.concatenate(trues, axis=0)
        inputx = np.concatenate(inputx, axis=0)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        inputx = inputx.reshape(-1, inputx.shape[-2], inputx.shape[-1])

        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        print('mse:{}, mae:{}, rse:{}'.format(mse, mae, rse))
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, rse:{}'.format(mse, mae, rse))
        f.write('\n')
        f.write('\n')
        f.close()
        return

    def predict(self, setting, load=False):
        # 只在主进程执行预测
        if hasattr(self.args, 'use_ddp') and self.args.use_ddp and self.args.rank != 0:
            return
            
        pred_data, pred_loader, pred_load_time = self._get_data(flag='pred')
        print(f"Prediction data loading time: {pred_load_time:.4f} seconds")

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            if hasattr(self.args, 'use_ddp') and self.args.use_ddp:
                self.model.module.load_state_dict(torch.load(best_model_path))
            else:
                self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.model=='PathFormer':
                            if hasattr(self.args, 'use_ddp') and self.args.use_ddp:
                                outputs, a_loss = self.model.module(batch_x)
                            else:
                                outputs, a_loss = self.model(batch_x)
                        else:
                            if hasattr(self.args, 'use_ddp') and self.args.use_ddp:
                                outputs = self.model.module(batch_x)
                            else:
                                outputs = self.model(batch_x)
                else:
                    if self.args.model == 'PathFormer':
                        if hasattr(self.args, 'use_ddp') and self.args.use_ddp:
                            outputs, a_loss = self.model.module(batch_x)
                        else:
                            outputs, a_loss = self.model(batch_x)
                    else:
                        if hasattr(self.args, 'use_ddp') and self.args.use_ddp:
                            outputs = self.model.module(batch_x)
                        else:
                            outputs = self.model(batch_x)
                
                pred = outputs.detach().cpu().numpy()
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        return