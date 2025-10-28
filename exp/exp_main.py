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

import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings('ignore')


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'PathFormer': PathFormer,
        }
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        start_time = time.time()
        data_set, data_loader = data_provider(self.args, flag)
        load_time = time.time() - start_time
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
                            outputs, balance_loss = self.model(batch_x)
                        else:
                            outputs = self.model(batch_x)

                else:
                    if self.args.model=='PathFormer':
                        outputs, balance_loss = self.model(batch_x)
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
        self.model.train()
        return total_loss

    def train(self, setting):
        # 统计数据加载时间
        load_start_time = time.time()
        train_data, train_loader, train_load_time = self._get_data(flag='train')
        vali_data, vali_loader, vali_load_time = self._get_data(flag='val')
        test_data, test_loader, test_load_time = self._get_data(flag='test')
        total_load_time = time.time() - load_start_time
        
        print(f"Total data loading time: {total_load_time:.4f} seconds")
        print(f"Train data loading time: {train_load_time:.4f} seconds")
        print(f"Validation data loading time: {vali_load_time:.4f} seconds")
        print(f"Test data loading time: {test_load_time:.4f} seconds")

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        total_num = sum(p.numel() for p in self.model.parameters())
        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                            steps_per_epoch=train_steps,
                                            pct_start=self.args.pct_start,
                                            epochs=self.args.train_epochs,
                                            max_lr=self.args.learning_rate)

        # 记录每个epoch的数据加载时间
        epoch_load_times = []
        
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            self.model.train()
            epoch_time = time.time()
            
            # 记录每个batch的数据加载时间
            batch_load_times = []
            
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                batch_load_end = time.time()
                batch_load_time = batch_load_end - epoch_time if i == 0 else batch_load_end - batch_start_time
                batch_load_times.append(batch_load_time)
                
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.model=='PathFormer':
                            outputs, balance_loss = self.model(batch_x)
                        else:
                            outputs = self.model(batch_x)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if self.args.model == 'PathFormer':
                        outputs, balance_loss = self.model(batch_x)
                    else:
                        outputs = self.model(batch_x)
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    if self.args.model=="PathFormer":
                        loss = loss + balance_loss
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    avg_batch_load_time = np.mean(batch_load_times[-100:]) if len(batch_load_times) >= 100 else np.mean(batch_load_times)
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f} | avg batch load time: {3:.4f}s".format(
                        i + 1, epoch + 1, loss.item(), avg_batch_load_time))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

                if self.args.lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()
                
                # 记录当前batch结束时间，用于计算下一个batch的加载时间
                batch_start_time = time.time()

            epoch_load_time = np.sum(batch_load_times)
            epoch_load_times.append(epoch_load_time)
            
            print("Epoch: {} cost time: {}, data loading time: {}".format(
                epoch + 1, time.time() - epoch_time, epoch_load_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        # 输出数据加载时间统计
        if epoch_load_times:
            avg_epoch_load_time = np.mean(epoch_load_times)
            total_epoch_load_time = np.sum(epoch_load_times)
            print(f"\nData Loading Time Statistics:")
            print(f"Total epoch data loading time: {total_epoch_load_time:.4f} seconds")
            print(f"Average epoch data loading time: {avg_epoch_load_time:.4f} seconds")
            print(f"Max epoch data loading time: {np.max(epoch_load_times):.4f} seconds")
            print(f"Min epoch data loading time: {np.min(epoch_load_times):.4f} seconds")

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        return self.model

    def test(self, setting, test=0):
        test_data, test_loader, test_load_time = self._get_data(flag='test')
        print(f"Test data loading time: {test_load_time:.4f} seconds")

        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

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
                            outputs, balance_loss = self.model(batch_x)
                        else:
                            outputs = self.model(batch_x)
                else:
                    if self.args.model == 'PathFormer':
                        outputs, balance_loss = self.model(batch_x)
                    else:
                        outputs = self.model(batch_x)
                f_dim = -1 if self.args.features == 'MS' else 0

                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

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
        preds = np.array(preds)
        trues = np.array(trues)
        inputx = np.array(inputx)

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
        pred_data, pred_loader, pred_load_time = self._get_data(flag='pred')
        print(f"Prediction data loading time: {pred_load_time:.4f} seconds")

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))


        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)


                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.model=='PathFormer':
                            outputs, a_loss = self.model(batch_x)
                        else:
                            outputs = self.model(batch_x)

                else:
                    if self.args.model == 'PathFormer':
                        outputs, a_loss = self.model(batch_x)
                    else:
                        outputs = self.model(batch_x)
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        # folder_path = './results/' + setting + '/'
        # if not os.path.exists(folder_path):
        #     os.makedirs(folder_path)
        #
        # np.save(folder_path + 'real_prediction.npy', preds)

        return