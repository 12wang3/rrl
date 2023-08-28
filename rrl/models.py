import sys
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn import metrics
from collections import defaultdict

from rrl.components import BinarizeLayer
from rrl.components import UnionLayer, LRLayer

TEST_CNT_MOD = 500


class MLLP(nn.Module):
    def __init__(self, dim_list, use_not=False, cl=None, cr=None, left=None, right=None, estimated_grad=False):
        super(MLLP, self).__init__()

        self.dim_list = dim_list
        self.use_not = use_not
        self.left = left
        self.right = right
        self.layer_list = nn.ModuleList([])

        prev_layer_dim = dim_list[0]
        for i in range(1, len(dim_list)):
            num = prev_layer_dim
            if i >= 4:
                num += self.layer_list[-2].output_dim

            if i == 1:
                layer = BinarizeLayer(dim_list[i], num, self.use_not, cl=cl, cr=cr, left=self.left, right=self.right)
                layer_name = 'binary{}'.format(i)
            elif i == len(dim_list) - 1:
                layer = LRLayer(dim_list[i], num)
                layer_name = 'lr{}'.format(i)
            else:
                layer = UnionLayer(dim_list[i], num, estimated_grad=estimated_grad)
                layer_name = 'union{}'.format(i)
            prev_layer_dim = layer.output_dim
            self.add_module(layer_name, layer)
            self.layer_list.append(layer)

    def forward(self, x):
        return self.continuous_forward(x), self.binarized_forward(x)

    def continuous_forward(self, x):
        x_res = None
        for i, layer in enumerate(self.layer_list):
            if i <= 1:
                x = layer(x)
            else:
                x_cat = torch.cat([x, x_res], dim=1) if x_res is not None else x
                x_res = x
                x = layer(x_cat)
        return x

    def binarized_forward(self, x):
        with torch.no_grad():
            x_res = None
            for i, layer in enumerate(self.layer_list):
                if i <= 1:
                    x = layer.binarized_forward(x)
                else:
                    x_cat = torch.cat([x, x_res], dim=1) if x_res is not None else x
                    x_res = x
                    x = layer.binarized_forward(x_cat)
            return x


class MyDistributedDataParallel(torch.nn.parallel.DistributedDataParallel):
    @property
    def layer_list(self):
        return self.module.layer_list


class RRL:
    def __init__(self, dim_list, device_id, use_not=False, is_rank0=False, log_file=None, writer=None, left=None,
                 right=None, cl=None, cr=None, save_best=False, estimated_grad=False, save_path=None, distributed=True):
        super(RRL, self).__init__()
        self.dim_list = dim_list
        self.use_not = use_not
        self.best_f1 = -1.

        self.device_id = device_id
        self.is_rank0 = is_rank0
        self.save_best = save_best
        self.estimated_grad = estimated_grad
        self.save_path = save_path
        if self.is_rank0:
            for handler in logging.root.handlers[:]:
                logging.root.removeHandler(handler)

            log_format = '%(asctime)s - [%(levelname)s] - %(message)s'
            if log_file is None:
                logging.basicConfig(level=logging.DEBUG, stream=sys.stdout, format=log_format)
            else:
                logging.basicConfig(level=logging.DEBUG, filename=log_file, filemode='w', format=log_format)
        self.writer = writer

        self.net = MLLP(dim_list, use_not=use_not, cl=cl, cr=cr, left=left, right=right,
                        estimated_grad=estimated_grad)

        if self.device_id and self.device_id.type == 'cuda':
            self.net.cuda(self.device_id)
        if distributed:
            self.net = MyDistributedDataParallel(self.net, device_ids=[self.device_id])

    def clip(self):
        """Clip the weights into the range [0, 1]."""
        for layer in self.net.layer_list[: -1]:
            layer.clip()

    def data_transform(self, X, y):
        X = X.astype(np.float)
        if y is None:
            return torch.tensor(X)
        y = y.astype(np.float)
        return torch.tensor(X), torch.tensor(y)

    @staticmethod
    def exp_lr_scheduler(optimizer, epoch, init_lr=0.001, lr_decay_rate=0.9, lr_decay_epoch=7):
        """Decay learning rate by a factor of lr_decay_rate every lr_decay_epoch epochs."""
        lr = init_lr * (lr_decay_rate ** (epoch // lr_decay_epoch))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return optimizer

    def train_model(self, X=None, y=None, X_validation=None, y_validation=None, data_loader=None, valid_loader=None,
                    epoch=50, lr=0.01, lr_decay_epoch=100, lr_decay_rate=0.75, batch_size=64, weight_decay=0.0,
                    log_iter=50):

        if (X is None or y is None) and data_loader is None:
            raise Exception("Both data set and data loader are unavailable.")
        if data_loader is None:
            X, y = self.data_transform(X, y)
            if X_validation is not None and y_validation is not None:
                X_validation, y_validation = self.data_transform(X_validation, y_validation)
            data_loader = DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=True)

        loss_log = []
        accuracy = []
        accuracy_b = []
        f1_score = []
        f1_score_b = []

        criterion = nn.CrossEntropyLoss().cuda(self.device_id)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=lr, weight_decay=weight_decay)

        cnt = -1
        avg_batch_loss_mllp = 0.0
        avg_batch_loss_rrl = 0.0
        epoch_histc = defaultdict(list)
        for epo in range(epoch):
            optimizer = self.exp_lr_scheduler(optimizer, epo, init_lr=lr, lr_decay_rate=lr_decay_rate,
                                              lr_decay_epoch=lr_decay_epoch)
            epoch_loss_mllp = 0.0
            epoch_loss_rrl = 0.0
            abs_gradient_max = 0.0
            abs_gradient_avg = 0.0

            ba_cnt = 0
            for X, y in data_loader:
                ba_cnt += 1
                if self.device_id and self.device_id.type == 'cuda':
                    X = X.cuda(self.device_id, non_blocking=True)
                    y = y.cuda(self.device_id, non_blocking=True)
                optimizer.zero_grad()  # Zero the gradient buffers.
                y_pred_mllp, y_pred_rrl = self.net.forward(X)
                with torch.no_grad():
                    y_prob = torch.softmax(y_pred_rrl, dim=1)
                    y_arg = torch.argmax(y, dim=1)
                    loss_mllp = criterion(y_pred_mllp, y_arg)
                    loss_rrl = criterion(y_pred_rrl, y_arg)
                    ba_loss_mllp = loss_mllp.item()
                    ba_loss_rrl = loss_rrl.item()
                    epoch_loss_mllp += ba_loss_mllp
                    epoch_loss_rrl += ba_loss_rrl
                    avg_batch_loss_mllp += ba_loss_mllp
                    avg_batch_loss_rrl += ba_loss_rrl
                y_pred_mllp.backward((y_prob - y) / y.shape[0])  # for CrossEntropy Loss
                cnt += 1

                if self.is_rank0 and cnt % log_iter == 0 and cnt != 0 and self.writer is not None:
                    self.writer.add_scalar('Avg_Batch_Loss_MLLP', avg_batch_loss_mllp / log_iter, cnt)
                    self.writer.add_scalar('Avg_Batch_Loss_GradGrafting', avg_batch_loss_rrl / log_iter, cnt)
                    avg_batch_loss_mllp = 0.0
                    avg_batch_loss_rrl = 0.0
                optimizer.step()
                if self.is_rank0:
                    for i, param in enumerate(self.net.parameters()):
                        abs_gradient_max = max(abs_gradient_max, abs(torch.max(param.grad)))
                        abs_gradient_avg += torch.sum(torch.abs(param.grad)) / (param.grad.numel())
                self.clip()

                if self.is_rank0 and cnt % TEST_CNT_MOD == 0:
                    if X_validation is not None and y_validation is not None:
                        acc, acc_b, f1, f1_b = self.test(X_validation, y_validation, batch_size=batch_size,
                                                         need_transform=False, set_name='Validation')
                    elif valid_loader is not None:
                        acc, acc_b, f1, f1_b = self.test(test_loader=valid_loader, need_transform=False,
                                                         set_name='Validation')
                    elif data_loader is not None:
                        acc, acc_b, f1, f1_b = self.test(test_loader=data_loader, need_transform=False,
                                                         set_name='Training')
                    else:
                        acc, acc_b, f1, f1_b = self.test(X, y, batch_size=batch_size, need_transform=False,
                                                         set_name='Training')
                    if self.save_best and f1_b > self.best_f1:
                        self.best_f1 = f1_b
                        self.save_model()
                    accuracy.append(acc)
                    accuracy_b.append(acc_b)
                    f1_score.append(f1)
                    f1_score_b.append(f1_b)
                    if self.writer is not None:
                        self.writer.add_scalar('Accuracy_MLLP', acc, cnt // TEST_CNT_MOD)
                        self.writer.add_scalar('Accuracy_RRL', acc_b, cnt // TEST_CNT_MOD)
                        self.writer.add_scalar('F1_Score_MLLP', f1, cnt // TEST_CNT_MOD)
                        self.writer.add_scalar('F1_Score_RRL', f1_b, cnt // TEST_CNT_MOD)
            if self.is_rank0:
                logging.info('epoch: {}, loss_mllp: {}, loss_rrl: {}'.format(epo, epoch_loss_mllp, epoch_loss_rrl))
                for name, param in self.net.named_parameters():
                    maxl = 1 if 'con_layer' in name or 'dis_layer' in name else 0
                    epoch_histc[name].append(torch.histc(param.data, bins=10, max=maxl).cpu().numpy())
                if self.writer is not None:
                    self.writer.add_scalar('Training_Loss_MLLP', epoch_loss_mllp, epo)
                    self.writer.add_scalar('Training_Loss_RRL', epoch_loss_rrl, epo)
                    self.writer.add_scalar('Abs_Gradient_Max', abs_gradient_max, epo)
                    self.writer.add_scalar('Abs_Gradient_Avg', abs_gradient_avg / ba_cnt, epo)
                loss_log.append(epoch_loss_rrl)
        if self.is_rank0 and not self.save_best:
            self.save_model()
        return epoch_histc

    def test(self, X=None, y=None, labels=None, test_loader=None,
             batch_size=32, need_transform=True, set_name='Validation'):
        if X is not None and y is not None and need_transform:
            X, y = self.data_transform(X, y)
        with torch.no_grad():
            if X is not None and y is not None:
                test_loader = DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=False)

            y_list = []
            for X, y in test_loader:
                y_list.append(y)
            y_true = torch.cat(y_list, dim=0)
            y_true = y_true.cpu().numpy().astype(np.int)
            y_true = np.argmax(y_true, axis=1)
            data_num = y_true.shape[0]
            slice_step = data_num // 40 if data_num >= 40 else 1
            logging.debug('y_true: {} {}'.format(y_true.shape, y_true[:: slice_step]))

            y_pred_list = []
            y_pred_b_list = []
            for X, y in test_loader:
                if self.device_id and self.device_id.type == 'cuda':
                    X = X.cuda(self.device_id, non_blocking=True)
                output = self.net.forward(X)
                y_pred_list.append(output[0])
                y_pred_b_list.append(output[1])

            y_pred = torch.cat(y_pred_list).cpu().numpy()
            y_pred = np.argmax(y_pred, axis=1)
            logging.debug('y_mllp: {} {}'.format(y_pred.shape, y_pred[:: slice_step]))

            y_pred_b = torch.cat(y_pred_b_list).cpu().numpy()
            y_pred_b_arg = np.argmax(y_pred_b, axis=1)
            logging.debug('y_rrl_: {} {}'.format(y_pred_b_arg.shape, y_pred_b_arg[:: slice_step]))
            logging.debug('y_rrl: {} {}'.format(y_pred_b.shape, y_pred_b[:: (slice_step)]))

            accuracy = metrics.accuracy_score(y_true, y_pred)
            accuracy_b = metrics.accuracy_score(y_true, y_pred_b_arg)

            f1_score = metrics.f1_score(y_true, y_pred, average='macro')
            f1_score_b = metrics.f1_score(y_true, y_pred_b_arg, average='macro')

            logging.info('-' * 60)
            logging.info('On {} Set:\n\tAccuracy of RRL  Model: {}'
                         '\n\tF1 Score of RRL  Model: {}'.format(set_name, accuracy_b, f1_score_b))
            logging.info('On {} Set:\nPerformance of  RRL Model: \n{}\n{}'.format(
                set_name, metrics.confusion_matrix(y_true, y_pred_b_arg),
                metrics.classification_report(y_true, y_pred_b_arg, target_names=labels)))
            logging.info('-' * 60)
        return accuracy, accuracy_b, f1_score, f1_score_b

    def save_model(self):
        rrl_args = {'dim_list': self.dim_list, 'use_not': self.use_not, 'estimated_grad': self.estimated_grad}
        torch.save({'model_state_dict': self.net.state_dict(), 'rrl_args': rrl_args}, self.save_path)

    def detect_dead_node(self, data_loader=None):
        with torch.no_grad():
            for layer in self.net.layer_list[:-1]:
                layer.node_activation_cnt = torch.zeros(layer.output_dim, dtype=torch.double, device=self.device_id)
                layer.forward_tot = 0
            for x, y in data_loader:
                if self.device_id and self.device_id.type == 'cuda':
                    x = x.cuda(self.device_id)
                x_res = None
                for i, layer in enumerate(self.net.layer_list[:-1]):
                    if i <= 1:
                        x = layer.binarized_forward(x)
                    else:
                        x_cat = torch.cat([x, x_res], dim=1) if x_res is not None else x
                        x_res = x
                        x = layer.binarized_forward(x_cat)
                    layer.node_activation_cnt += torch.sum(x, dim=0)
                    layer.forward_tot += x.shape[0]

    def rule_print(self, feature_name, label_name, train_loader, file=sys.stdout, mean=None, std=None):
        if self.net.layer_list[1] is None and train_loader is None:
            raise Exception("Need train_loader for the dead nodes detection.")
        if self.net.layer_list[1].node_activation_cnt is None:
            self.detect_dead_node(train_loader)

        bound_name = self.net.layer_list[0].get_bound_name(feature_name, mean, std)
        self.net.layer_list[1].get_rules(self.net.layer_list[0], None)
        self.net.layer_list[1].get_rule_description((None, bound_name))

        if len(self.net.layer_list) >= 4:
            self.net.layer_list[2].get_rules(self.net.layer_list[1], None)
            self.net.layer_list[2].get_rule_description((None, self.net.layer_list[1].rule_name), wrap=True)

        if len(self.net.layer_list) >= 5:
            for i in range(3, len(self.net.layer_list) - 1):
                self.net.layer_list[i].get_rules(self.net.layer_list[i - 1], self.net.layer_list[i - 2])
                self.net.layer_list[i].get_rule_description(
                    (self.net.layer_list[i - 2].rule_name, self.net.layer_list[i - 1].rule_name), wrap=True)

        prev_layer = self.net.layer_list[-2]
        skip_connect_layer = self.net.layer_list[-3]
        always_act_pos = (prev_layer.node_activation_cnt == prev_layer.forward_tot)
        if skip_connect_layer.layer_type == 'union':
            shifted_dim2id = {(k + prev_layer.output_dim): (-2, v) for k, v in skip_connect_layer.dim2id.items()}
            prev_dim2id = {k: (-1, v) for k, v in prev_layer.dim2id.items()}
            merged_dim2id = defaultdict(lambda: -1, {**shifted_dim2id, **prev_dim2id})
            always_act_pos = torch.cat(
                [always_act_pos, (skip_connect_layer.node_activation_cnt == skip_connect_layer.forward_tot)])
        else:
            merged_dim2id = {k: (-1, v) for k, v in prev_layer.dim2id.items()}

        Wl, bl = list(self.net.layer_list[-1].parameters())
        bl = torch.sum(Wl.T[always_act_pos], dim=0) + bl
        Wl = Wl.cpu().detach().numpy()
        bl = bl.cpu().detach().numpy()

        marked = defaultdict(lambda: defaultdict(float))
        rid2dim = {}
        for label_id, wl in enumerate(Wl):
            for i, w in enumerate(wl):
                rid = merged_dim2id[i]
                if rid == -1 or rid[1] == -1:
                    continue
                marked[rid][label_id] += w
                rid2dim[rid] = i % prev_layer.output_dim

        kv_list = sorted(marked.items(), key=lambda x: max(map(abs, x[1].values())), reverse=True)
        print('RID', end='\t', file=file)
        for i, ln in enumerate(label_name):
            print('{}(b={:.4f})'.format(ln, bl[i]), end='\t', file=file)
        print('Support\tRule', file=file)
        for k, v in kv_list:
            rid = k
            print(rid, end='\t', file=file)
            for li in range(len(label_name)):
                print('{:.4f}'.format(v[li]), end='\t', file=file)
            now_layer = self.net.layer_list[-1 + rid[0]]
            # print('({},{})'.format(now_layer.node_activation_cnt[rid2dim[rid]].item(), now_layer.forward_tot))
            print('{:.4f}'.format((now_layer.node_activation_cnt[rid2dim[rid]] / now_layer.forward_tot).item()),
                  end='\t', file=file)
            print(now_layer.rule_name[rid[1]], end='\n', file=file)
        print('#' * 60, file=file)
        return kv_list
