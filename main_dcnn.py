#!/usr/bin/env python
from __future__ import print_function
import argparse
import os
import time
import numpy as np
import yaml
import pickle
from collections import OrderedDict, defaultdict
# torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm
from tensorboardX import SummaryWriter
import shutil
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.lr_scheduler import _LRScheduler
import random
import inspect
import torch.backends.cudnn as cudnn
import gc
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter


class GradualWarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, total_epoch, after_scheduler=None):
        self.total_epoch = total_epoch  # warm_UP epochs
        self.after_scheduler = after_scheduler
        self.finished = False
        self.last_epoch = -1
        super().__init__(optimizer)

    def get_lr(self):
        return [base_lr * (self.last_epoch + 1) / self.total_epoch for base_lr in self.base_lrs]

    def step(self, epoch=None, metric=None):
        if self.last_epoch >= self.total_epoch - 1:
            if metric is None:
                return self.after_scheduler.step(epoch)
            else:
                return self.after_scheduler.step(metric, epoch)
        else:
            return super(GradualWarmupScheduler, self).step(epoch)


def init_seed(_):
    torch.cuda.manual_seed_all(1)
    torch.manual_seed(1)
    np.random.seed(1)
    random.seed(1)
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_parser():
    # parameter priority: command line > config > default
    parser = argparse.ArgumentParser(
        description='Directed Graph Neural Net for Skeleton Action Recognition')
    parser.add_argument(
        '--work-dir',
        default='./work_dir/temp',
        help='the work folder for storing results')

    parser.add_argument(
        '--model-saved-name', default='')
    parser.add_argument(
        '--config',
        default='./config/nturgbd-cross-view/test_bone.yaml',
        help='path to the configuration file')

    # processor
    parser.add_argument(
        '--phase', default='train', help='must be train or test')
    parser.add_argument(
        '--save-score',
        type=str2bool,
        default=False,
        help='if ture, the classification score will be stored')

    # visulize and debug
    parser.add_argument(
        '--seed', type=int, default=1, help='random seed for pytorch')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=100,
        help='the interval for printing messages (#iteration)')
    parser.add_argument(
        '--save-interval',
        type=int,
        default=3,
        help='the interval for storing models (#iteration)')
    parser.add_argument(
        '--eval-interval',
        type=int,
        default=1,
        help='the interval for evaluating models (#iteration)')
    parser.add_argument(
        '--print-log',
        type=str2bool,
        default=True,
        help='print logging or not')
    parser.add_argument(
        '--show-topk',
        type=int,
        default=[1, 5],
        nargs='+',
        help='which Top K accuracy will be shown')

    # feeder
    parser.add_argument(
        '--feeder', default='feeder.feeder', help='data loader will be used')
    parser.add_argument(
        '--num-worker',
        type=int,
        default=os.cpu_count(),
        help='the number of worker for data loader')
    parser.add_argument(
        '--train-feeder-args',
        default=dict(),
        help='the arguments of data loader for training')
    parser.add_argument(
        '--test-feeder-args',
        default=dict(),
        help='the arguments of data loader for test')

    # model
    parser.add_argument(
        '--model', default=None, help='the model will be used')
    parser.add_argument(
        '--model-args',
        type=dict,
        default=dict(),
        help='the arguments of model')
    parser.add_argument(
        '--weights',
        default=None,
        help='the weights for network initialization')
    parser.add_argument(
        '--ignore-weights',
        type=str,
        default=[],
        nargs='+',
        help='the name of weights which will be ignored in the initialization')

    # optim
    parser.add_argument(
        '--base-lr', type=float, default=0.01, help='initial learning rate')
    parser.add_argument(
        '--step',
        type=int,
        default=[15, 30, 45],
        nargs='+',
        help='the epoch where optimizer reduce the learning rate')
    parser.add_argument(
        '--device',
        type=int,
        default=0,
        nargs='+',
        help='the indexes of GPUs for training or testing')
    parser.add_argument(
        '--optimizer', default='SGD', help='type of optimizer')
    parser.add_argument(
        '--nesterov', type=str2bool, default=True, help='use nesterov or not')
    parser.add_argument(
        '--batch-size', type=int, default=32, help='training batch size')
    parser.add_argument(
        '--test-batch-size', type=int, default=32, help='test batch size')
    parser.add_argument(
        '--start-epoch',
        type=int,
        default=0,
        help='start training from which epoch')
    parser.add_argument(
        '--num-epoch',
        type=int,
        default=120,
        help='stop training in which epoch')
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=0.0005,
        help='weight decay for optimizer')
    parser.add_argument(
        '--freeze-graph-until',
        type=int,
        default=10,
        help='number of epochs before making graphs learnable')

    parser.add_argument('--only_train_part', default=False)
    parser.add_argument('--only_train_epoch', default=0)
    parser.add_argument('--warm_up_epoch', default=0)
    return parser


class Processor():
    """Processor for Skeleton-based Action Recgnition"""

    def __init__(self, arg):
        self.arg = arg
        self.save_arg()
        if arg.phase == 'train':
            if not arg.train_feeder_args['debug']:
                if os.path.isdir(arg.model_saved_name):
                    print('log_dir: ', arg.model_saved_name, 'already exist')
                    answer = input('delete it? [y]/n:')
                    if answer.lower() in ('y', ''):
                        shutil.rmtree(arg.model_saved_name)
                        print('Dir removed: ', arg.model_saved_name)
                        input(
                            'Refresh the website of tensorboard by pressing any keys')
                    else:
                        print('Dir not removed: ', arg.model_saved_name)

            self.train_writer = SummaryWriter(
                os.path.join(arg.model_saved_name, 'train'), 'train')
            self.val_writer = SummaryWriter(
                os.path.join(arg.model_saved_name, 'val'), 'val')
            # self.writer = SummaryWriter(os.path.join(arg.model_saved_name, 'training'), 'both')

        self.global_step = 0
        self.load_model()
        self.load_param_groups()    # Group parameters to apply different learning rules
        self.load_data()
        self.load_optimizer()
        self.lr = self.arg.base_lr
        self.best_acc = 0
        self.best_acc_epoch = 0
        self.train_loss_values = []
        self.train_acc_values = []
        self.val_loss_values = []
        self.val_acc_values = []
        self.lr_values = []

    def load_data(self):
        Feeder = import_class(self.arg.feeder)
        self.data_loader = dict()
        if self.arg.phase == 'train':
            self.data_loader['train'] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.arg.train_feeder_args),
                batch_size=self.arg.batch_size,
                shuffle=True,
                num_workers=self.arg.num_worker,
                drop_last=True,
                worker_init_fn=init_seed)

        # Load test data regardless
        self.data_loader['test'] = torch.utils.data.DataLoader(
            dataset=Feeder(**self.arg.test_feeder_args),
            batch_size=self.arg.test_batch_size,
            shuffle=False,
            num_workers=self.arg.num_worker,
            drop_last=False,
            worker_init_fn=init_seed)

    def load_model(self):
        output_device = self.arg.device[0] if type(
            self.arg.device) is list else self.arg.device
        self.output_device = output_device
        Model = import_class(self.arg.model)
        # Copy model file to output dir
        shutil.copy2(inspect.getfile(Model), self.arg.work_dir)
        print(Model)
        self.model = Model(**self.arg.model_args).cuda(output_device)
        self.loss = nn.CrossEntropyLoss().cuda(output_device)

        # Load weights
        if self.arg.weights:
            self.global_step = int(arg.weights[:-3].split('-')[-1])
            self.print_log('Load weights from {}.'.format(self.arg.weights))
            if '.pkl' in self.arg.weights:
                with open(self.arg.weights, 'r') as f:
                    weights = pickle.load(f)
            else:
                weights = torch.load(self.arg.weights)

            weights = OrderedDict(
                [[k.split('module.')[-1],
                  v.cuda(output_device)] for k, v in weights.items()])

            for w in self.arg.ignore_weights:
                if weights.pop(w, None) is not None:
                    self.print_log('Sucessfully Remove Weights: {}.'.format(w))
                else:
                    self.print_log('Can Not Remove Weights: {}.'.format(w))

            try:
                self.model.load_state_dict(weights)
            except:
                state = self.model.state_dict()
                diff = list(set(state.keys()).difference(set(weights.keys())))
                print('Can not find these weights:')
                for d in diff:
                    print('  ' + d)
                state.update(weights)
                self.model.load_state_dict(state)

        # Parallelise data if multiple GPUs
        if type(self.arg.device) is list:
            if len(self.arg.device) > 1:
                # print("============================================")
                self.model = nn.DataParallel(
                    self.model,
                    device_ids=self.arg.device,
                    output_device=output_device)

    def load_optimizer(self):
        p_groups = list(self.optim_param_groups.values())
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                p_groups,
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)

        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                p_groups,
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        else:
            raise ValueError(
                'Unsupported optimizer: {}'.format(self.arg.optimizer))
        # self.scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=0.01, max_lr=0.1)
        # Warmup StepLR
        lr_scheduler = MultiStepLR(
            self.optimizer, milestones=self.arg.step, gamma=0.1)
        self.lr_scheduler = GradualWarmupScheduler(self.optimizer, total_epoch=self.arg.warm_up_epoch,
                                                   after_scheduler=lr_scheduler)
        self.print_log('using warm up, epoch: {}'.format(
            self.arg.warm_up_epoch))

        # OneCycleLR
        # self.OC_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        #     self.optimizer, max_lr=0.1, steps_per_epoch=len(self.data_loader['train']), epochs=self.arg.num_epoch)

    def save_arg(self):
        # save arg
        arg_dict = vars(self.arg)
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)
        with open('{}/config.yaml'.format(self.arg.work_dir), 'w') as f:
            yaml.dump(arg_dict, f)

    def adjust_learning_rate(self, epoch):
        if self.arg.optimizer == 'SGD' or self.arg.optimizer == 'Adam':
            if epoch < self.arg.warm_up_epoch:
                lr = self.arg.base_lr * (epoch + 1) / self.arg.warm_up_epoch
            else:
                lr = self.arg.base_lr * (
                    0.1 ** np.sum(epoch >= np.array(self.arg.step)))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            return lr
        else:
            raise ValueError()

    def print_time(self):
        localtime = time.asctime(time.localtime(time.time()))
        self.print_log("Local current time :  " + localtime)

    def print_log(self, s, print_time=True):
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            s = '[ {} ] {}'.format(localtime, s)
        print(s)
        if self.arg.print_log:
            with open('{}/log.txt'.format(self.arg.work_dir), 'a') as f:
                print(s, file=f)

    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time

    def load_param_groups(self):
        self.param_groups = defaultdict(list)
        for name, params in self.model.named_parameters():
            if ('source_M' in name) or ('target_M' in name):
                self.param_groups['graph'].append(params)
            else:
                self.param_groups['other'].append(params)

        # NOTE: Different parameter groups should have different learning behaviour
        self.optim_param_groups = {
            'graph': {'params': self.param_groups['graph']},
            'other': {'params': self.param_groups['other']}
        }

    def update_graph_freeze(self, epoch):
        graph_requires_grad = (epoch > self.arg.freeze_graph_until)
        self.print_log('Graphs are {} at epoch {}'.format(
            'learnable' if graph_requires_grad else 'frozen', epoch + 1))
        for param in self.param_groups['graph']:
            param.requires_grad = graph_requires_grad
        # graph_weight_decay = 0 if freeze_graphs else self.arg.weight_decay
        # NOTE: will decide later whether we need to change weight decay as well
        # self.optim_param_groups['graph']['weight_decay'] = graph_weight_decay

    def train(self, epoch, save_model=False):
        self.print_log('Training epoch: {}'.format(epoch + 1))
        self.model.train()
        loader = self.data_loader['train']

        self.adjust_learning_rate(epoch)

        loss_values = []
        self.train_writer.add_scalar('epoch', epoch, self.global_step)
        self.record_time()
        timer = dict(dataloader=0.001, model=0.001, statistics=0.001)

        if self.arg.only_train_part:
            if epoch > self.arg.only_train_epoch:
                print('only train part, require grad')
                for key, value in self.model.named_parameters():
                    if 'PA_s' in key:
                        value.requires_grad = True
                    if 'PA_t' in key:
                        value.requires_grad = True
            else:
                print('only train part, do not require grad')
                for key, value in self.model.named_parameters():
                    if 'PA_s' in key:
                        value.requires_grad = True
                    if 'PA_t' in key:
                        value.requires_grad = True

        self.update_graph_freeze(epoch)

        process = tqdm(loader)
        # for batch_idx, (data, label, index) in enumerate(process):
        for batch_idx, (joint_data, bone_data, label, index) in enumerate(process):
            self.global_step += 1
            # # get data
            with torch.no_grad():
                joint_data = joint_data.float().cuda(self.output_device)
                bone_data = bone_data.float().cuda(self.output_device)
                label = label.long().cuda(self.output_device)

            # data = np.stack((joint_data.numpy(), bone_data.numpy()), axis=1)
            # with torch.no_grad():
            #     data = torch.tensor(data).float().cuda(self.output_device)
            #     label = label.long().cuda(self.output_device)
            # timer['dataloader'] += self.split_time()

            # Clear gradients
            self.optimizer.zero_grad()

            ################################
            # Multiple forward passes + 1 backward pass to simulate larger batch size
            real_batch_size = 16
            splits = len(joint_data) // real_batch_size
            assert len(
                joint_data) % real_batch_size == 0, 'Real batch size should be a factor of arg.batch_size!'

            for i in range(splits):
                left = i * real_batch_size
                right = left + real_batch_size
                batch_joint_data, batch_bone_data = joint_data[left:right], bone_data[left:right]
                batch_label = label[left:right]

                # forward
                output = self.model(batch_joint_data, batch_bone_data)
                if isinstance(output, tuple):
                    output, l1 = output
                    l1 = l1.mean()
                else:
                    l1 = 0

                loss = self.loss(output, batch_label) / float(splits) + l1
                loss.backward()

                loss_values.append(loss.item())
                self.train_loss_values.append(loss.item())
                timer['model'] += self.split_time()

                # Display loss
                process.set_description('loss: {:.4f}'.format(loss.item()))

                value, predict_label = torch.max(output, 1)
                acc = torch.mean((predict_label == batch_label).float())
                self.train_acc_values.append(acc.item())
                self.train_writer.add_scalar('acc', acc, self.global_step)
                self.train_writer.add_scalar(
                    'loss', loss.item(), self.global_step)
                self.train_writer.add_scalar('loss_l1', l1, self.global_step)

            # Step after looping over batch splits
            # self.scheduler.step()
            self.optimizer.step()
            # oc_scheduler
            # self.OC_scheduler.step()

            # statistics
            self.lr = self.optimizer.param_groups[0]['lr']
            self.lr_values.append(self.lr)
            self.train_writer.add_scalar('lr', self.lr, self.global_step)
            timer['statistics'] += self.split_time()

        # statistics of time consumption and loss
        proportion = {
            k: '{: 2d}%'.format(int(round(v * 100 / sum(timer.values()))))
            for k, v in timer.items()
        }
        self.print_log('\tMean training loss: {:.4f}.'.format(
            np.mean(loss_values)))
        self.print_log(
            '\tTime consumption: [Data]{dataloader}, [Network]{model}'.format(**proportion))
        # Warmup scheduler
        # self.lr_scheduler.step(epoch)

        if save_model:
            state_dict = self.model.state_dict()
            weights = OrderedDict([[k.split('module.')[-1], v.cpu()]
                                  for k, v in state_dict.items()])
            torch.save(weights, self.arg.model_saved_name + '-' +
                       str(epoch) + '-' + str(int(self.global_step)) + '.pt')
        # gc.collect()
        # torch.cuda.empty_cache()

    def eval(self, epoch, save_score=False, loader_name=['test'], wrong_file=None, result_file=None):
        if wrong_file is not None:
            f_w = open(wrong_file, 'w')
        if result_file is not None:
            f_r = open(result_file, 'w')

        self.model.eval()
        self.print_log('Eval epoch: {}'.format(epoch + 1))

        for ln in loader_name:
            loss_values, score_batches = [], []
            step = 0
            process = tqdm(self.data_loader[ln])
            for batch_idx, (joint_data, bone_data, label, index) in enumerate(process):
                step += 1
                with torch.no_grad():
                    joint_data = joint_data.float().cuda(self.output_device)
                    bone_data = bone_data.float().cuda(self.output_device)
                    label = label.long().cuda(self.output_device)
                    output = self.model(joint_data, bone_data)
                    if isinstance(output, tuple):
                        output, l1 = output
                        l1 = l1.mean()
                    else:
                        l1 = 0
                    loss = self.loss(output, label)
                    score_batches.append(output.cpu().numpy())
                    loss_values.append(loss.data.item())

                    self.val_loss_values.append(loss.data.item())
                    # for idx, i in enumerate(self.val_loss_values):
                    #     if i > 2:
                    #         self.val_loss_values[idx] = 0
                    # Argmax over logits = labels
                    _, predict_label = torch.max(output, dim=1)

                    if wrong_file is not None or result_file is not None:
                        predict = list(predict_label.cpu().numpy())
                        true = list(label.cpu().numpy())
                        for i, pred in enumerate(predict):
                            if result_file is not None:
                                f_r.write('{},{}\n'.format(pred, true[i]))
                            if pred != true[i] and wrong_file is not None:
                                f_w.write('{},{},{}\n'.format(
                                    index[i], pred, true[i]))

            # Concatenate along the batch dimension, and 1st dim ~= `len(dataset)`
            score = np.concatenate(score_batches)
            loss = np.mean(loss_values)
            accuracy = self.data_loader[ln].dataset.top_k(score, 1)

            self.val_acc_values.append(accuracy)
            if accuracy > self.best_acc:
                self.best_acc = accuracy
                self.best_acc_epoch = epoch

            print('Accuracy: ', accuracy, ' Model: ', self.arg.model_saved_name)
            if self.arg.phase == 'train':
                self.val_writer.add_scalar('loss', loss, self.global_step)
                self.val_writer.add_scalar('loss_l1', l1, self.global_step)
                self.val_writer.add_scalar('acc', accuracy, self.global_step)

            self.print_log('\tMean {} loss of {} batches: {}.'.format(
                ln, len(self.data_loader[ln]), np.mean(loss_values)))

            for k in self.arg.show_topk:
                self.print_log('\tTop{}: {:.2f}%'.format(
                    k, 100 * self.data_loader[ln].dataset.top_k(score, k)))

            if save_score:
                score_dict = dict(
                    zip(self.data_loader[ln].dataset.sample_name, score))
                with open('{}/epoch{}_{}_score.pkl'.format(
                        self.arg.work_dir, epoch + 1, ln), 'wb') as f:
                    pickle.dump(score_dict, f)

    def plot_metrics(self):
        epochs = range(1, len(self.train_loss_values) + 1)
        val_epochs = range(1, len(self.val_loss_values) + 1)

        fig, axs = plt.subplots(2, 2, figsize=(12, 10))

        axs[0, 0].plot(epochs, self.train_loss_values, label='Training Loss')
        axs[0, 0].set_xlabel('Epochs')
        axs[0, 0].set_ylabel('Loss')
        axs[0, 0].legend()
        axs[0, 0].set_title('Training Loss')

        axs[0, 1].plot(val_epochs, self.val_loss_values,
                       label='Validation Loss')
        axs[0, 1].set_xlabel('Epochs')
        axs[0, 1].set_ylabel('Loss')
        axs[0, 1].legend()
        axs[0, 1].set_title('Validation Loss')

        axs[1, 0].plot(epochs, self.train_acc_values,
                       label='Training Accuracy')
        axs[1, 0].set_xlabel('Epochs')
        axs[1, 0].set_ylabel('Accuracy')
        axs[1, 0].legend()
        axs[1, 0].set_title('Training Accuracy')

        axs[1, 1].plot(range(len(self.val_acc_values)),
                       self.val_acc_values, label='Val Acc')
        axs[1, 1].set_xlabel('Epoch')
        axs[1, 1].set_ylabel('A')
        axs[1, 1].legend()
        axs[1, 1].set_title('Val Acc')

        plt.tight_layout()
        plt.savefig(f'{self.arg.work_dir}/metrics_plot.png')
        plt.close()

    def start(self):
        if self.arg.phase == 'train':
            self.print_log('Parameters:\n{}\n'.format(str(vars(self.arg))))
            self.global_step = self.arg.start_epoch * \
                len(self.data_loader['train']) / self.arg.batch_size
            # print(range(self.arg.start_epoch, self.arg.num_epoch))
            for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                if self.lr < 1e-3:
                    break
                save_model = ((epoch + 1) % self.arg.save_interval == 0) or (
                    epoch + 1 == self.arg.num_epoch)

                self.train(epoch, save_model=save_model)
                if (epoch + 1) % self.arg.eval_interval == 0:
                    self.eval(epoch, save_score=self.arg.save_score,
                              loader_name=['test'])
                # self.eval(epoch, save_score=self.arg.save_score,
                #           loader_name=['test'])
                self.plot_metrics()

            print('Best accuracy: {}, epoch: {}, model_name: {}'
                  .format(self.best_acc, self.best_acc_epoch, self.arg.model_saved_name))

        elif self.arg.phase == 'test':
            if not self.arg.test_feeder_args['debug']:
                wf = self.arg.model_saved_name + '_wrong.txt'
                rf = self.arg.model_saved_name + '_right.txt'
            else:
                wf = rf = None
            if self.arg.weights is None:
                raise ValueError('Please appoint --weights.')
            self.arg.print_log = False
            self.print_log('Model:   {}.'.format(self.arg.model))
            self.print_log('Weights: {}.'.format(self.arg.weights))
            self.eval(epoch=0, save_score=self.arg.save_score,
                      loader_name=['test'], wrong_file=wf, result_file=rf)
            self.print_log('Done.\n')


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])  # import return model
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


if __name__ == '__main__':
    parser = get_parser()

    # load arg form config file
    p = parser.parse_args()
    if p.config is not None:
        with open(p.config, 'r') as f:
            default_arg = yaml.load(f, Loader=yaml.FullLoader)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                assert (k in key)
        parser.set_defaults(**default_arg)

    arg = parser.parse_args()
    init_seed(0)
    processor = Processor(arg)
    processor.start()
