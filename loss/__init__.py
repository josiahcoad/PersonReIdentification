import os
import numpy as np
from importlib import import_module

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from loss.triplet import TripletLoss, TripletSemihardLoss, AlignedTripletLoss, TripletLoss2

import pdb

class Loss(nn.modules.loss._Loss):
    def __init__(self, args, ckpt):
        super(Loss, self).__init__()
        print('[INFO] Making loss...')

        self.nGPU = args.nGPU
        self.args = args
        self.loss = []
        self.loss_module = nn.ModuleList()
        for loss in args.loss.split('+'):
            weight, loss_type = loss.split('*')
            if loss_type == 'CrossEntropy':
                loss_function = nn.CrossEntropyLoss()
            elif loss_type == 'Triplet':
                loss_function = TripletLoss(args.margin)
            # CSCE 625: Aligned loss evaluation for aligned features
            elif loss_type == 'AlignedTriplet':
                tri_loss = TripletLoss2(margin=0.3)
                loss_function = AlignedTripletLoss(tri_loss)
            
            self.loss.append({
                'type': loss_type,
                'weight': float(weight),
                'function': loss_function
                })
            
        # CSCE 625: Mutual Learning
        if args.mutual_learning:
            self.ml_pm_weight = 1.0
            self.ml_global_weight = 0.0
            self.ml_local_weight = 1.0
            # self.loss.append({
            #     'type': 'ProbabilityML',
            #     'weight': 1.0
            #     })
            # self.loss.append({
            #     'type': 'GlobalML',
            #     'weight': 0.0 # TODO: update after implementation
            #     })
            # self.loss.append({
            #     'type': 'LocalML',
            #     'weight': 0.0 # TODO: update after implementation
            #     })

        if len(self.loss) > 1:
            self.loss.append({'type': 'Total', 'weight': 0, 'function': None})

        for l in self.loss:
            if l['function'] is not None:
                print('{:.3f} * {}'.format(l['weight'], l['type']))
                self.loss_module.append(l['function'])

        self.log = torch.Tensor()

        self.device = torch.device('cpu' if args.cpu else 'cuda')
        self.loss_module.to(self.device)
        
        if args.load != '': self.load(ckpt.dir, cpu=args.cpu)
        if not args.cpu and args.nGPU > 1:
            self.loss_module = nn.DataParallel(
                self.loss_module, range(args.nGPU)
            )

    def forward(self, outputs_vec, labels):

        loss_sums = [0] * 2
        local_dist_mats = [[]] * 2

        # Compute loss_sum of each model loss
        for index in range(len(outputs_vec)):
            losses = []
            for i, l in enumerate(self.loss):
                # Triplet loss 
                if self.args.model == 'MGN' and l['type'] == 'Triplet':
                    loss = [l['function'](output, labels) for output in outputs_vec[index][1:4]]
                # Cross Entropy loss
                elif self.args.model == 'MGN' and l['type'] == 'CrossEntropy':
                    loss = [l['function'](output, labels) for output in outputs_vec[index][4:-3]]
                # CSCE 625: Aligned Parts Branch Loss
                elif self.args.model == 'MGN' and l['type'] == 'AlignedTriplet':
                    loss, dist_mat = l['function'](outputs_vec[index][-2], labels)
                    local_dist_mats[index] = dist_mat.to(self.device)
                    loss = [loss]
                else:
                    continue

                loss = sum(loss) / len(loss)
                effective_loss = l['weight'] * loss
                losses.append(effective_loss)
                if index == 0:
                    self.log[-1, i] += effective_loss.item()
        
            loss_sum = sum(losses)
            loss_sums[index] = loss_sum
            if index == 0 and len(self.loss) > 1:
                self.log[-1, -1] += loss_sum.item()

        # Not using mutual learning
        if len(outputs_vec) == 1:
            return loss_sums[0]

        # --- Using mutual learning (CSCE 625) ---

        # Compute mutual learning losses
        imgs = float(self.args.batchid)
        ml_pm_loss = [0] * 2
        ml_global_loss = [0] * 2
        ml_local_loss = [0] * 2
        
        # Probability mutual Loss
        probs = [0] * 2
        log_probs = [0] * 2
        probs[0] = F.softmax(outputs_vec[0][-1], dim=1)
        probs[1] = F.softmax(outputs_vec[1][-1], dim=1)
        log_probs[0] = F.log_softmax(outputs_vec[0][-1], dim=1)
        log_probs[1] = F.log_softmax(outputs_vec[1][-1], dim=1)
        
        ml_pm_loss[0] = F.kl_div(log_probs[0], probs[1], False) / imgs
        ml_pm_loss[1] = F.kl_div(log_probs[1], probs[0], False) / imgs
        
        # Local mutual loss
        ml_local_loss[0] = torch.sum(torch.pow(
            local_dist_mats[0] - local_dist_mats[1], 2)) \
            / (imgs * imgs)
        ml_local_loss[1] = torch.sum(torch.pow(
            local_dist_mats[1] - local_dist_mats[0], 2)) \
            / (imgs * imgs)

        for i in range(2):
            loss_sums[i] += \
                ml_pm_loss[i] * self.ml_pm_weight \
                + ml_global_loss[i] * self.ml_global_weight \
                + ml_local_loss[i] * self.ml_local_weight
        
        return loss_sums[0], loss_sums[1]

    def start_log(self):
        self.log = torch.cat((self.log, torch.zeros(1, len(self.loss))))

    def end_log(self, batches):
        self.log[-1].div_(batches)

    def display_loss(self, batch):
        n_samples = batch + 1
        log = []
        for l, c in zip(self.loss, self.log[-1]):
            log.append('[{}: {:.4f}]'.format(l['type'], c / n_samples))

        return ''.join(log)

    def plot_loss(self, apath, epoch):
        axis = np.linspace(1, epoch, epoch)
        for i, l in enumerate(self.loss):
            label = '{} Loss'.format(l['type'])
            fig = plt.figure()
            plt.title(label)
            plt.plot(axis, self.log[:, i].numpy(), label=label)
            plt.legend()
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.grid(True)
            plt.savefig('{}/loss_{}.jpg'.format(apath, l['type']))
            plt.close(fig)

    def step(self):
        for l in self.get_loss_module():
            if hasattr(l, 'scheduler'):
                l.scheduler.step()

    def get_loss_module(self):
        if self.nGPU == 1:
            return self.loss_module
        else:
            return self.loss_module.module

    def save(self, apath):
        torch.save(self.state_dict(), os.path.join(apath, 'loss.pt'))
        torch.save(self.log, os.path.join(apath, 'loss_log.pt'))

    def load(self, apath, cpu=False):
        if cpu:
            kwargs = {'map_location': lambda storage, loc: storage}
        else:
            kwargs = {}

        self.load_state_dict(torch.load(
            os.path.join(apath, 'loss.pt'),
            **kwargs
        ))
        self.log = torch.load(os.path.join(apath, 'loss_log.pt'))
        for l in self.loss_module:
            if hasattr(l, 'scheduler'):
                for _ in range(len(self.log)): l.scheduler.step()

