import os
import numpy as np
from importlib import import_module

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from itertools import cycle

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
        self.circular_mixed_loss_queue = None
        mixed_subloss_options = {
            'TripletSemihard': {
                'function': TripletSemihardLoss(torch.device('cpu' if args.cpu else 'cuda'), args.margin),
                'subtype': 'TripletSemihard'
            },
            'Triplet': {
                'function': TripletLoss(args.margin),
                'subtype': 'Triplet'
            }
        }
        subtype = ''
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
            # ------------ BELOW CODE FOR CSCE 625 ---------------
            # Allow a mixed loss function for the training...
            # switch out loss functions after a fixed number of epochs
            # set in args
            elif loss_type.startswith('Mixed'):
                print("Will cycle loss functions {} every {} epochs."\
                    .format(repr(loss_type.split('-')[1:]), self.args.switch_loss_every))
                self.circular_mixed_loss_queue = cycle(
                    mixed_subloss_options[l] for l in loss_type.split('-')[1:])
                # also append other loss functions to be used in the mix
                l = next(self.circular_mixed_loss_queue)
                loss_function = l['function']
                subtype = l['subtype']
                loss_type = 'Mixed'
            # ----------------------------------------------------
            self.loss.append({
                'type': loss_type,
                'weight': float(weight),
                'function': loss_function,
                'subtype': subtype
                })
            subtype = ''
            
        # CSCE 625: Mutual Learning
        if args.mutual_learning:
            self.ml_pm_weight = 1.0
            self.ml_global_weight = 0.0
            self.ml_local_weight = 1.0
            self.loss.append({
                'type': 'ProbabilityML',
                'weight': 1.0,
                'function': None
                })
            self.loss.append({
                'type': 'GlobalML',
                'weight': 0.0, # update after implementation
                'function': None
                })
            self.loss.append({
                'type': 'LocalML',
                'weight': 1.0,
                'function': None
                })

        if len(self.loss) > 1:
            self.loss.append({'type': 'Total', 'weight': 0, 'function': None})

        for l in self.loss:
            if l['function'] is not None:
                print('{:.3f} * {}'.format(l['weight'], l['type']))
                self.loss_module.append(l['function'])

        self.device = torch.device('cpu' if args.cpu else 'cuda')
        self.loss_module.to(self.device)
        
        if args.load != '': self.load(ckpt.dir, cpu=args.cpu)
        if not args.cpu and args.nGPU > 1:
            self.loss_module = nn.DataParallel(
                self.loss_module, range(args.nGPU)
            )

        self.log = [torch.Tensor()]
        if args.mutual_learning:
            self.log.append(torch.Tensor())

    def swap_mixed_loss(self):
        is_mixed = lambda x: x['type'] == 'Mixed'
        old_mixed = next(filter(is_mixed, self.loss), None)
        if old_mixed: # if we have a mixed loss function...
            new_mixed = {**old_mixed, **next(self.circular_mixed_loss_queue)}
            print("Changing loss from type {} to type {}".format(
                old_mixed['subtype'], new_mixed['subtype']))
            # swap out the old mixed for the new
            self.loss = [l if not is_mixed(l) else new_mixed for l in self.loss]

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
                # CSCE 625: Mixed Loss Functions
                elif self.args.model == 'MGN' and l['type'] == 'Mixed':
                    loss = [l['function'](output, labels) for output in outputs_vec[index][1:4]]
                else:
                    continue

                loss = sum(loss) / len(loss)
                effective_loss = l['weight'] * loss
                losses.append(effective_loss)
                self.log[index][-1, i] += effective_loss.item()
        
            loss_sum = sum(losses)
            loss_sums[index] = loss_sum

        # Not using mutual learning
        if len(outputs_vec) == 1:
            if len(self.loss) > 1:
                self.log[0][-1, -1] += loss_sum.item()
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
            loss_sums[i] = loss_sums[i]\
                + ml_pm_loss[i] * self.ml_pm_weight \
                + ml_global_loss[i] * self.ml_global_weight \
                + ml_local_loss[i] * self.ml_local_weight

            self.log[i][-1, -4] = ml_pm_loss[i] * self.ml_pm_weight  
            self.log[i][-1, -3] = ml_global_loss[i] * self.ml_global_weight
            self.log[i][-1, -2] = ml_local_loss[i] * self.ml_local_weight

        # finish logging
        if len(self.loss) > 1:
            self.log[0][-1, -1] += loss_sums[0].item()
            self.log[1][-1, -1] += loss_sums[1].item()        

        return loss_sums[0], loss_sums[1]

    def start_log(self):
        self.log[0] = torch.cat((self.log[0], torch.zeros(1, len(self.loss))))

        if self.args.mutual_learning:
            self.log[1] = torch.cat((self.log[1], torch.zeros(1, len(self.loss))))

    def end_log(self, batches):
        self.log[0][-1].div_(batches)

        if self.args.mutual_learning:
            self.log[1][-1].div_(batches)

    def display_loss(self, batch):

        n_samples = batch + 1
        log = []

        if self.args.mutual_learning:
            log.append('loss1: ')

        # Log for first loss
        for l, c in zip(self.loss, self.log[0][-1]):
            subtype = '~' + l['subtype'] if l.get('subtype', '') else ''
            log.append('[{}: {:.4f}]'.format(l['type']+subtype, c / n_samples))

        if self.args.mutual_learning:
            log.append('  loss2: ')
            for l, c in zip(self.loss, self.log[1][-1]):
                log.append('[{}: {:.4f}]'.format(l['type'], c / n_samples))

        return ''.join(log)

    def plot_loss(self, apath, epoch):
        axis = np.linspace(1, epoch, epoch)
        for i, l in enumerate(self.loss):
            label = '{} Loss'.format(l['type'])
            fig = plt.figure()
            plt.title(label)
            plt.plot(axis, self.log[0][:, i].numpy(), label=label)
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
        torch.save(self.log[0], os.path.join(apath, 'loss_log.pt'))

        if self.args.mutual_learning:
            torch.save(self.state_dict(), os.path.join(apath, 'loss2.pt'))
            torch.save(self.log[1], os.path.join(apath, 'loss2_log.pt'))

    def load(self, apath, cpu=False):
        if cpu:
            kwargs = {'map_location': lambda storage, loc: storage}
        else:
            kwargs = {}

        self.load_state_dict(torch.load(
            os.path.join(apath, 'loss.pt'),
            **kwargs
        ))
        self.log[0] = torch.load(os.path.join(apath, 'loss_log.pt'))
        for l in self.loss_module:
            if hasattr(l, 'scheduler'):
                for _ in range(len(self.log[0])): l.scheduler.step()

