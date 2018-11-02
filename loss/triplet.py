#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable # TODO: remove after removeing TripletLoss2

import pdb # TODO: remove after debugging

class TripletSemihardLoss(nn.Module):
    """
    Shape:
        - Input: :math:`(N, C)` where `C = number of channels`
        - Target: :math:`(N)`
        - Output: scalar.
    """

    def __init__(self, device, margin=0, size_average=True):
        super(TripletSemihardLoss, self).__init__()
        self.margin = margin
        self.size_average = size_average
        self.device = device

    def forward(self, input, target):
        y_true = target.int().unsqueeze(-1)
        same_id = torch.eq(y_true, y_true.t()).type_as(input)

        pos_mask = same_id
        neg_mask = 1 - same_id

        def _mask_max(input_tensor, mask, axis=None, keepdims=False):
            input_tensor = input_tensor - 1e6 * (1 - mask)
            _max, _idx = torch.max(input_tensor, dim=axis, keepdim=keepdims)
            return _max, _idx

        def _mask_min(input_tensor, mask, axis=None, keepdims=False):
            input_tensor = input_tensor + 1e6 * (1 - mask)
            _min, _idx = torch.min(input_tensor, dim=axis, keepdim=keepdims)
            return _min, _idx

        # output[i, j] = || feature[i, :] - feature[j, :] ||_2
        dist_squared = torch.sum(input ** 2, dim=1, keepdim=True) + \
                       torch.sum(input.t() ** 2, dim=0, keepdim=True) - \
                       2.0 * torch.matmul(input, input.t())
        dist = dist_squared.clamp(min=1e-16).sqrt()

        pos_max, pos_idx = _mask_max(dist, pos_mask, axis=-1)
        neg_min, neg_idx = _mask_min(dist, neg_mask, axis=-1)

        # loss(x, y) = max(0, -y * (x1 - x2) + margin)
        y = torch.ones(same_id.size()[0]).to(self.device)
        return F.margin_ranking_loss(neg_min.float(),
                                     pos_max.float(),
                                     y,
                                     self.margin,
                                     self.size_average)

class TripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.

    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.

    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.

    Args:
        margin (float): margin for triplet.
    """
    def __init__(self, margin=0.3, mutual_flag = False):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.mutual = mutual_flag

    def forward(self, inputs, targets):
        """
        Args:
            inputs: feature matrix with shape (batch_size, feat_dim)
            targets: ground truth labels with shape (num_classes)
        """
        n = inputs.size(0)
        #inputs = 1. * inputs / (torch.norm(inputs, 2, dim=-1, keepdim=True).expand_as(inputs) + 1e-12)
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        if self.mutual:
            return loss, dist
        return loss


"""
------------ BELOW CODE FOR CSCE 625 ---------------

Code adapted from: https://github.com/huanghoujing/AlignedReID-Re-Production-Pytorch 
"""

class TripletLoss2(object):
  """Modified from Tong Xiao's open-reid (https://github.com/Cysu/open-reid). 
  Related Triplet Loss theory can be found in paper 'In Defense of the Triplet 
  Loss for Person Re-Identification'."""
  def __init__(self, margin=None):
    self.margin = margin
    if margin is not None:
      self.ranking_loss = nn.MarginRankingLoss(margin=margin)
    else:
      self.ranking_loss = nn.SoftMarginLoss()

  def __call__(self, dist_ap, dist_an):
    """
    Args:
      dist_ap: pytorch Variable, distance between anchor and positive sample, 
        shape [N]
      dist_an: pytorch Variable, distance between anchor and negative sample, 
        shape [N]
    Returns:
      loss: pytorch Variable, with shape [1]
    """
    y = Variable(dist_an.data.new().resize_as_(dist_an.data).fill_(1))
    if self.margin is not None:
      loss = self.ranking_loss(dist_an, dist_ap, y)
    else:
      loss = self.ranking_loss(dist_an - dist_ap, y)
    return loss

class AlignedTripletLoss(nn.Module):
    """
    *** CSCE 625 ***
    Local feature parts aligned loss 
        - Runs feature alignment over distance matrix
        - Uses triplet loss with hard positive/negative mining

    Reference:
    Zhang et al. AlignedReID: Surpassing Human-Level Performance in Person Re-Identification. arXiv:1711.08184.

    Args:
        margin (float): margin for triplet.

    Shapes:
        N: num_classes    
    """
    def __init__(self, tri_loss):
        super(AlignedTripletLoss, self).__init__()
        self.tri_loss = tri_loss

    def forward(self, inputs, labels, normalize_features=True):
        """
        Args:
            inputs: N local feature parts, shape (batch_size, feat_dim)
            labels: ground truth labels with shape (num_classes)
            normalize: whether to normalize the feature vectors
        Returns:
            loss: loss value, shape [1]
            dist_ap: distance(anchor, positive), shape [N]
            dist_an: distance(anchor, negative), shape [N]
        """

        if normalize_features:  
            inputs = self.normalize(inputs, axis=-1)
        
        # Create distance matrix 
        dist_mat = self.local_dist(inputs, inputs)
        dist_ap, dist_an = self.hard_example_mining(dist_mat, labels, return_inds=False)
        loss = self.tri_loss(dist_ap, dist_an)
        loss = loss.to(0)
        return loss #, dist_ap, dist_an, dist_mat

    def local_dist(self, x, y):
        """
        Args:
            x: pytorch Variable, with shape [M, m, d]
            y: pytorch Variable, with shape [N, n, d]
        Returns:
            dist: pytorch Variable, with shape [M, N]
        """
        M, m, d = x.size()
        N, n, d = y.size()
        x = x.contiguous().view(M * m, d)
        y = y.contiguous().view(N * n, d)
        # shape [M * m, N * n]
        dist_mat = self.euclidean_dist(x, y)
        dist_mat = (torch.exp(dist_mat) - 1.) / (torch.exp(dist_mat) + 1.)
        # shape [M * m, N * n] -> [M, m, N, n] -> [m, n, M, N]
        dist_mat = dist_mat.contiguous().view(M, m, N, n).permute(1, 3, 0, 2)
        # shape [M, N]
        dist_mat = self.shortest_dist(dist_mat)
        return dist_mat

    def shortest_dist(self, dist_mat):
        """Parallel version.
        Args:
            dist_mat: pytorch Variable, available shape:
            1) [m, n]
            2) [m, n, N], N is batch size
            3) [m, n, *], * can be arbitrary additional dimensions
        Returns:
            dist: three cases corresponding to `dist_mat`:
            1) scalar
            2) pytorch Variable, with shape [N]
            3) pytorch Variable, with shape [*]
        """
        m, n = dist_mat.size()[:2]
        # Just offering some reference for accessing intermediate distance.
        dist = [[0 for _ in range(n)] for _ in range(m)]
        for i in range(m):
            for j in range(n):
                if (i == 0) and (j == 0):
                    dist[i][j] = dist_mat[i, j]
                elif (i == 0) and (j > 0):
                    dist[i][j] = dist[i][j - 1] + dist_mat[i, j]
                elif (i > 0) and (j == 0):
                    dist[i][j] = dist[i - 1][j] + dist_mat[i, j]
                else:
                    dist[i][j] = torch.min(dist[i - 1][j], dist[i][j - 1]) + dist_mat[i, j]
        dist = dist[-1][-1]
        return dist
    
    def euclidean_dist(self, x, y):
        """
        Args:
            x: pytorch Variable, with shape [m, d]
            y: pytorch Variable, with shape [n, d]
        Returns:
            dist: pytorch Variable, with shape [m, n]
        """
        m, n = x.size(0), y.size(0)
        xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
        yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
        dist = xx + yy
        dist.addmm_(1, -2, x, y.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        return dist
    
    def normalize(self, x, axis=-1):
        """Normalizing to unit length along the specified dimension.
        Args:
            x: pytorch Variable
        Returns:
            x: pytorch Variable, same shape as input      
        """
        x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
        return x
    
    def hard_example_mining(self, dist_mat, labels, return_inds=False):
        """For each anchor, find the hardest positive and negative sample.
        Args:
            dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
            labels: pytorch LongTensor, with shape [N]
            return_inds: whether to return the indices. Save time if `False`(?)
        Returns:
            dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
            dist_an: pytorch Variable, distance(anchor, negative); shape [N]
            p_inds: pytorch LongTensor, with shape [N]; 
            indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
            n_inds: pytorch LongTensor, with shape [N];
            indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
        NOTE: Only consider the case in which all labels have same num of samples, 
            thus we can cope with all anchors in parallel.
        """

        assert len(dist_mat.size()) == 2
        assert dist_mat.size(0) == dist_mat.size(1)
        N = dist_mat.size(0)

        # shape [N, N]
        is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
        is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())

        # `dist_ap` means distance(anchor, positive)
        # both `dist_ap` and `relative_p_inds` with shape [N, 1]
        dist_ap, relative_p_inds = torch.max(
            dist_mat[is_pos].contiguous().view(N, -1), 1, keepdim=True)
        # `dist_an` means distance(anchor, negative)
        # both `dist_an` and `relative_n_inds` with shape [N, 1]
        dist_an, relative_n_inds = torch.min(
            dist_mat[is_neg].contiguous().view(N, -1), 1, keepdim=True)
        # shape [N]
        dist_ap = dist_ap.squeeze(1)
        dist_an = dist_an.squeeze(1)

        if return_inds:
            # shape [N, N]
            ind = (labels.new().resize_as_(labels)
                .copy_(torch.arange(0, N).long())
                .unsqueeze( 0).expand(N, N))
            # shape [N, 1]
            p_inds = torch.gather(
                ind[is_pos].contiguous().view(N, -1), 1, relative_p_inds.data)
            n_inds = torch.gather(
                ind[is_neg].contiguous().view(N, -1), 1, relative_n_inds.data)
            # shape [N]
            p_inds = p_inds.squeeze(1)
            n_inds = n_inds.squeeze(1)
            return dist_ap, dist_an, p_inds, n_inds

        return dist_ap, dist_an