# ------------------------------------------------------------------------------
# Portions of this code are from
# CornerNet (https://github.com/princeton-vl/CornerNet)
# Copyright (c) 2018, University of Michigan
# Licensed under the BSD 3-Clause License
# ------------------------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from .utils import _tranpose_and_gather_feat
import torch.nn.functional as F
from fvcore.nn import sigmoid_focal_loss_jit
import math

def _only_neg_loss(pred, gt):
  gt = torch.pow(1 - gt, 4)
  neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * gt
  return neg_loss.sum()


class FastFocalLoss(nn.Module):
  '''
  Reimplemented focal loss, exactly the same as the CornerNet version.
  Faster and costs much less memory.
  '''
  def __init__(self, opt=None):
    super(FastFocalLoss, self).__init__()
    self.only_neg_loss = _only_neg_loss

  def forward(self, out, target, ind, mask, cat):
    '''
    Arguments:
      out, target: B x C x H x W
      ind, mask: B x M
      cat (category id for peaks): B x M
    '''
    neg_loss = self.only_neg_loss(out, target)
    pos_pred_pix = _tranpose_and_gather_feat(out, ind) # B x M x C
    pos_pred = pos_pred_pix.gather(2, cat.unsqueeze(2)) # B x M
    num_pos = mask.sum()
    pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, 2) * \
               mask.unsqueeze(2)
    pos_loss = pos_loss.sum()
    if num_pos == 0:
      return - neg_loss
    return - (pos_loss + neg_loss) / num_pos

class RegWeightedL1Loss(nn.Module):
  def __init__(self):
    super(RegWeightedL1Loss, self).__init__()

  def forward(self, output, mask, ind, target):
    pred = _tranpose_and_gather_feat(output, ind)
    loss = F.l1_loss(pred * mask, target * mask, reduction='sum')
    loss = loss / (mask.sum() + 1e-4)
    return loss


class WeightedBCELoss(nn.Module):
  def __init__(self):
    super(WeightedBCELoss, self).__init__()
    self.bceloss = torch.nn.BCEWithLogitsLoss(reduction='none')

  def forward(self, output, mask, ind, target):
    # output: B x F x H x W
    # ind: B x M
    # mask: B x M x F
    # target: B x M x F
    pred = _tranpose_and_gather_feat(output, ind) # B x M x F
    loss = mask * self.bceloss(pred, target)
    loss = loss.sum() / (mask.sum() + 1e-4)
    return loss

class BinRotLoss(nn.Module):
  def __init__(self):
    super(BinRotLoss, self).__init__()
  
  def forward(self, output, mask, ind, rotbin, rotres):
    pred = _tranpose_and_gather_feat(output, ind)
    loss = compute_rot_loss(pred, rotbin, rotres, mask)
    return loss

def compute_res_loss(output, target):
    return F.smooth_l1_loss(output, target, reduction='mean')

def compute_bin_loss(output, target, mask):
    mask = mask.expand_as(output)
    output = output * mask.float()
    return F.cross_entropy(output, target, reduction='mean')

def compute_rot_loss(output, target_bin, target_res, mask):
    # output: (B, 128, 8) [bin1_cls[0], bin1_cls[1], bin1_sin, bin1_cos, 
    #                 bin2_cls[0], bin2_cls[1], bin2_sin, bin2_cos]
    # target_bin: (B, 128, 2) [bin1_cls, bin2_cls]
    # target_res: (B, 128, 2) [bin1_res, bin2_res]
    # mask: (B, 128, 1)
    output = output.view(-1, 8)
    target_bin = target_bin.view(-1, 2)
    target_res = target_res.view(-1, 2)
    mask = mask.view(-1, 1)
    loss_bin1 = compute_bin_loss(output[:, 0:2], target_bin[:, 0], mask)
    loss_bin2 = compute_bin_loss(output[:, 4:6], target_bin[:, 1], mask)
    loss_res = torch.zeros_like(loss_bin1)
    if target_bin[:, 0].nonzero().shape[0] > 0:
        idx1 = target_bin[:, 0].nonzero()[:, 0]
        valid_output1 = torch.index_select(output, 0, idx1.long())
        valid_target_res1 = torch.index_select(target_res, 0, idx1.long())
        loss_sin1 = compute_res_loss(
          valid_output1[:, 2], torch.sin(valid_target_res1[:, 0]))
        loss_cos1 = compute_res_loss(
          valid_output1[:, 3], torch.cos(valid_target_res1[:, 0]))
        loss_res += loss_sin1 + loss_cos1
    if target_bin[:, 1].nonzero().shape[0] > 0:
        idx2 = target_bin[:, 1].nonzero()[:, 0]
        valid_output2 = torch.index_select(output, 0, idx2.long())
        valid_target_res2 = torch.index_select(target_res, 0, idx2.long())
        loss_sin2 = compute_res_loss(
          valid_output2[:, 6], torch.sin(valid_target_res2[:, 1]))
        loss_cos2 = compute_res_loss(
          valid_output2[:, 7], torch.cos(valid_target_res2[:, 1]))
        loss_res += loss_sin2 + loss_cos2
    return loss_bin1 + loss_bin2 + loss_res


class TripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.
    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.
    Args:
        margin (float): margin for triplet.
    """

    def __init__(self, margin=0.3, mutual_flag=False):
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
        # inputs = 1. * inputs / (torch.norm(inputs, 2, dim=-1, keepdim=True).expand_as(inputs) + 1e-12)
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(inputs, inputs.t(), beta=1, alpha=-2)
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


class EmbeddingLoss(nn.Module):
    def __init__(self, opt):
        super(EmbeddingLoss, self).__init__()
        self.nID = opt.nID
        self.emb_scale = math.sqrt(2) * math.log(self.nID - 1)
        self.emb_dim = opt.embedding_dim
        self.classifier = nn.Linear(self.emb_dim, self.nID)
        if opt.embedding_loss == 'focal':
            torch.nn.init.normal_(self.classifier.weight, std=0.01)
            prior_prob = 0.01
            bias_value = -math.log((1 - prior_prob) / prior_prob)
            torch.nn.init.constant_(self.classifier.bias, bias_value)
        self.opt = opt
        self.IDLoss = nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self, output, mask, ind, target):
        # Gather the embeddings for the specified indices and apply the mask
        id_head = _tranpose_and_gather_feat(output, ind)  # Assuming output is directly passed
        id_head = id_head[mask > 0].contiguous()
        id_head = self.emb_scale * F.normalize(id_head, dim=1)

        # Gather the target IDs for the masked positions
        id_target = target[mask > 0]

        # Compute the classification output using the classifier
        id_output = self.classifier(id_head).contiguous()
            
        # Compute the classification loss
        if self.opt.embedding_loss == 'focal':
            # Assuming `sigmoid_focal_loss_jit` function is defined elsewhere
            id_target_one_hot = id_output.new_zeros((id_head.size(0), self.nID)).scatter_(
                1, id_target.long().view(-1, 1), 1)
            classification_loss = sigmoid_focal_loss_jit(
                id_output, id_target_one_hot, alpha=0.25, gamma=2.0, reduction="sum"
            ) / id_output.size(0)
        else:
            classification_loss = self.IDLoss(id_output, id_target)
        return classification_loss

class EmbeddingVectorLoss(nn.Module):
    def __init__(self, opt):
        super(EmbeddingVectorLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.opt = opt
        self.emb_scale = math.sqrt(2) * math.log(self.opt.nID - 1)

    def forward(self, output, mask, ind, target):
        vector_head = _tranpose_and_gather_feat(output, ind)
        vector_head_masked = vector_head[mask > 0].contiguous()
        vector_head_normalized = self.emb_scale * F.normalize(vector_head_masked, dim=1)
        if vector_head_normalized.numel() > 0:
          vector_target = target[mask > 0].contiguous()
          vector_loss = self.mse_loss(vector_head_normalized, vector_target)
        else:
          vector_loss = torch.tensor(0.0).to(output.device)
        return vector_loss
      
class EmbeddingVectorCosineSimilarityLoss(nn.Module):
    def __init__(self, opt):
        super(EmbeddingVectorCosineSimilarityLoss, self).__init__()
        self.opt = opt
        self.emb_scale = math.sqrt(2) * math.log(self.opt.nID - 1)

    def forward(self, output, mask, ind, target):
        # Gather and transpose features as in the MSE loss
        vector_head = _tranpose_and_gather_feat(output, ind)
        vector_head_masked = vector_head[mask > 0].contiguous()
        vector_head_normalized = self.emb_scale * F.normalize(vector_head_masked, dim=1)
        
        if vector_head_normalized.numel() > 0:
            vector_target = target[mask > 0].contiguous()
            vector_target_normalized = F.normalize(vector_target, dim=1)
            cosine_similarity = F.cosine_similarity(vector_head_normalized, vector_target_normalized, dim=1)
            vector_loss = 1 - cosine_similarity.mean()
        else:
            vector_loss = torch.tensor(0.0).to(output.device)

        return vector_loss