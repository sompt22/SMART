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
from .utils import _tranpose_and_gather_feat, _nms, _topk
import torch.nn.functional as F
from utils.image import draw_umich_gaussian
from fvcore.nn import sigmoid_focal_loss_jit
import math

def _slow_neg_loss(pred, gt):
  '''focal loss from CornerNet'''
  pos_inds = gt.eq(1).float()
  neg_inds = gt.lt(1).float()

  neg_weights = torch.pow(1 - gt[neg_inds], 4)

  loss = 0
  pos_pred = pred[pos_inds]
  neg_pred = pred[neg_inds]

  pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, 2)
  neg_loss = torch.log(1 - neg_pred) * torch.pow(neg_pred, 2) * neg_weights

  num_pos  = pos_inds.float().sum()
  pos_loss = pos_loss.sum()
  neg_loss = neg_loss.sum()

  if pos_pred.nelement() == 0:
    loss = loss - neg_loss
  else:
    loss = loss - (pos_loss + neg_loss) / num_pos
  return loss

def _neg_loss(pred, gt):
  ''' Reimplemented focal loss. Exactly the same as CornerNet.
      Runs faster and costs a little bit more memory
    Arguments:
      pred (batch x c x h x w)
      gt_regr (batch x c x h x w)
  '''
  pos_inds = gt.eq(1).float()
  neg_inds = gt.lt(1).float()

  neg_weights = torch.pow(1 - gt, 4)

  loss = 0
  pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
  neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

  num_pos  = pos_inds.float().sum()
  pos_loss = pos_loss.sum()
  neg_loss = neg_loss.sum()
  if num_pos == 0:
    loss = loss - neg_loss
  else:
    loss = loss - (pos_loss + neg_loss) / num_pos
  return loss

def _not_faster_neg_loss(pred, gt):
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()    
    num_pos  = pos_inds.float().sum()
    neg_weights = torch.pow(1 - gt, 4)

    loss = 0
    trans_pred = pred * neg_inds + (1 - pred) * pos_inds
    weight = neg_weights * neg_inds + pos_inds
    all_loss = torch.log(1 - trans_pred) * torch.pow(trans_pred, 2) * weight
    all_loss = all_loss.sum()

    if num_pos > 0:
        all_loss /= num_pos
    loss -=  all_loss
    return loss

def _slow_reg_loss(regr, gt_regr, mask):
    num  = mask.float().sum()
    mask = mask.unsqueeze(2).expand_as(gt_regr)

    regr    = regr[mask]
    gt_regr = gt_regr[mask]
    
    regr_loss = nn.functional.smooth_l1_loss(regr, gt_regr, size_average=False)
    regr_loss = regr_loss / (num + 1e-4)
    return regr_loss


def _only_neg_loss(pred, gt):
  gt = torch.pow(1 - gt, 4)
  neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * gt
  return neg_loss.sum()

class FocalLoss(nn.Module):
  '''nn.Module warpper for focal loss'''
  def __init__(self):
    super(FocalLoss, self).__init__()
    self.neg_loss = _neg_loss

  def forward(self, out, target):
    return self.neg_loss(out, target)


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

def _reg_loss(regr, gt_regr, mask):
  ''' L1 regression loss
    Arguments:
      regr (batch x max_objects x dim)
      gt_regr (batch x max_objects x dim)
      mask (batch x max_objects)
  '''
  num = mask.float().sum()
  mask = mask.unsqueeze(2).expand_as(gt_regr).float()

  regr = regr * mask
  gt_regr = gt_regr * mask
    
  regr_loss = nn.functional.smooth_l1_loss(regr, gt_regr, reduction='sum')
  regr_loss = regr_loss / (num + 1e-4)
  return regr_loss

class RegLoss(nn.Module):
  '''Regression loss for an output tensor
    Arguments:
      output (batch x dim x h x w)
      mask (batch x max_objects)
      ind (batch x max_objects)
      target (batch x max_objects x dim)
  '''
  def __init__(self):
    super(RegLoss, self).__init__()
  
  def forward(self, output, mask, ind, target):
    pred = _tranpose_and_gather_feat(output, ind)
    loss = _reg_loss(pred, target, mask)
    return loss

class RegL1Loss(nn.Module):
  def __init__(self):
    super(RegL1Loss, self).__init__()
  
  def forward(self, output, mask, ind, target):
    pred = _tranpose_and_gather_feat(output, ind)
    mask = mask.unsqueeze(2).expand_as(pred).float()
    # loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
    loss = F.l1_loss(pred * mask, target * mask, size_average=False)
    loss = loss / (mask.sum() + 1e-4)
    return loss

class NormRegL1Loss(nn.Module):
  def __init__(self):
    super(NormRegL1Loss, self).__init__()
  
  def forward(self, output, mask, ind, target):
    pred = _tranpose_and_gather_feat(output, ind)
    mask = mask.unsqueeze(2).expand_as(pred).float()
    # loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
    pred = pred / (target + 1e-4)
    target = target * 0 + 1
    loss = F.l1_loss(pred * mask, target * mask, size_average=False)
    loss = loss / (mask.sum() + 1e-4)
    return loss

class RegWeightedL1Loss(nn.Module):
  def __init__(self):
    super(RegWeightedL1Loss, self).__init__()
  
  def forward(self, output, mask, ind, target):
    pred = _tranpose_and_gather_feat(output, ind)
    # loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
    loss = F.l1_loss(pred * mask, target * mask, reduction='sum')
    loss = loss / (mask.sum() + 1e-4)
    return loss

class L1Loss(nn.Module):
  def __init__(self):
    super(L1Loss, self).__init__()
  
  def forward(self, output, mask, ind, target):
    pred = _tranpose_and_gather_feat(output, ind)
    mask = mask.unsqueeze(2).expand_as(pred).float()
    loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
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
    return F.smooth_l1_loss(output, target, reduction='elementwise_mean')

def compute_bin_loss(output, target, mask):
    mask = mask.expand_as(output)
    output = output * mask.float()
    return F.cross_entropy(output, target, reduction='elementwise_mean')

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
          # Reshape the target to match the embeddings dimension if necessary
          # It's assumed target is equivalent to 'batch['vectors']' and already
          # filtered by 'batch['vectors_mask'] > 0'
          vector_target = target[mask > 0].contiguous()      
          # Calculate MSE Loss     
          vector_loss = self.mse_loss(vector_head_masked, vector_target)
          if self.opt.debug == 4:
            print("vector head shape: \n", vector_head.shape)
            print("vector head masked shape: \n", vector_head_masked.shape)
            print("vector head normalized shape: \n", vector_head_normalized.shape)
            print("vector head normalized min max: \n", vector_head_normalized.min(), vector_head_normalized.max()) 
            print("vector target shape: \n", vector_target.shape)
            print("vector target min max: \n", vector_target.min(), vector_target.max())
        else:
          vector_loss = torch.tensor(0.0).to(vector_head_normalized.device) 
          if self.opt.debug == 4:
            print("vector head shape: \n", vector_head.shape)
            print("vector head masked shape: \n", vector_head_masked.shape)
            print("vector head normalized shape: \n", vector_head_normalized.shape)
            print("mask shape: \n", mask.shape)
            print("output shape: \n", output.shape)
            print("ind shape: \n", ind.shape)
            print("vector head empty! \n")       
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
            # Normalize the target vectors
            vector_target = target[mask > 0].contiguous()
            vector_target_normalized = F.normalize(vector_target, dim=1)
            
            # Calculate cosine similarity loss: 1 - cosine similarity
            cosine_similarity = F.cosine_similarity(vector_head_normalized, vector_target_normalized, dim=1)
            vector_loss = 1 - cosine_similarity.mean()
            
            if self.opt.debug == 4:
              print("vector head shape: \n", vector_head.shape)
              print("vector head masked shape: \n", vector_head_masked.shape)
              print("vector head normalized shape: \n", vector_head_normalized.shape)
              print("vector head normalized min max: \n", vector_head_normalized.min(), vector_head_normalized.max())
              print("vector target shape: \n", vector_target.shape)
              print("vector target normalized shape: \n", vector_target_normalized.shape)
              print("vector target min max: \n", vector_target_normalized.min(), vector_target_normalized.max())
        else:
            vector_loss = torch.tensor(0.0).to(vector_head_normalized.device)
            if self.opt.debug == 4:
              print("vector head shape: \n", vector_head.shape)
              print("vector head masked shape: \n", vector_head_masked.shape)
              print("vector head normalized shape: \n", vector_head_normalized.shape)
              print("mask shape: \n", mask.shape)
              print("output shape: \n", output.shape)
              print("ind shape: \n", ind.shape)
              print("vector head empty! \n")
        
        return vector_loss