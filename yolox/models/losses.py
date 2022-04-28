#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import torch
import torch.nn as nn


class IOUloss(nn.Module):
    def __init__(self, reduction="none", loss_type="iou"):
        super(IOUloss, self).__init__()
        self.reduction = reduction
        self.loss_type = loss_type

    def forward(self, pred, target):
        assert pred.shape[0] == target.shape[0]

        pred = pred.view(-1, 4)
        target = target.view(-1, 4)
        tl = torch.max(
            (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
        )
        br = torch.min(
            (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
        )

        area_p = torch.prod(pred[:, 2:], 1)
        area_g = torch.prod(target[:, 2:], 1)

        en = (tl < br).type(tl.type()).prod(dim=1)
        area_i = torch.prod(br - tl, 1) * en
        area_u = area_p + area_g - area_i
        iou = (area_i) / (area_u + 1e-16)

        if self.loss_type == "iou":
            loss = 1 - iou ** 2
        elif self.loss_type == "giou":
            c_tl = torch.min(
                (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
            )
            c_br = torch.max(
                (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
            )
            area_c = torch.prod(c_br - c_tl, 1)
            giou = iou - (area_c - area_u) / area_c.clamp(1e-16)
            loss = 1 - giou.clamp(min=-1.0, max=1.0)

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss


def softmax_loss(score, label, num_classes, ignore_label=-1):
    with torch.no_grad():
        max_score = score.max(axis=1, keepdims=True)[0]
    score -= max_score
    log_prob = score - torch.log(torch.exp(score).sum(axis=1, keepdims=True))
    mask = label != ignore_label
    vlabel = label * mask
    onehot = torch.zeros(vlabel.shape[0], num_classes, device=score.device)
    onehot.scatter_(1, vlabel.reshape(-1, 1), 1)
    loss = -(log_prob * onehot).sum(axis=1)
    loss = loss * mask
    return loss


def smooth_l1_loss(pred, target, beta: float):
    if beta < 1e-5:
        loss = torch.abs(input - target)
    else:
        abs_x = torch.abs(pred- target)
        in_mask = abs_x < beta
        loss = torch.where(in_mask, 0.5 * abs_x ** 2 / beta, abs_x - 0.5 * beta)
    return loss.sum(axis=1)


def focal_loss(inputs, targets, alpha=-1, gamma=2, eps=1e-8):
    class_range = torch.arange(1, inputs.shape[1] + 1, device=inputs.device)
    pos_pred = (1 - inputs) ** gamma * torch.log(inputs + eps)
    neg_pred = inputs ** gamma * torch.log(1 - inputs)

    pos_loss = (targets == class_range) * pos_pred * alpha
    neg_loss = (targets != class_range) * neg_pred * (1 - alpha)
    loss = -(pos_loss + neg_loss)
    return loss.sum(axis=1)


def emd_loss_softmax(p_b0, p_s0, p_b1, p_s1, targets, labels, num_classes, rcnn_smooth_l1_beta=1):
    # reshape
    pred_delta = torch.cat([p_b0, p_b1], axis=1).reshape(-1, p_b0.shape[-1])
    pred_score = torch.cat([p_s0, p_s1], axis=1).reshape(-1, p_s0.shape[-1])
    targets = targets.reshape(-1, 4)
    labels = labels.long().flatten()
    # cons masks
    valid_masks = labels >= 0
    fg_masks = labels > 0
    # multiple class
    pred_delta = pred_delta.reshape(-1, num_classes, 4)
    fg_gt_classes = labels[fg_masks]
    pred_delta = pred_delta[fg_masks, fg_gt_classes, :]
    # loss for regression
    localization_loss = smooth_l1_loss(
        pred_delta,
        targets[fg_masks],
        rcnn_smooth_l1_beta)
    # loss for classification
    objectness_loss = softmax_loss(pred_score, labels, num_classes)
    loss = objectness_loss * valid_masks
    loss[fg_masks] = loss[fg_masks] + localization_loss
    loss = loss.reshape(-1, 2).sum(axis=1)
    return loss.reshape(-1, 1)

def emd_loss_focal(p_b0, p_s0, p_b1, p_s1, targets, labels, focal_loss_alpha=0.25, focal_loss_gamma=2, smooth_l1_beta=0.1):
    pred_delta = torch.cat([p_b0, p_b1], axis=1).reshape(-1, p_b0.shape[-1])
    pred_score = torch.cat([p_s0, p_s1], axis=1).reshape(-1, p_s0.shape[-1])
    print(pred_delta.shape)
    targets = targets.reshape(-1, 4)
    labels = labels.long().reshape(-1, 1)
    valid_mask = (labels >= 0).flatten()
    objectness_loss = focal_loss(pred_score, labels,
            focal_loss_alpha, focal_loss_gamma)
    fg_masks = (labels > 0).flatten()
    localization_loss = smooth_l1_loss(
            pred_delta[fg_masks],
            targets[fg_masks],
            smooth_l1_beta)
    loss = objectness_loss * valid_mask
    loss[fg_masks] = loss[fg_masks] + localization_loss
    loss = loss.reshape(-1, 2).sum(axis=1)
    return loss.reshape(-1, 1)

def emd_loss(p_b0, p_s0, p_b1, p_s1, targets, labels, focal_loss_alpha=0.25, focal_loss_gamma=2, smooth_l1_beta=0.1):
    targets0 = torch.zeros_like(targets, dtype=torch.float32)
    pred_delta = torch.cat([p_b0, p_b1], axis=1).reshape(-1, p_b0.shape[-1])
    targets = torch.cat([targets, targets0], axis=1)
    targets = targets.reshape(-1, 4)
    localization_loss = smooth_l1_loss(
            pred_delta,
            targets,
            smooth_l1_beta)
    loss = localization_loss
    loss = loss.reshape(-1, 2).sum(axis=1)
    return loss.reshape(-1, 1)