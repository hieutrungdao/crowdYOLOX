#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import numpy as np

import torch
import torchvision

__all__ = [
    "filter_box",
    "postprocess",
    "bboxes_iou",
    "matrix_iou",
    "adjust_box_anns",
    "xyxy2xywh",
    "xyxy2cxcywh",
]

def set_cpu_nms(dets, thresh):
    """Pure Python NMS baseline."""
    def _overlap(det_boxes, basement, others):
        eps = 1e-8
        x1_basement, y1_basement, x2_basement, y2_basement \
                = det_boxes[basement, 0], det_boxes[basement, 1], \
                  det_boxes[basement, 2], det_boxes[basement, 3]
        x1_others, y1_others, x2_others, y2_others \
                = det_boxes[others, 0], det_boxes[others, 1], \
                  det_boxes[others, 2], det_boxes[others, 3]
        areas_basement = (x2_basement - x1_basement) * (y2_basement - y1_basement)
        areas_others = (x2_others - x1_others) * (y2_others - y1_others)
        xx1 = np.maximum(x1_basement, x1_others)
        yy1 = np.maximum(y1_basement, y1_others)
        xx2 = np.minimum(x2_basement, x2_others)
        yy2 = np.minimum(y2_basement, y2_others)
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas_basement + areas_others - inter + eps)
        return ovr

    dets = dets.cpu()
    scores = dets[:, 4]
    order = np.argsort(-scores)
    dets = dets[order]

    numbers = dets[:, -1]
    keep = np.ones(len(dets)) == 1
    ruler = np.arange(len(dets))
    while ruler.size>0:
        basement = ruler[0]
        ruler=ruler[1:]
        num = numbers[basement]
        # calculate the body overlap
        overlap = _overlap(dets[:, :4], basement, ruler)
        indices = np.where(overlap > thresh)[0]
        loc = np.where(numbers[ruler][indices] == num)[0]
        # the mask won't change in the step
        mask = keep[ruler[indices][loc]]#.copy()
        keep[ruler[indices]] = False
        keep[ruler[indices][loc][mask]] = True
        ruler[~keep[ruler]] = -1
        ruler = ruler[ruler>0]
    keep = keep[np.argsort(order)]
    return keep


def cpu_nms(dets, base_thr):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1) * (y2 - y1)
    order = np.argsort(-scores)

    keep = []
    eps = 1e-8
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter + eps)

        inds = np.where(ovr <= base_thr)[0]
        indices = np.where(ovr > base_thr)[0]
        order = order[inds + 1]
    return np.array(keep)


def filter_box(output, scale_range):
    """
    output: (N, 5+class) shape
    """
    min_scale, max_scale = scale_range
    w = output[:, 2] - output[:, 0]
    h = output[:, 3] - output[:, 1]
    keep = (w * h > min_scale * min_scale) & (w * h < max_scale * max_scale)
    return output[keep]


def postprocess(prediction, num_classes, conf_thre=0.7, nms_thre=0.45, class_agnostic=False):

    box_corner = prediction.new(prediction.shape)

    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2

    box_corner[:, :, 4] = prediction[:, :, 4] - prediction[:, :, 6] / 2
    box_corner[:, :, 5] = prediction[:, :, 5] - prediction[:, :, 7] / 2
    box_corner[:, :, 6] = prediction[:, :, 4] + prediction[:, :, 6] / 2
    box_corner[:, :, 7] = prediction[:, :, 5] + prediction[:, :, 7] / 2

    prediction[:, :, :9] = box_corner[:, :, :9]

    output = [None for _ in range(len(prediction))]
    for i, image_pred in enumerate(prediction):

        # If none are remaining => process next image
        if not image_pred.size(0):
            continue

        # Get score and class with highest confidence
        class_conf, class_pred = torch.max(image_pred[:, 10: 10 + num_classes], 1, keepdim=True)
        conf_mask = (image_pred[:, 8] * class_conf.squeeze() >= conf_thre).squeeze()
        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        detections = torch.cat((image_pred[:, :4], image_pred[:, 8:9], class_conf, class_pred.float()), 1)
        detections = detections[conf_mask]

        # # Get score and class with highest confidence
        # class_conf, class_pred = torch.max(image_pred[:, 10+num_classes: 10 + 2*num_classes], 1, keepdim=True)
        # conf_mask = (image_pred[:, 9] * class_conf.squeeze() >= conf_thre).squeeze()
        # # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        # detections = torch.cat((image_pred[:, 4:8], image_pred[:, 9:10], class_conf, class_pred.float()), 1)
        # detections = detections[conf_mask]

        if not detections.size(0):
            continue
        
        pred_boxes = detections[:, :5]
        pred_boxes = torch.cat((detections[:, :4], detections[:, 4:5] * detections[:, 5:6]), axis=1)
        nms_out_index = set_cpu_nms(pred_boxes, nms_thre)

        # if class_agnostic:
        #     nms_out_index = torchvision.ops.nms(
        #         detections[:, :4],
        #         detections[:, 4] * detections[:, 5],
        #         nms_thre,
        #     )
        # else:
        #     nms_out_index = torchvision.ops.batched_nms(
        #         detections[:, :4],
        #         detections[:, 4] * detections[:, 5],
        #         detections[:, 6],
        #         nms_thre,
        #     )

        detections = detections[nms_out_index]

        if output[i] is None:
            output[i] = detections
        else:
            output[i] = torch.cat((output[i], detections))

    return output


def bboxes_iou(bboxes_a, bboxes_b, xyxy=True):
    if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
        raise IndexError

    if xyxy:
        tl = torch.max(bboxes_a[:, None, :2], bboxes_b[:, :2])
        br = torch.min(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
        area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
        area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)
    else:
        tl = torch.max(
            (bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
            (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2),
        )
        br = torch.min(
            (bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
            (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2),
        )

        area_a = torch.prod(bboxes_a[:, 2:], 1)
        area_b = torch.prod(bboxes_b[:, 2:], 1)
    en = (tl < br).type(tl.type()).prod(dim=2)
    area_i = torch.prod(br - tl, 2) * en  # * ((tl < br).all())
    return area_i / (area_a[:, None] + area_b - area_i)


def matrix_iou(a, b):
    """
    return iou of a and b, numpy version for data augenmentation
    """
    lt = np.maximum(a[:, np.newaxis, :2], b[:, :2])
    rb = np.minimum(a[:, np.newaxis, 2:], b[:, 2:])

    area_i = np.prod(rb - lt, axis=2) * (lt < rb).all(axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
    return area_i / (area_a[:, np.newaxis] + area_b - area_i + 1e-12)


def adjust_box_anns(bbox, scale_ratio, padw, padh, w_max, h_max):
    bbox[:, 0::2] = np.clip(bbox[:, 0::2] * scale_ratio + padw, 0, w_max)
    bbox[:, 1::2] = np.clip(bbox[:, 1::2] * scale_ratio + padh, 0, h_max)
    return bbox


def xyxy2xywh(bboxes):
    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
    bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
    return bboxes


def xyxy2cxcywh(bboxes):
    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
    bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
    bboxes[:, 0] = bboxes[:, 0] + bboxes[:, 2] * 0.5
    bboxes[:, 1] = bboxes[:, 1] + bboxes[:, 3] * 0.5
    return bboxes
