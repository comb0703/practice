from __future__ import division
import os
import json
import numpy as np
from collections import defaultdict
import itertools
import six
import torch
from torchvision.ops.boxes import box_iou
import argparse

# converter =============
def box_convert(bbox_tensor):
    # xywh to left,top,right,bottom
    converted = bbox_tensor.clone()
    converted[:, 2] += bbox_tensor[:, 0]
    converted[:, 3] += bbox_tensor[:, 1]
    return converted

# calculate mAP =========
def eval_detection(
        pred_bboxes,
        pred_labels,
        pred_scores,
        gt_bboxes,
        gt_labels,
        gt_difficults=None,
        iou_thresh=0.5,
        use_07_metric=False):

    prec, rec = calc_detection_prec_rec(pred_bboxes,
                                            pred_labels,
                                            pred_scores,
                                            gt_bboxes,
                                            gt_labels,
                                            gt_difficults,
                                            iou_thresh=iou_thresh)

    ap = calc_detection_ap(prec, rec, use_07_metric=use_07_metric)
    ap = np.nan_to_num(ap)
    ap = ap[:30]

    return {'ap': ap, 'map': np.mean(ap)}

def calc_detection_prec_rec(
        pred_bboxes, pred_labels, pred_scores, gt_bboxes, gt_labels,
        gt_difficults=None,
        iou_thresh=0.5):

    pred_bboxes = iter(pred_bboxes)
    pred_labels = iter(pred_labels)
    pred_scores = iter(pred_scores)
    gt_bboxes = iter(gt_bboxes)
    gt_labels = iter(gt_labels)
    if gt_difficults is None:
        gt_difficults = itertools.repeat(None)
    else:
        gt_difficults = iter(gt_difficults)

    n_pos = defaultdict(int)
    score = defaultdict(list)
    match = defaultdict(list)

    for pred_bbox, pred_label, pred_score, gt_bbox, gt_label, gt_difficult in \
            six.moves.zip(
                pred_bboxes, pred_labels, pred_scores,
                gt_bboxes, gt_labels, gt_difficults):

        if gt_difficult is None:
            gt_difficult = np.zeros(gt_bbox.shape[0], dtype=bool)

        for l in np.unique(np.concatenate((pred_label, gt_label)).astype(int)):
            pred_mask_l = pred_label == l
            pred_bbox_l = pred_bbox[pred_mask_l]
            pred_score_l = pred_score[pred_mask_l]
            # sort by score
            order = pred_score_l.argsort()[::-1]
            pred_bbox_l = pred_bbox_l[order]
            pred_score_l = pred_score_l[order]

            gt_mask_l = gt_label == l
            gt_bbox_l = gt_bbox[gt_mask_l]
            gt_difficult_l = gt_difficult[gt_mask_l]

            n_pos[l] += np.logical_not(gt_difficult_l).sum()
            score[l].extend(pred_score_l)

            if len(pred_bbox_l) == 0:
                continue
            if len(gt_bbox_l) == 0:
                match[l].extend((0,) * pred_bbox_l.shape[0])
                continue

            # VOC evaluation follows integer typed bounding boxes.
            pred_bbox_l = pred_bbox_l.copy()
            pred_bbox_l[:, 2:] += 1
            gt_bbox_l = gt_bbox_l.copy()
            gt_bbox_l[:, 2:] += 1

            # pred_bbox_l = box_convert(torch.tensor(pred_bbox_l), in_fmt='xywh', out_fmt='xyxy')
            # gt_bbox_l = box_convert(torch.tensor(gt_bbox_l), in_fmt='xywh', out_fmt='xyxy')
            pred_bbox_l = box_convert(torch.tensor(pred_bbox_l))
            gt_bbox_l = box_convert(torch.tensor(gt_bbox_l))

            iou = box_iou(pred_bbox_l, gt_bbox_l).numpy()
            gt_index = iou.argmax(axis=1)
            # set -1 if there is no matching ground truth
            gt_index[iou.max(axis=1) < iou_thresh] = -1
            del iou

            selec = np.zeros(gt_bbox_l.shape[0], dtype=bool)
            for gt_idx in gt_index:
                if gt_idx >= 0:
                    if gt_difficult_l[gt_idx]:
                        match[l].append(-1)
                    else:
                        if not selec[gt_idx]:
                            match[l].append(1)
                        else:
                            match[l].append(0)
                    selec[gt_idx] = True
                else:
                    match[l].append(0)

    for iter_ in (
            pred_bboxes, pred_labels, pred_scores,
            gt_bboxes, gt_labels, gt_difficults):
        if next(iter_, None) is not None:
            raise ValueError('Length of input iterables need to be same.')

    n_fg_class = max(n_pos.keys()) + 1
    prec = [None] * n_fg_class
    rec = [None] * n_fg_class

    for l in n_pos.keys():
        score_l = np.array(score[l])
        match_l = np.array(match[l], dtype=np.int8)

        order = score_l.argsort()[::-1]
        match_l = match_l[order]

        tp = np.cumsum(match_l == 1)
        fp = np.cumsum(match_l == 0)

        # If an element of fp + tp is 0,
        # the corresponding element of prec[l] is nan.
        prec[l] = tp / (fp + tp)
        # If n_pos[l] is 0, rec[l] is None.
        if n_pos[l] > 0:
            rec[l] = tp / n_pos[l]

    return prec, rec

def calc_detection_ap(prec, rec, use_07_metric=False):

    n_fg_class = len(prec)
    ap = np.empty(n_fg_class)
    for l in six.moves.range(n_fg_class):
        if prec[l] is None or rec[l] is None:
            ap[l] = np.nan
            continue

        if use_07_metric:
            # 11 point metric
            ap[l] = 0
            for t in np.arange(0., 1.1, 0.1):
                if np.sum(rec[l] >= t) == 0:
                    p = 0
                else:
                    p = np.max(np.nan_to_num(prec[l])[rec[l] >= t])
                ap[l] += p / 11
        else:
            # correct AP calculation
            # first append sentinel values at the end
            mpre = np.concatenate(([0], np.nan_to_num(prec[l]), [0]))
            mrec = np.concatenate(([0], rec[l], [1]))

            mpre = np.maximum.accumulate(mpre[::-1])[::-1]

            # to calculate area under PR curve, look for points
            # where X axis (recall) changes value
            i = np.where(mrec[1:] != mrec[:-1])[0]

            # and sum (\Delta recall) * prec
            ap[l] = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    return ap

# prepare result ========
def read_result(file_path, DATASET_PATH):

    with open(DATASET_PATH, 'r', encoding="utf-8") as f:
        gt_dict = json.load(f)

    with open(file_path, 'r', encoding="utf-8") as f:
        pred_dict = json.load(f)
    
    # list to return
    pred_b = []
    pred_l = []
    pred_s = []

    gt_b = []
    gt_l = []

    for idx, file_name in enumerate(list(gt_dict.keys())):
        # gt
        cur_gt = np.array(gt_dict[file_name], dtype=np.float32)
        gt_l.append(cur_gt[:,0])
        gt_b.append(cur_gt[:,1:])
        
        # pred
        pred_ar = np.array(pred_dict[file_name], dtype=np.float32)
        
        if pred_ar.shape[0] > 100:
            pred_ar = pred_ar[:100, :, :]

        if pred_ar.shape[0] == 0: 
            pred_l.append(np.array([]))
            pred_b.append(np.array([]))
            pred_s.append(np.array([]))
        else:
            pred_l.append(pred_ar[:,0])
            pred_b.append(pred_ar[:,1:5])
            pred_s.append(pred_ar[:, 5]) 

    return pred_b, pred_l, pred_s, gt_b, gt_l

def evaluate(pred_b, pred_l, pred_s, gt_b, gt_l):
    mAP = eval_detection(pred_b, pred_l, pred_s, gt_b, gt_l)
    return mAP

def evaluation_metrics(pred_file_path, gt_file_path):
    pred_b, pred_l, pred_s, gt_b, gt_l = read_result(pred_file_path, gt_file_path)
    score_dict = evaluate(pred_b, pred_l, pred_s, gt_b, gt_l) # score
    return score_dict['map']

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--prediction', type=str, default='submitted_result.json')
    args.add_argument('--test_label_path',type=str,default='test_label')
    config = args.parse_args()

    print(evaluation_metrics(config.prediction, config.test_label_path))