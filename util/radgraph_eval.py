import os.path
import sys

import numpy as np
import math
import pickle
from functools import reduce
import torch
import pdb
import torch.distributed as dist

EDGE_THRESHOLD = 0.0
np.set_printoptions(precision=3)
from abc import ABC


class BasicRadGraphEvaluator(ABC):
    def __init__(self, **kwargs):
        super(BasicRadGraphEvaluator, self).__init__()
        self.token_tp = 0.0
        self.token_fp = 0.0
        self.token_fn = 0.0

        self.token_tp_uc = 0.0
        self.token_fp_uc = 0.0
        self.token_fn_uc = 0.0

        self.relation_tp = 0.0
        self.relation_fp = 0.0
        self.relation_fn = 0.0

        self.relation_tp_uc = 0.0
        self.relation_fp_uc = 0.0
        self.relation_fn_uc = 0.0

        self.loss_token = []
        self.loss_label = []
        self.loss_edge = []
        self.loss_total = []

    def evaluate_radgraph_entry(self, gt_entry, pred_entry, losses):
        res = evaluate_from_dict(gt_entry, pred_entry, losses, self=self)
        return res

    def reset(self):
        self.token_tp = 0.0
        self.token_fp = 0.0
        self.token_fn = 0.0

        self.token_tp_uc = 0.0
        self.token_fp_uc = 0.0
        self.token_fn_uc = 0.0

        self.relation_tp = 0.0
        self.relation_fp = 0.0
        self.relation_fn = 0.0

        self.relation_tp_uc = 0.0
        self.relation_fp_uc = 0.0
        self.relation_fn_uc = 0.0

        self.loss_token = []
        self.loss_label = []
        self.loss_edge = []
        self.loss_total = []

    def print_stats(self, epoch_num=None, writer=None, return_output=False, file_path=None):
        writer.add_scalar("val_token_classification_loss", np.mean(np.array(self.loss_token)), epoch_num)
        writer.add_scalar("val_label_classification_loss", np.mean(np.array(self.loss_label)), epoch_num)
        writer.add_scalar("val_edge_loss", np.mean(np.array(self.loss_edge)), epoch_num)
        writer.add_scalar("val_total_loss", np.mean(np.array(self.loss_total)), epoch_num)

        print("##################EPOCH" + str(epoch_num) + "##################")

        print("token precision:")
        token_precision = self.token_tp / (self.token_fp + self.token_tp + 1e-6)
        print(str(round(token_precision, 2)))
        writer.add_scalar('evaluation metrics/token precision', token_precision, epoch_num)

        print("token recall:")
        token_recall = self.token_tp / (self.token_fn + self.token_tp + 1e-6)
        print(str(round(token_recall, 2)))
        writer.add_scalar('evaluation metrics/token recall', token_recall, epoch_num)

        print("token f1-score:")
        token_f1 = 2 * token_precision * token_recall / (token_precision + token_recall + 1e-6)
        print(str(round(token_f1, 2)))
        writer.add_scalar('evaluation metrics/token f1-score', token_f1, epoch_num)

        print("token precision with uncertainty:")
        token_precision_uc = self.token_tp_uc / (self.token_fp_uc + self.token_tp_uc + 1e-6)
        print(str(round(token_precision_uc, 2)))
        writer.add_scalar('evaluation metrics/token precision with uncertainty', token_precision_uc, epoch_num)

        print("token recall with uncertainty:")
        token_recall_uc = self.token_tp_uc / (self.token_fn_uc + self.token_tp_uc + 1e-6)
        print(str(round(token_recall_uc, 2)))
        writer.add_scalar('evaluation metrics/token recall with uncertainty', token_recall_uc, epoch_num)

        print("token f1-score with uncertainty:")
        token_f1_uc = 2 * token_precision_uc * token_recall_uc / (token_precision_uc + token_recall_uc + 1e-6)
        print(str(round(token_f1_uc, 2)))
        writer.add_scalar('evaluation metrics/token f1-score with uncertainty', token_f1_uc, epoch_num)

        print("relation precision:")
        relation_precision = self.relation_tp / (self.relation_fp + self.relation_tp + 1e-6)
        print(str(round(relation_precision, 2)))
        writer.add_scalar('evaluation metrics/relation precision', relation_precision, epoch_num)

        print("relation recall:")
        relation_recall = self.relation_tp / (self.relation_fn + self.relation_tp + 1e-6)
        print(str(round(relation_recall, 2)))
        writer.add_scalar('evaluation metrics/relation recall', relation_recall, epoch_num)

        print("relation f1-score:")
        relation_f1 = 2 * relation_precision * relation_recall / (relation_precision + relation_recall + 1e-6)
        print(str(round(relation_f1, 2)))
        writer.add_scalar('evaluation metrics/relation f1-score', relation_f1, epoch_num)

        print("relation precision with uncertainty:")
        relation_precision_uc = self.relation_tp_uc / (self.relation_fp_uc + self.relation_tp_uc + 1e-6)
        print(str(round(relation_precision_uc, 2)))
        writer.add_scalar('evaluation metrics/relation precision with uncertainty', relation_precision_uc, epoch_num)

        print("relation recall with uncertainty:")
        relation_recall_uc = self.relation_tp_uc / (self.relation_fn_uc + self.relation_tp_uc + 1e-6)
        print(str(round(relation_recall_uc, 2)))
        writer.add_scalar('evaluation metrics/relation recall with uncertainty', relation_recall_uc, epoch_num)

        print("relation f1-score with uncertainty:")
        relation_f1_uc = 2 * relation_precision_uc * relation_recall_uc / (relation_precision_uc + relation_recall_uc +
                                                                           1e-6)
        print(str(round(relation_f1_uc, 2)))
        writer.add_scalar('evaluation metrics/relation f1-score with uncertainty', relation_f1_uc, epoch_num)

        print("average f1 score:")
        print((str(round((token_f1 + relation_f1) / 2, 2))))
        writer.add_scalar('evaluation metrics/average f1 score', (token_f1 + relation_f1) / 2, epoch_num)

        print("average f1 score with uncertainty:")
        print((str(round((token_f1_uc + relation_f1_uc) / 2, 2))))
        writer.add_scalar('evaluation metrics/average f1 score with uncertainty', (token_f1_uc + relation_f1_uc) / 2,
                          epoch_num)


def evaluate_from_dict(gt_entry, pred_entry, losses, self=None, **kwargs):
    # eval loss
    self.loss_token.append(losses['tokens'].item())
    self.loss_label.append(losses['labels'].item())
    self.loss_edge.append(losses['edges'].item())
    self.loss_total.append(losses['total'].item())

    # eval tokens
    gt_tokens = gt_entry['tokens']
    gt_labels = gt_entry['labels']

    pred_tokens = pred_entry[0]['tokens']
    pred_labels = pred_entry[0]['labels']

    assigned_gt_index = []
    num_targets = 0
    for i in range(gt_tokens.shape[0]):
        if gt_tokens[i] != 0:
            num_targets += 1
        else:
            break

    for i in range(len(pred_tokens)):
        is_contain = False  # whether pre is in gt
        for j in range(gt_tokens.shape[0]):
            if gt_tokens[j] != 0:

                if pred_tokens[i] == gt_tokens[j]:
                    is_contain = True
                    if j not in assigned_gt_index:
                        assigned_gt_index.append(j)
                        self.token_tp += 1
                        # consider uncertainty
                        if pred_labels[i] == gt_labels[j]:
                            self.token_tp_uc += 1
                        else:
                            self.token_fp_uc += 1
                        break
                    else:
                        # pass  # multiple are fine?
                        self.token_fp += 1  # multiple match to one also regard as false?
                        self.token_fp_uc += 1
        if not is_contain:
            self.token_fp += 1
            self.token_fp_uc += 1

    self.token_fn += (num_targets - len(assigned_gt_index))
    self.token_fn_uc += (num_targets - len(assigned_gt_index))

    # extract relations
    gt_rels = gt_entry['edges']
    pred_rels = pred_entry[1]['pred_rels']  # index of relation in **pred_tokens** not in gt
    pred_edge = pred_entry[1]['pred_edge']  # class of relation
    edge_pair_above_threshold = []
    edge_class_above_threshold = []
    if pred_edge.shape[0] != 0:
        for i in range(pred_edge.shape[0]):
            if pred_entry[1]['pred_rel_score'][i, pred_edge[i]] >= EDGE_THRESHOLD:
                edge_pair_above_threshold.append(pred_rels[i])
                edge_class_above_threshold.append(pred_edge[i])

    # eval relations
    assigned_gt_index = []
    num_targets = 0
    for i in range(gt_rels.shape[0]):
        if gt_rels[i, 2] != 0:
            num_targets += 1
        else:
            break

    for i in range(len(edge_class_above_threshold)):
        is_contain = False
        for j in range(gt_rels.shape[0]):
            if gt_rels[j, 2] != 0:

                if pred_tokens[edge_pair_above_threshold[i][0]] == gt_tokens[gt_rels[j, 0]] \
                        and pred_tokens[edge_pair_above_threshold[i][1]] == gt_tokens[gt_rels[j, 1]] \
                        and edge_class_above_threshold[i] == gt_rels[j, 2]:
                    is_contain = True
                    if j not in assigned_gt_index:
                        assigned_gt_index.append(j)
                        self.relation_tp += 1
                        if pred_labels[edge_pair_above_threshold[i][0]] == gt_labels[gt_rels[j, 0]] \
                                and pred_labels[edge_pair_above_threshold[i][1]] == gt_labels[gt_rels[j, 1]]:
                            self.relation_tp_uc += 1
                        else:
                            self.relation_fp_uc += 1
                        break
                    else:
                        pass  # multiple relations are fine?
        if not is_contain:
            self.relation_fp += 1
            self.relation_fp_uc += 1
    self.relation_fn += (num_targets - len(assigned_gt_index))
    self.relation_fn_uc += (num_targets - len(assigned_gt_index))
    
    return {'token_tp': self.token_tp, 'token_fp': self.token_fp, 'token_fn': self.token_fn, 'relation_tp': self.relation_tp,
            'relation_fp': self.relation_fp, 'relation_fn': self.relation_fn}
