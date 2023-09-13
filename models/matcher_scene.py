# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self,
                 cost_token_class: float = 3,
                 cost_label_class: float = 3,
                 **kwargs):
        """Creates the matcher

        Params:
            cost_token_class: This is the relative weight of the classification error of token in the matching cost
            cost_label_class: This is the relative weight of the classification error of label in the matching cost
        """
        super().__init__()
        self.cost_token_class = cost_token_class
        self.cost_label_class = cost_label_class
        assert cost_token_class != 0 or cost_label_class != 0, "all costs cant be 0"
        self.focal_loss = kwargs['config'].TRAIN.FOCAL_LOSS_ALPHA


    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_token_logits": Tensor of dim [batch_size, num_queries, num_token_classes] with the classification logits
                 "pred_label_logits": Tensor of dim [batch_size, num_queries, num_label_classes] with the classification logits

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "tokens": Tensor of dim [num_targets] (where num_targets is the number of ground-truth
                           objects in the target)
                 "labels": Tensor of dim [num_targets] (where num_targets is the number of ground-truth
                           objects in the target)

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:11
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_tokens)
        """
        with torch.no_grad():
            bs, num_queries = outputs["pred_token_logits"].shape[:2]

            # Also concat the target labels and boxes
            tgt_token_ids = torch.cat([v["tokens"] for v in targets])
            tgt_label_ids = torch.cat([v["labels"] for v in targets])

            # We flatten to compute the cost matrices in a batch
            if self.focal_loss:
                out_token_prob = outputs["pred_token_logits"].flatten(0, 1).sigmoid()  # [batch_size, num_queries,
                # num_token_classes]
                # out_label_prob = outputs["pred_label_logits"].flatten(0, 1).sigmoid()
                # Compute the classification cost.
                alpha = 0.25
                gamma = 2.0
                neg_token_cost_class = (1 - alpha) * (out_token_prob ** gamma) * (-(1 - out_token_prob + 1e-8).log())
                pos_token_cost_class = alpha * ((1 - out_token_prob) ** gamma) * (-(out_token_prob + 1e-8).log())
                cost_token_class = pos_token_cost_class[:, tgt_token_ids] - neg_token_cost_class[:, tgt_token_ids]
                # neg_label_cost_class = (1 - alpha) * (out_label_prob ** gamma) * (-(1 - out_label_prob + 1e-8).log())
                # pos_label_cost_class = alpha * ((1 - out_label_prob) ** gamma) * (-(out_label_prob + 1e-8).log())
                # cost_label_class = pos_label_cost_class[:, tgt_label_ids] - neg_label_cost_class[:, tgt_label_ids]
            else:
                out_token_prob = outputs["pred_token_logits"].flatten(0, 1).softmax(-1)
                cost_token_class = -out_token_prob[:, tgt_token_ids]

            out_label_prob = outputs["pred_label_logits"].flatten(0, 1).softmax(-1)
            cost_label_class = -out_label_prob[:, tgt_label_ids]


            # Final cost matrix
            C = self.cost_token_class * cost_token_class + self.cost_label_class * cost_label_class
            C = C.view(bs, num_queries, -1).cpu()

            sizes = [len(v["tokens"]) for v in targets]
            indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
            return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def build_matcher(**kwargs):
    return HungarianMatcher(**kwargs)
