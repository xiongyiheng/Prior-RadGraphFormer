import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from datetime import datetime


def sigmoid_focal_loss(inputs, targets, num_targets, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.sum() / num_targets


@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class SetCriterion(nn.Module):
    """ This class computes the loss for Graphformer.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth objects and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction
    """

    def __init__(self, config, matcher, asm=None, project=None, **kwargs):
        """ Create the criterion.
        Parameters:
            num_token_classes: number of object token categories, omitting the special no-object category
            num_label_classes: number of object label categories
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.matcher = matcher
        self.freq = config.MODEL.DECODER.FREQ_BIAS
        if config.MODEL.DECODER.FREQ_BIAS:
            self.freq_baseline = kwargs['freq_baseline']
            self.use_target = kwargs['use_target']
        self.focal_alpha = None if kwargs['focal_alpha'] == '' else kwargs['focal_alpha']
        self.num_token_classes = config.MODEL.NUM_TOKEN_CLS + 1
        self.num_label_classes = config.MODEL.NUM_LABEL_CLS
        self.losses = config.TRAIN.LOSSES

        self.add_emd_rel = config.MODEL.DECODER.ADD_EMB_REL
        self.weight_dict = {'tokens': config.TRAIN.W_TOKEN,
                            'labels': config.TRAIN.W_LABEL,
                            'edges': config.TRAIN.W_EDGE,
                            }
        # TODO this is a hack
        if config.MODEL.DECODER.AUX_LOSS:
            aux_weight_dict = {}
            for i in range(config.MODEL.DECODER.DEC_LAYERS - 1):
                aux_weight_dict.update({k + f'_{i}': v for k, v in self.weight_dict.items()})
            aux_weight_dict.update({k + f'_enc': v for k, v in self.weight_dict.items()})
            self.weight_dict.update(aux_weight_dict)
        self.fg_edge = config.DATA.FG_EDGE_PER_IMG
        self.bg_edge = config.DATA.BG_EDGE_PER_IMG  # background edges

        if asm:
            self.asm = asm
            self.project = project

    def loss_token(self, outputs, targets, indices, num_targets=None):
        """Compute the losses related to token classfication
        """
        weight = torch.ones(self.num_token_classes).to(outputs.get_device())  # TODO; fix the class weight
        weight[0] = 0.1

        idx = self._get_src_permutation_idx(indices)

        target_token_classes_o = torch.cat([t[J] for t, (_, J) in zip(targets, indices)])
        target_token_classes = torch.zeros(outputs[..., 0].shape, dtype=torch.long).to(outputs.get_device())
        target_token_classes[idx] = target_token_classes_o
        if self.focal_alpha is not None:
            target_token_classes_onehot = torch.zeros([outputs.shape[0], outputs.shape[1], outputs.shape[2]],
                                                      dtype=outputs.dtype, layout=outputs.layout,
                                                      device=outputs.device)
            target_token_classes_onehot.scatter_(2, target_token_classes.unsqueeze(-1), 1)

            loss = sigmoid_focal_loss(outputs, target_token_classes_onehot, num_targets, alpha=self.focal_alpha,
                                      gamma=2) * \
                   outputs.shape[1]
        else:
            loss = F.cross_entropy(outputs.permute(0, 2, 1), target_token_classes, weight=weight, reduction='mean')

        # cls_acc = 100 - accuracy(outputs, targets_one_hot)[0]
        return loss

    def loss_label(self, outputs, targets, indices, num_targets=None):
        """Compute the losses related to label classfication
        """
        weight = torch.ones(self.num_label_classes).to(outputs.get_device())

        idx = self._get_src_permutation_idx(indices)

        target_label_classes_o = torch.cat([t[J] for t, (_, J) in zip(targets, indices)])
        target_label_classes = torch.zeros(outputs[..., 0].shape, dtype=torch.long).to(outputs.get_device())
        target_label_classes[idx] = target_label_classes_o
        loss = F.cross_entropy(outputs.permute(0, 2, 1), target_label_classes, weight=weight, reduction='mean')

        # cls_acc = 100 - accuracy(outputs, targets_one_hot)[0]
        return loss

    def loss_edges(self, object_token, relation_token, pred_token_classes, tgt_token_labels,
                   pred_label_classes, tgt_label_labels, target_edges, indices, cls_dist=None):
        """Compute the losses related to relation(edges)
        """

        # # last token is relation token
        # relation_token = h[...,-1,:]
        rel_labels = [t[:, 2] for t in target_edges]  # indicate what kind of relation it refers to
        target_edges = [t[:, :2] for t in target_edges]  # the nodes -> 2 elements
        # map the ground truth edge indices by the matcher ordering
        target_edges = [[t for t in tgt if t[0].cpu() in i and t[1].cpu() in i] for tgt, (_, i) in
                        zip(target_edges, indices)]
        target_edges = [
            torch.stack(t, 0) if len(t) > 0 else torch.zeros((0, 2), dtype=torch.long).to(object_token.device) for t in
            target_edges]

        filtered_edges = []  # predicted stuff
        for t, (_, i) in zip(target_edges, indices):
            if t.shape[0] > 0:
                tx = t.detach().clone()
                for idx, k in enumerate(i):
                    t[tx == k] = idx
            filtered_edges.append(t)
        all_edge_lbl = []
        all_node_lbl = []
        freq_token_dist = []
        freq_label_dist = []
        total_edge = 0
        total_fg = 0  # foreground
        relation_feature = []
        node_dists = {}
        rel_dists = {}

        # loop through each of batch to collect the edge and node
        for b_id, (filtered_edge, rel_label, n, t_token_lbl, p_token_lbl, t_label_lbl, p_label_lbl) in enumerate \
                    (zip(filtered_edges, rel_labels, tgt_token_labels, tgt_token_labels, pred_token_classes,
                         tgt_label_labels, pred_label_classes)):
            # find the -ve edges for training
            full_adj = torch.ones((n.shape[0], n.shape[0])) - torch.diag(
                torch.ones(n.shape[0]))  # connecting itself is always 0
            # make a n x n matrix indicating whether row element connects with column element
            full_adj[filtered_edge[:, 0], filtered_edge[:, 1]] = 0
            neg_edges = torch.nonzero(full_adj).to(
                filtered_edge.device)  # returns a 2-D tensor where each row is the index (x, y) for a nonzero value.

            # restrict unbalance in the +ve/-ve edge
            if filtered_edge.shape[0] > self.fg_edge:
                idx_ = torch.randperm(filtered_edge.shape[0])[
                       :self.fg_edge]  # returns a random permutation of integers from 0 to self.fg_edge - 1 ->
                # randomly choose
                filtered_edge = filtered_edge[idx_, :]
                rel_label = rel_label[idx_]
            # check whether the number of -ve edges are within limit
            if neg_edges.shape[0] >= self.bg_edge:  # self.bg_edge:  # random sample -ve edge
                idx_ = torch.randperm(neg_edges.shape[0])[:self.bg_edge]  # similar operation with above
                neg_edges = neg_edges[idx_, :]
            all_edges_ = torch.cat((filtered_edge, neg_edges), 0)
            total_edge += all_edges_.shape[0]
            total_fg += filtered_edge.shape[0]
            edge_labels = torch.cat(  # contains all edge labels for this batch
                (rel_label, torch.zeros(neg_edges.shape[0], dtype=torch.long).to(object_token.device)),
                0)  # 0 indicates invalid relation
            # now permute all the combination
            idx_ = torch.randperm(all_edges_.shape[0])
            all_edges_ = all_edges_[idx_, :]
            edge_labels = edge_labels[idx_]
            all_edge_lbl.append(edge_labels)  # append current batch's all edge labels to the global all edge labels

            # calculate gt_edge_dist based on edge_labels
            gt_edge_dists = F.one_hot(edge_labels, num_classes=4).float()
            # calculate gt_node_dists based on target token label
            all_node_lbl.append(t_token_lbl[indices[b_id][1]])
            gt_node_dists = F.one_hot(t_token_lbl[indices[b_id][1]], num_classes=230).float()

            # get the valid predicted matching
            pred_ids = indices[b_id][0]
            joint_emb = object_token[b_id, pred_ids, :]
            if self.asm:
                node_emb = joint_emb
                edge_emb = self.project(torch.cat((joint_emb[all_edges_[:, 0], :], joint_emb[all_edges_[:, 1], :],
                                                   relation_token[b_id, ...].repeat(all_edges_.shape[0], 1)), 1))
                head_ind = all_edges_[:, 0]
                tail_ind = all_edges_[:, 1]
                edge_class, _, node_class, _ = self.asm(init_node_emb=node_emb,
                                                        init_edge_emb=edge_emb,
                                                        head_ind=head_ind,
                                                        tail_ind=tail_ind,
                                                        is_training=True,
                                                        gt_node_dists=gt_node_dists,
                                                        gt_edge_dists=gt_edge_dists,
                                                        destroy_visual_input=False,
                                                        keep_inds=None
                                                        )
                for i in range(len(edge_class)):
                    if b_id == 0:  # only the first batch do initialization
                        node_dists['node_asm%d' % i] = []
                        rel_dists['rel_asm%d' % i] = []
                    node_dists['node_asm%d' % i].append(node_class[i])
                    rel_dists['rel_asm%d' % i].append(edge_class[i])
            else:
                relation_feature.append(torch.cat((joint_emb[all_edges_[:, 0], :], joint_emb[all_edges_[:, 1], :],
                                                   relation_token[b_id, ...].repeat(all_edges_.shape[0], 1)), 1))
        if self.asm:
            all_edge_lbl = torch.cat(all_edge_lbl, 0).to(object_token.get_device())  # transfer it into a tensor
            all_node_lbl = torch.cat(all_node_lbl, 0).to(object_token.get_device())
            losses = {}
            for i in range(len(node_dists)):
                node_pred = torch.cat(node_dists['node_asm%d' % i], 0)
                losses['node_loss%d' % i] = F.cross_entropy(node_pred, all_node_lbl, reduction='mean', ignore_index=0)
                relation_pred = torch.cat(rel_dists['rel_asm%d' % i], 0)
                losses['edge_loss%d' % i] = F.cross_entropy(relation_pred, all_edge_lbl, reduction='mean')
            loss = sum(losses.values())
        else:
            relation_feature = torch.cat(relation_feature, 0)  # transfer it into a tensor

            all_edge_lbl = torch.cat(all_edge_lbl, 0).to(object_token.get_device())  # transfer it into a tensor
            relation_pred = self.relation_embed(relation_feature)

            loss = F.cross_entropy(relation_pred, all_edge_lbl, reduction='mean')
        return loss

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def forward(self, h, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'enc_outputs'}
        valid_targets = []
        for t in targets:
            valid_target = {}
            token = []
            label = []
            edge = []
            for i in range(t["tokens"].shape[0]):
                if t["tokens"][i] != 0:
                    token.append(t["tokens"][i])
                    label.append(t["labels"][i])
            for i in range(t["edges"].shape[0]):
                if t["edges"][i, 2] != 0:
                    edge.append(np.asarray(t["edges"][i].detach().cpu()))
            valid_target["tokens"] = torch.LongTensor(token).to(next(iter(outputs.values())).device)
            valid_target["labels"] = torch.LongTensor(label).to(next(iter(outputs.values())).device)
            if len(edge) > 0:
                valid_target["edges"] = torch.LongTensor(np.asarray(edge)).to(next(iter(outputs.values())).device)
            else:
                valid_target["edges"] = torch.empty((0, 3), dtype=torch.long).to(next(iter(outputs.values())).device)
            valid_targets.append(valid_target)

        indices = self.matcher(outputs_without_aux, valid_targets)

        # Compute the average number of target tokens across all nodes, for normalization purposes
        num_targets = sum(len(t["tokens"]) for t in valid_targets)
        num_targets = torch.as_tensor([num_targets], dtype=torch.float, device=next(iter(outputs.values())).device)

        tgt_token_labels = [v["tokens"] for v in valid_targets]
        tgt_label_labels = [v["labels"] for v in valid_targets]
        tgt_edges = [v["edges"] for v in valid_targets]
        object_token, relation_token = h

        if 'aux_outputs' in outputs:
            final_obj_tkn, aux_obj_tkn = object_token[-1], object_token[:-1]
            final_rel_tkn, aux_rel_tkn = relation_token[-1], relation_token[:-1]
        else:
            final_obj_tkn = object_token
            final_rel_tkn = relation_token
        # calculate losses
        losses = {}
        losses = self.get_loss(final_obj_tkn, final_rel_tkn, outputs, tgt_token_labels, tgt_label_labels, tgt_edges,
                               indices,
                               num_targets, losses)

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, (obj_tkn, rel_tkn, aux_outputs) in enumerate(zip(aux_obj_tkn, aux_rel_tkn, outputs['aux_outputs'])):
                indices = self.matcher(aux_outputs, valid_targets)
                l_dict = {}
                l_dict = self.get_loss(obj_tkn, rel_tkn, aux_outputs, tgt_token_labels, tgt_label_labels, tgt_edges,
                                       indices,
                                       num_targets, l_dict)
                l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                losses.update(l_dict)
        # sum up whole loss
        losses['total'] = sum([losses[key] * self.weight_dict[key] for key in losses if key in self.weight_dict])

        return losses

    def get_loss(self, obj_tkn, rel_tkn, outputs, tgt_token_labels, tgt_label_labels, tgt_edges, indices, num_targets,
                 losses):
        '''
        calculate losses across all data
        '''
        losses['tokens'] = self.loss_token(outputs['pred_token_logits'], tgt_token_labels, indices, num_targets)
        losses['labels'] = self.loss_label(outputs['pred_label_logits'], tgt_label_labels, indices, num_targets)
        if obj_tkn is not None:  # for two stage, we are only interested in
            pred_token_labels = torch.argmax(outputs['pred_token_logits'], -1)
            pred_label_labels = torch.argmax(outputs['pred_label_logits'], -1)

            # if self.add_emd_rel: losses['edges'] = self.loss_edges(obj_tkn, rel_tkn, pred_token_labels,
            # tgt_token_labels, pred_label_labels, tgt_label_labels, indices, outputs['pred_token_logits'],
            # outputs['pred_label_logits']) else:
            losses['edges'] = self.loss_edges(obj_tkn, rel_tkn, pred_token_labels,
                                              tgt_token_labels,
                                              pred_label_labels, tgt_label_labels, tgt_edges, indices)

        return losses
