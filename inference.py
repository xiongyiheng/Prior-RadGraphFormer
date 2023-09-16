import torch

import numpy as np


def graph_infer(h, out, relation_embed=None, freq=None, asm=None, project=None, emb=False, thresh=0.5):
    # all token except the last one is object token
    object_token, relation_token = h
    if object_token.dim() == 4:
        object_token = object_token[-1].detach()  # when using aux loss -1
        relation_token = relation_token[-1].detach()
    else:
        object_token = object_token.detach()  # when using aux loss -1
        relation_token = relation_token.detach()

    # valid tokens and labels
    valid_tokens = torch.max(out['pred_token_logits'].softmax(-1).detach(), -1)
    valid_labels = torch.max(out['pred_label_logits'].softmax(-1).detach(), -1)

    pred_tokens = []
    pred_labels = []
    pred_rels = []
    pred_rel_class = []
    pred_rel_score = []
    all_node_pairs = []
    all_relation = []
    valid_nodes = []
    for batch_id in range(out['pred_token_logits'].shape[0]):

        # ID of the valid tokens
        node_id = torch.nonzero(valid_tokens[1][batch_id]).squeeze(1)  # id of token which are not 'none'
        valid_nodes.append(node_id.cpu().numpy())
        pred_token_classes = valid_tokens[1][batch_id, node_id]
        pred_label_classes = valid_labels[1][batch_id, node_id]

        pred_tokens.append(pred_token_classes.cpu().numpy())
        pred_labels.append(pred_label_classes.cpu().numpy())

        if node_id.dim() != 0 and node_id.nelement() != 0 and node_id.shape[0] > 1:

            # all possible node pairs in all token ordering
            node_pairs = torch.cat((torch.combinations(node_id), torch.combinations(node_id)[:, [1, 0]]), 0)
            id_rel = torch.tensor(list(range(len(node_id))))
            node_pairs_rel = torch.cat((torch.combinations(id_rel), torch.combinations(id_rel)[:, [1, 0]]), 0)

            joint_emb = object_token
            if asm:
                assert relation_embed is None
                # get the combined relation embedding
                node_emb = joint_emb[batch_id, node_id]
                edge_emb = project(
                    torch.cat((joint_emb[batch_id, node_pairs[:, 0], :], joint_emb[batch_id, node_pairs[:, 1], :],
                               relation_token[batch_id, ...].repeat(node_pairs.shape[0], 1)), 1))
                _, head_ind = torch.unique(node_pairs[:, 0], sorted=True, return_inverse=True)
                _, tail_ind = torch.unique(node_pairs[:, 1], sorted=True, return_inverse=True)
                edge_class, _, node_class, _ = asm(init_node_emb=node_emb,
                                                   init_edge_emb=edge_emb,
                                                   head_ind=head_ind,
                                                   tail_ind=tail_ind,
                                                   is_training=False,
                                                   gt_node_dists=None,
                                                   gt_edge_dists=None,
                                                   destroy_visual_input=False,
                                                   keep_inds=None
                                                   )
                relation_pred = edge_class[-1].detach()  # here just use the last asm's result
            else:
                assert asm is None
                joint_emb = object_token
                rln_feat = torch.cat(
                    (joint_emb[batch_id, node_pairs[:, 0], :], joint_emb[batch_id, node_pairs[:, 1], :],
                     relation_token[batch_id, ...].repeat(len(node_pairs), 1)), 1)
                relation_pred = relation_embed(rln_feat).detach()
            all_node_pairs.append(node_pairs_rel.cpu().numpy())
            all_relation.append(relation_pred.softmax(-1).detach().cpu().numpy())
            rel_id = torch.nonzero(torch.argmax(relation_pred, -1)).squeeze(1)
            if rel_id.dim() != 0 and rel_id.nelement() != 0 and rel_id.shape[0] > 1:
                rel_id = rel_id.cpu().numpy()
                pred_rels.append(node_pairs_rel[rel_id].cpu().numpy())
                pred_rel_class.append(
                    torch.argmax(relation_pred, -1)[rel_id].cpu().numpy())
                pred_rel_score.append(torch.softmax(relation_pred, -1)[rel_id].cpu().numpy())
            else:
                pred_rels.append(torch.empty(0, 2))
                pred_rel_class.append(torch.empty(0, 1))
                pred_rel_score.append(torch.empty(0, 1))
        else:
            all_node_pairs.append(None)
            all_relation.append(None)
            pred_rels.append(torch.empty(0, 2))
            pred_rel_class.append(torch.empty(0, 1))
            pred_rel_score.append(torch.empty(0, 1))

        out = {'node_id': valid_nodes, 'pred_tokens': pred_tokens, 'pred_labels': pred_labels, 'pred_rels': pred_rels,
               'pred_rels_class': pred_rel_class, 'pred_rels_score': pred_rel_score, 'all_node_pairs': all_node_pairs,
               'all_relation': all_relation}

        return out
