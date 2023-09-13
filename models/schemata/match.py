import torch

from torch import nn
import torch.nn.functional as F


class Match(nn.Module):
    """
    Apply Attention Between Contextualized Scene Graph Representations and Schemata to match them.
    """

    def __init__(self,
                 in_edge_feats,
                 n_edge_classes,
                 in_node_feats,
                 n_node_classes,
                 sigmoid_uncertainty=False):
        """
        :param in_edge_feats: edge dim
        :param n_edge_classes: number of predicate classes
        :param in_node_feats: node dim
        :param n_node_classes: number of object classes
        """
        super(Match, self).__init__()
        self._in_edge_feats = in_edge_feats
        self.n_edge_classes = n_edge_classes
        self._in_node_feats = in_node_feats
        self.n_node_classes = n_node_classes
        self.sigmoid_uncertainty = sigmoid_uncertainty

        self.edges_schema = nn.Parameter(torch.Tensor(in_edge_feats, n_edge_classes))
        self.nodes_schema = nn.Parameter(torch.Tensor(in_node_feats, n_node_classes))

        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('leaky_relu', param=0.2)
        nn.init.xavier_normal_(self.edges_schema, gain=gain)
        nn.init.xavier_normal_(self.nodes_schema, gain=gain)

    def forward(self, node_emb, edge_emb, is_training,
                gt_node_dists, gt_edge_dists,
                node_destroy_index=None, edge_destroy_index=None, gt=False):
        raw_edge_class, h_edge_emb = \
            self.send_message_kg2sg(feat=edge_emb, schema=self.edges_schema, is_training=is_training,
                                    gt_dist=gt_edge_dists, gt_schema=gt, destroy_index=edge_destroy_index)
        raw_node_class, h_node_emb = \
            self.send_message_kg2sg(feat=node_emb, schema=self.nodes_schema, is_training=is_training,
                                    gt_dist=gt_node_dists, gt_schema=gt, destroy_index=node_destroy_index)
        return raw_edge_class, h_edge_emb, raw_node_class, h_node_emb

    @staticmethod
    def send_message_kg2sg(feat: torch.Tensor, schema: torch.Tensor, is_training: bool,
                           gt_dist, gt_schema=False, destroy_index=None):
        raw_att = feat @ schema
        # if gt_schema:  # IC
        #     if is_training:
        #         # Teacher Forcing; During training we pass the GT labels
        #         att = (torch.clone(gt_dist))
        #     else:
        #         att = F.softmax(raw_att, dim=1)
        #     mask_s = torch.zeros(att.shape[0], device=att.device)
        #     att = att * mask_s[:, None]
        # else:  # ICP
        if is_training:
            # Teacher Forcing; During training we pass the GT labels
            att = (torch.clone(gt_dist))
        else:
            att = F.softmax(raw_att, dim=1)
        schema_msg = att.detach() @ (torch.transpose(schema, 0, 1))
        return raw_att, schema_msg.detach()
