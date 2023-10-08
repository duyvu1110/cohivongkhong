"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
from pdb import set_trace as stop


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, matcher):
        super().__init__()
        # self.cost_relation = loss_weight["relation"]
        # self.cost_head = loss_weight["head_entity"]
        # self.cost_tail = loss_weight["tail_entity"]
        self.matcher = matcher

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_rel_logits": Tensor of dim [batch_size, num_generated_triples, num_classes] with the classification logits
                 "{head, tail}_{start, end}_logits": Tensor of dim [batch_size, num_generated_triples, seq_len] with the predicted index logits
            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_generated_triples, num_gold_triples)
        """
        bsz, num_generated_triples = outputs["pred_rel_logits"].shape[:2] # outputs["pred_rel_logits"].shape: bsz, q_num, num_class
        # We flatten to compute the cost matrices in a batch
        # [bsz * num_generated_triples, num_classes]
        pred_rel = outputs["pred_rel_logits"].flatten(0, 1).softmax(-1) # pred_rel: type:(tensor),dim:(bsz*q_num, num_classes)
        gold_rel = torch.cat([v["relation"] for v in targets])
        # after masking the pad token
        pred_sub_start = outputs["sub_start_logits"].flatten(0, 1).softmax(-1)  # [bsz * num_generated_triples, seq_len]
        pred_sub_end = outputs["sub_end_logits"].flatten(0, 1).softmax(-1)
        pred_obj_start = outputs["obj_start_logits"].flatten(0, 1).softmax(-1)  # [bsz * num_generated_triples, seq_len]
        pred_obj_end = outputs["obj_end_logits"].flatten(0, 1).softmax(-1)
        pred_aspect_start = outputs["aspect_start_logits"].flatten(0, 1).softmax(-1)  # [bsz * num_generated_triples, seq_len]
        pred_aspect_end = outputs["aspect_end_logits"].flatten(0, 1).softmax(-1)
        pred_opinion_start = outputs["opinion_start_logits"].flatten(0, 1).softmax(-1)  # [bsz * num_generated_triples, seq_len]
        pred_opinion_end = outputs["opinion_end_logits"].flatten(0, 1).softmax(-1)

        gold_sub_start = torch.cat([v["sub_start_index"] for v in targets])
        gold_sub_end = torch.cat([v["sub_end_index"] for v in targets])
        gold_obj_start = torch.cat([v["obj_start_index"] for v in targets])
        gold_obj_end = torch.cat([v["obj_end_index"] for v in targets])
        gold_aspect_start = torch.cat([v["aspect_start_index"] for v in targets])
        gold_aspect_end = torch.cat([v["aspect_end_index"] for v in targets])
        gold_opinion_start = torch.cat([v["opinion_start_index"] for v in targets])
        gold_opinion_end = torch.cat([v["opinion_end_index"] for v in targets])
        if self.matcher == "avg":
            cost = - pred_rel[:, gold_rel] \
                - (pred_sub_start[:, gold_sub_start] + pred_sub_end[:, gold_sub_end]) \
                - (pred_obj_start[:, gold_obj_start] + pred_obj_end[:, gold_obj_end]) \
                - (pred_aspect_start[:, gold_aspect_start] + pred_aspect_end[:, gold_aspect_end]) \
                - (pred_opinion_start[:, gold_opinion_start] + pred_opinion_end[:, gold_opinion_end]) \
        # elif self.matcher == "min":
        #     cost = torch.cat([pred_head_start[:, gold_head_start].unsqueeze(1), pred_rel[:, gold_rel].unsqueeze(1), pred_head_end[:, gold_head_end].unsqueeze(
        #         1), pred_tail_start[:, gold_tail_start].unsqueeze(1), pred_tail_end[:, gold_tail_end].unsqueeze(1)], dim=1)
        #     cost = - torch.min(cost, dim=1)[0]
        else:
            raise ValueError("Wrong matcher")

        cost = cost.view(bsz, num_generated_triples, -1).cpu() # bsz, q_num, 3
        num_gold_triples = [len(v["relation"]) for v in targets] # num_gold_triples: 统计得到有效的triples, if len(r)=0，则表示该三元组无效
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(
            cost.split(num_gold_triples, -1))] # indices: 先遍历得到组合，再线性最优
        # stop()
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
