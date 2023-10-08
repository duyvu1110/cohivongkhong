import torch.nn.functional as F
import torch.nn as nn
import torch, math
from models.matcher import HungarianMatcher
from pdb import set_trace as stop
from collections import defaultdict

class SetCriterion(nn.Module):
    """ This class computes the loss for Set_RE.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class, subject position and object position)
    """
    def __init__(self, num_classes, na_coef, losses, matcher):
        """ Create the criterion.
        Parameters:
            num_classes: number of relation categories
            matcher: module able to compute a matching between targets and proposals
            loss_weight: dict containing as key the names of the losses and as values their relative weight.
            na_coef: list containg the relative classification weight applied to the NA category and positional classification weight applied to the [SEP]
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = HungarianMatcher(matcher)
        self.losses = losses
        rel_weight = torch.ones(self.num_classes)
        rel_weight[0] = na_coef
        self.register_buffer('rel_weight', rel_weight)
        # self.rel_emb = nn.Embedding()

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs, targets) # indices:
        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            '''
            # 增加限定条件:Relation不为空,才会解码entity
            即能够解码entity的loss的前提是relation预测有值
            if the len of relation is None, the self.empty_targets(targets) is True
            '''
            if loss == "entity" and self.empty_targets(targets):
                pass
            else:
                losses.update(self.get_loss(loss, outputs, targets, indices))
        losses = sum(losses[k] for k in losses.keys())
        return losses

    def getMaskMatrix(self, labels):
        n = len(labels)
        mask = [[0 for i in range(n)] for i in range(n)]
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if labels[i] == 1 and labels[j] == 1:
                    mask[i][j] = 1

        return torch.tensor(mask).to(self.args.device)

    def ContrastiveLoss(self, logits, labels, rel_logits=None):
        '''
        logits : (n * hidden_size) n为这个batch中有的五元组对数
        labels : (n * 1)
        '''
        if len(logits) < 2:
            return 0.
        logits = torch.nn.functional.normalize(logits, dim=1) 
        n = len(labels)
        if rel_logits == None:
            simi_matrix = torch.mm(logits, logits.T).to(self.args.device) # (n * n) 
        else:
            rel_logits = torch.nn.functional.normalize(rel_logits, dim=1)
            simi_matrix = torch.mm(logits, rel_logits.T).to(self.args.device)
        zeros = [1 if i == 0 else 0 for i in labels] 
        ones = [1 if i == 1 else 0 for i in labels]
        twos = [1 if i == 2 else 0 for i in labels]
        threes = [1 if i == 3 else 0 for i in labels]
        # fours = [1 if i == 4 else 0 for i in labels]

        mask_labels = []
        if sum(zeros) > 1:
            mask_labels.append(zeros)
        if sum(ones) > 1:
            mask_labels.append(ones)
        if sum(twos) > 1:
            mask_labels.append(twos)
        if sum(threes) > 1:
            mask_labels.append(threes)
        # if sum(fours) > 1:
        #     mask_labels.append(threes)

        losses = 0.
        loss_nums = 0
        for mask_label in mask_labels:
            mask = self.getMaskMatrix(mask_label)
            # print(mask)
            res_matrix = simi_matrix * mask
            denominator = 0.
            numerator = []
            for i in range(n):
                for j in range(n):
                    if res_matrix[i][j] != 0:
                        numerator.append(torch.exp(res_matrix[i][j])) 
                    if i != j:
                        denominator += torch.exp(simi_matrix[i][j])
                for num in numerator:
                    losses += torch.log(num / denominator)
                    loss_nums += 1
                numerator = []
                denominator = 0. 
                
        return -1 * losses if loss_nums ==0 else -1 * losses / loss_nums # 取loss的平均值
        # return -1 * losses  # 不取平均值

    def quintuples_loss(self, outputs, targets, indices): # 对应公式（9）
        '''
        Args:
        outputs: dict
        targets: list
        indices: [(outputs_idx, targets_idx), ...(outputs_idx, targets_idx)]
        '''
        v_logits = outputs['v_logits'] # bsz, q_num, seq_len
        logits = []
        for i in range(len(indices)):
            for index in indices[i][0]:
                logits.append(v_logits[i][index].cpu().detach().numpy())
        labels = []
        for target in targets:
            labels += list(target['relation'].cpu().detach().numpy())

        logits = torch.Tensor(logits)
        labels = torch.Tensor(labels)
        loss = self.ContrastiveLoss(logits, labels)
        losses = {'quintuples_loss': loss}

        return losses
 
    def quintuple_relation_loss(self, outputs, targets, indices): # 对应公式（10）
        '''
        Args:
        outputs: dict
        targets: list
        indices: [(outputs_idx, targets_idx), ...(outputs_idx, targets_idx)]
        '''
        # Step 1: mapping
        v_logits = outputs['v_logits'] # bsz, q_num, seq_len
        idx = self._get_src_permutation_idx(indices)  # # obtain outputs 与target对应的id， batch_idx, src_idx
        
        mapping = defaultdict(lambda: self.num_classes)
        
        for out_idx, tgt_idx in sorted(indices, key=lambda x: x[0]):
            mapping[out_idx] = tgt_idx

        real_tgts = []
        bsz = len(targets) # obtain bsz
        for i in range(bsz):
            real_tgts[i] = mapping[i]
    
        # Step2: construct loss
        rel_reps = nn.Embedding(real_tgts) # 获取相应type的embedding
        cosine_similarity = nn.CosineSimilarity(dim=0)

        batch_pos = []
        batch_tot = []
        for rel in range(self.num_classes): # 遍历5个类别
            pos = []
            tot = []
            for ri, _r in enumerate(real_tgts): # 遍历相同类别下的正样本数
                if rel == _r:
                    pos.append(cosine_similarity(v_logits[ri], rel_reps[ri]))
                tot.append(cosine_similarity(v_logits[ri], rel_reps[ri]))
            
            if len(pos) > 0: # consider have at least one positive sample, that is, quintuple have 
                batch_pos.append(pos)
                batch_tot.append(tot)
        losses = []
        for pi, pos in enumerate(batch_pos):
            for p in pos:
                cur_loss = torch.log_softmax(p / torch.sum(batch_tot[pi]))
                losses.append(cur_loss)
            
        if len(losses) == 0: # that is, there is not positive sample in one batch
            return 0.0

        losses = {'quintuple_relation': -torch.sum(losses)}

        return losses
    

    # （A,B,C）三个N元组， 正例对 A内 or A与batch内其他样本找正例
    def relation_loss(self, outputs, targets, indices):
        """Classification loss (NLL)
        targets dicts must contain the key "relation" containing a tensor of dim [bsz]
        indices: list, len(indices)=bsz
        """
        src_logits = outputs['pred_rel_logits'] # [bsz, num_generated_triples, num_rel]
        idx = self._get_src_permutation_idx(indices) # idx: indices中有效的匹配结果, return: batch_idx, src_idx
        target_classes_o = torch.cat([t["relation"][i] for t, (_, i) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], 0,
                                    dtype=torch.int64, device=src_logits.device)
        # bsz, num_generated_triples
        target_classes[idx] = target_classes_o
        loss = F.cross_entropy(src_logits.flatten(0, 1), target_classes.flatten(0, 1), weight=self.rel_weight)
        # stop()
        losses = {'relation': loss}
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty triples
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_rel_logits = outputs['pred_rel_logits']
        device = pred_rel_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_rel_logits.argmax(-1) != pred_rel_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        '''
        batch_idx: 表示一个batch内的第几个样本
        src_idx:表示每一个batch内的,第i个样本的匹配结果
        '''
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        # torch.full_like(input, fill_value)，就是将input的形状作为返回结果tensor的形状, 返回结果的值全是fill_value
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices,  **kwargs):
        loss_map = {
            'relation': self.relation_loss,
            'cardinality': self.loss_cardinality,
            'entity': self.entity_loss,
            'quintuple_relation': self.quintuple_relation_loss,
            'quintuples_loss': self.quintuples_loss,
        }
        return loss_map[loss](outputs, targets, indices, **kwargs)

    def entity_loss(self, outputs, targets, indices):
        """Compute the losses related to the position of head entity or tail entity
        """
        idx = self._get_src_permutation_idx(indices)
        selected_pred_sub_start = outputs["sub_start_logits"][idx]
        selected_pred_sub_end = outputs["sub_end_logits"][idx]
        selected_pred_obj_start = outputs["obj_start_logits"][idx]
        selected_pred_obj_end = outputs["obj_end_logits"][idx]
        selected_pred_aspect_start = outputs["aspect_start_logits"][idx]
        selected_pred_aspect_end = outputs["aspect_end_logits"][idx]
        selected_pred_opinion_start = outputs["opinion_start_logits"][idx]
        selected_pred_opinion_end = outputs["opinion_end_logits"][idx]

        target_sub_start = torch.cat([t["sub_start_index"][i] for t, (_, i) in zip(targets, indices)])
        target_sub_end = torch.cat([t["sub_end_index"][i] for t, (_, i) in zip(targets, indices)])
        target_obj_start = torch.cat([t["obj_start_index"][i] for t, (_, i) in zip(targets, indices)])
        target_obj_end = torch.cat([t["obj_end_index"][i] for t, (_, i) in zip(targets, indices)])
        target_aspect_start = torch.cat([t["aspect_start_index"][i] for t, (_, i) in zip(targets, indices)])
        target_aspect_end = torch.cat([t["aspect_end_index"][i] for t, (_, i) in zip(targets, indices)])
        target_opinion_start = torch.cat([t["opinion_start_index"][i] for t, (_, i) in zip(targets, indices)])
        target_opinion_end = torch.cat([t["opinion_end_index"][i] for t, (_, i) in zip(targets, indices)])


        sub_start_loss = F.cross_entropy(selected_pred_sub_start, target_sub_start)
        sub_end_loss = F.cross_entropy(selected_pred_sub_end, target_sub_end)
        obj_start_loss = F.cross_entropy(selected_pred_obj_start, target_obj_start)
        obj_end_loss = F.cross_entropy(selected_pred_obj_end, target_obj_end)
        aspect_start_loss = F.cross_entropy(selected_pred_aspect_start, target_aspect_start)
        aspect_end_loss = F.cross_entropy(selected_pred_aspect_end, target_aspect_end)
        opinion_start_loss = F.cross_entropy(selected_pred_opinion_start, target_opinion_start)
        opinion_end_loss = F.cross_entropy(selected_pred_opinion_end, target_opinion_end)
        losses = {
            'sub': 1/2*(sub_start_loss + sub_end_loss), 
            'obj': 1/2*(obj_start_loss + obj_end_loss), 
            'aspect': 1/2*(aspect_start_loss + aspect_end_loss), 
            'opinion': 1/2*(opinion_start_loss + opinion_end_loss), 
        }
        # print(losses)
        return losses

    @staticmethod
    def empty_targets(targets):
        flag = True
        for target in targets:
            if len(target["relation"]) != 0:
                flag = False
                break
        return flag
