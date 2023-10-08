import torch.nn as nn
import torch
from models.set_decoder_GCN import SetDecoder_GCN
from models.set_criterion import SetCriterion
from utils.functions import generate_triple
import copy
from pdb import set_trace as stop
# from utils.infoNCE_loss import InfoNCE, InfoNCE_add_temperature


class SetPred4RE_GCN(nn.Module):

    def __init__(self, args, num_classes):
        super(SetPred4RE_GCN, self).__init__()
        self.args = args
        self.encoder = BERTEncoder_GAT(args)
        config = self.encoder.config
        self.num_classes = num_classes
        # self.linear = nn.Linear()
        self.decoder = SetDecoder_GCN(args, config, args.num_generated_triples, args.num_decoder_layers, num_classes, return_intermediate=False)
        self.criterion = SetCriterion(num_classes, na_coef=args.na_rel_coef, losses=["entity", "relation"], matcher=args.matcher)
        # '--matcher', type=str, default="avg", choices=['avg', 'min']

    def forward(self, input_ids, attention_mask, dependency_graph, targets=None):
        # last_hidden_state, pooler_output = self.encoder(input_ids, attention_mask) # hidden state, cls
        last_hidden_state, pooler_output = self.encoder(input_ids, attention_mask, dependency_graph)
        class_logits, sub_start_logits, sub_end_logits, obj_start_logits, obj_end_logits, aspect_start_logits, \
        aspect_end_logits, opinion_start_logits, opinion_end_logits = self.decoder(encoder_hidden_states=last_hidden_state, encoder_attention_mask=attention_mask)
        # head_start_logits, head_end_logits, tail_start_logits, tail_end_logits = span_logits.split(1, dim=-1)
        sub_start_logits = sub_start_logits.squeeze(-1).masked_fill((1 - attention_mask.unsqueeze(1)).bool(), -10000.0)
        sub_end_logits = sub_end_logits.squeeze(-1).masked_fill((1 - attention_mask.unsqueeze(1)).bool(), -10000.0)
        obj_start_logits = obj_start_logits.squeeze(-1).masked_fill((1 - attention_mask.unsqueeze(1)).bool(), -10000.0)
        obj_end_logits = obj_end_logits.squeeze(-1).masked_fill((1 - attention_mask.unsqueeze(1)).bool(), -10000.0)
        aspect_start_logits = aspect_start_logits.squeeze(-1).masked_fill((1 - attention_mask.unsqueeze(1)).bool(), -10000.0)
        aspect_end_logits = aspect_end_logits.squeeze(-1).masked_fill((1 - attention_mask.unsqueeze(1)).bool(), -10000.0)
        opinion_start_logits = opinion_start_logits.squeeze(-1).masked_fill((1 - attention_mask.unsqueeze(1)).bool(), -10000.0)
        opinion_end_logits = opinion_end_logits.squeeze(-1).masked_fill((1 - attention_mask.unsqueeze(1)).bool(), -10000.0)
        
        outputs = {
            'pred_rel_logits': class_logits,
            'sub_start_logits': sub_start_logits, 
            'sub_end_logits': sub_end_logits,
            'obj_start_logits': obj_start_logits, 
            'obj_end_logits': obj_end_logits,
            'aspect_start_logits': aspect_start_logits, 
            'aspect_end_logits': aspect_end_logits,
            'opinion_start_logits': opinion_start_logits, 
            'opinion_end_logits': opinion_end_logits,
        }
        
        if targets is not None:
            loss = self.criterion(outputs, targets) # main loss
            return loss, outputs
        else:
            return outputs

    def gen_triples(self, input_ids, attention_mask, info, dependency_graph):
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask, dependency_graph)
            pred_triple = generate_triple(outputs, info, self.args, self.num_classes)
        return pred_triple

    @staticmethod
    def get_loss_weight(args):
        return {"relation": args.rel_loss_weight, "head_entity": args.head_ent_loss_weight, "tail_entity": args.tail_ent_loss_weight}





