import torch.nn as nn
import torch
from transformers import BertModel


class SeqEncoder(nn.Module):
    def __init__(self, args):
        super(SeqEncoder, self).__init__()
        self.args = args
        self.bert = BertModel.from_pretrained(args.bert_directory)
        self.config = self.bert.config
        # self.gat_attention = nn.MultiheadAttention(self.config.hidden_size, num_heads=args.num_heads, dropout=args.gat_dropout) # 设置的num_heads要能够被1024整除

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids, attention_mask=attention_mask) # input_ids: bs,hidden
        last_hidden_state, pooler_output = out.last_hidden_state, out.pooler_output
        # last_hidden_state = self.gat(last_hidden_state, last_hidden_state, attention_mask)

        return last_hidden_state, pooler_output

    def gat(self, target, source, key_padding_mask):
        '''target: bsz, max_len, hidden_size   '''
        key_padding_mask = key_padding_mask == 0
        # adjacency_matrix = adjacency_matrix.to(self.args.device) # bsz, max_len, max_len
        output, attention_adj = self.gat_attention(target.transpose(0,1),
                                                   source.transpose(0,1),
                                                   source.transpose(0,1),
                                                   key_padding_mask=key_padding_mask,
                                                   ) # output: max_len, bsz, hidden_size attention_adj: bsz, max_len, max_len
        # target = output.transpose(1,0) # method 1
        target = target + output.transpose(1,0) # method 2
        # target = torch.cat((target, output.transpose(1,0)), dim=-1) # method 3

        return target
    
class SeqEncoder_last(nn.Module):
    def __init__(self, args):
        super(SeqEncoder, self).__init__()
        self.args = args
        self.bert = BertModel.from_pretrained(args.bert_directory)
        self.config = self.bert.config

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        last_hidden_state, pooler_output = out.last_hidden_state, out.pooler_output

        if self.args.use_last_hidden_state=="True":
            return last_hidden_state, pooler_output
        elif self.args.use_last_hidden_state == "False": # use last four hidden state concat
            hidden_state = out.hidden_states[-4:]
            hidden_state = torch.stack(hidden_state, dim=-1)
            batch_size, length, _, _ = hidden_state.shape 
            hidden_state = hidden_state.reshape(batch_size, length, -1)
            return hidden_state, pooler_output
        
        # return hidden_state, pooler_output # 4*bs,se,hi
        # return last_hidden_state, pooler_output # bs,se,hi
