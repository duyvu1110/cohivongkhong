import torch.nn as nn
import torch
from transformers import BertModel
import torch.nn.functional as F

from pdb import set_trace as stop


class BERTEncoder_GCN(nn.Module):
    def __init__(self,args):
        super(BERTEncoder_GCN,self).__init__()
        self.args = args
        self.bert = BertModel.from_pretrained(args.bert_directory)
        self.config = self.bert.config
        self.rnn1 = nn.GRU(input_size=self.config.hidden_size, hidden_size=self.config.hidden_size//2, 
                           num_layers=2, batch_first=True, 
                           bidirectional=True)
        self.gcn1 = GCN(self.config.hidden_size, self.config.hidden_size)
        # self.gcn2 = GCN(self.config.hidden_size, self.config.hidden_size)
        self.dense = nn.Linear(self.config.hidden_size * 2,self.config.hidden_size)
        self.dropout = nn.Dropout(0.1) # 

    
    def forward(self, input_ids, attention_mask, dependency_graph):
        out = self.bert(input_ids, attention_mask=attention_mask) # input_ids: bs,hidden
        last_hidden_state, pooler_output = out.last_hidden_state, out.pooler_output # last_hidden_state: bsz, seq, hid; pooler_output: bsz, hidden_size
        # sequence_output, _ = self.rnn1(last_hidden_state)
        # add 残差信息的GCN类似于Transformer中的结构
        x1 = F.relu(self.gcn1(self.args.device, last_hidden_state, dependency_graph))
        x1_ = torch.cat((last_hidden_state, x1), dim=-1)
        x1_f = self.dense(x1_)
        # one layer GCN

        # x2 = F.relu(self.gcn2(self.args.device, x1_f, dependency_graph))
        # x2_ = torch.cat((x1_f, x2), dim=-1)
        # x2_f = self.dense(x2_)

        ##### 方案三：  totally same with chen ####
        # last_hidden_state,_ = self.rnn1(last_hidden_state)
        # x1 = F.relu(self.gcn1(self.args.device, last_hidden_state, dependency_graph))
        # x2 = F.relu(self.gcn2(self.args.device, x1, dependency_graph))
        # x2_f = torch.cat((last_hidden_state,x2), dim=-1) #残差连接
        # x2_f = self.dense(x2_f)
        # bz, seq, h_d = last_hidden_state.shape
        # assert x2_f.shape == (bz, seq, h_d)
        # sequence_out = self.dropout(x2_f)

        #####   totally same with chen ####

        bz, seq, h_d = last_hidden_state.shape
        assert x1_f.shape == (bz, seq, h_d)
        sequence_out = self.dropout(x1_f)

        return sequence_out, pooler_output


class GCN(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Linear(in_features, out_features)

    def forward(self, used_device, sentence_embedding, adjacency_matrix):
        stop()
        sentence_embedding = sentence_embedding.to(used_device)
        adjacency_matrix = adjacency_matrix.to(used_device)
        # lens, lens * lens, h_d --> lens, in_hd

        text_weighted = torch.matmul(adjacency_matrix, sentence_embedding) 
        hidden = self.weight(text_weighted)
        denom = torch.sum(adjacency_matrix, dim =2, keepdim=True) + 1  # keepdim=True: function, 在求和后，保留原来的维度
        output = hidden /denom
        return output
