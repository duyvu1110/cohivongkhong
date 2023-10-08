import os
import torch
from torch import nn
import json
from tqdm import tqdm, trange
from typing import List
from transformers import AutoTokenizer, BertTokenizerFast
from collections import defaultdict
import re

from pdb import set_trace as stop
import numpy as np
import pickle

EMO_MAP = {
    -1: 1,
    0: 2,
    1: 3,
    2: 4
}

def pass_offset(data_path, offset):
    """
    Set the offset in english dataset(Camera-COQE) starts from 0.
    """
    if 'Camera' in data_path:
        return offset - 1
    else:
        return pass_offset

def proc_raw_offset(offset_spans: str, text, data_path):
    if offset_spans == '':
        # use 1 to denotes the empty span
        return (0, 0)
    # 7&&all 8&&of 9&&the 10&&Nikon 11&&DLSR 12&&models
    if 'Camera' in data_path:
        offsets = re.findall(r'([0-9]+)&&(\S+)', offset_spans)
    else:
        offsets = re.findall(r'([0-9]+)&(\S+)', offset_spans) # type(offset_spans):str
    # [7&&all, 8&&of, 9&&the, 10&&Nikon, 11&&DLSR, 12&&models]

    return int(offsets[0][0]), int(offsets[-1][0]) # obtain start token and end token for each span, [('5', '幸'), ('6', '福'), ('7', '使'), ('8', '者')]--> (5,8)
    indexs = []
    for off, off_text in offsets:
        pos = pass_offset(data_path, int(off))
        span_word = off_text.strip()
        if span_word[-1] == ',':
            span_word = span[:-1]

        if 'Camera' in data_path:
            pos_text = text.split(' ')[pos]
        else:
            pos_text = text[pos]
        assert pos_text.strip() == span_word

        # if len(indexs) > 0:
        #     assert indexs[-1] + 1 == pos
        indexs.append(pos)
    return indexs[0], indexs[-1]


def process_line_GCN(args, text_line, label_line, tokenizer: BertTokenizerFast, sample_id, file_name):
    text = text_line.split('\t')[0].strip() # text_line:当前行， text：sentence
    have_triples = int(text_line.split('\t')[1]) # obtain the label is comparative (1) or no-comparative (0)

    re_result = re.findall(r'\[\[(.*?)\];\[(.*?)\];\[(.*?)\];\[(.*?)\];\[(.*?)\]\]', label_line)
    # label_line--> re_result:去除原始数据中的[]，以及;
    raw_labels: List = [[x for x in y] for y in re_result] #一个样本存放在一个list中 
    # List of triples for a sentence

    tokens_output = tokenizer(text, max_length=args.max_text_length - 1, pad_to_max_length=True) # input_ids, token_type_ids, attention_mask
    token_ids = [tokenizer.convert_tokens_to_ids('[unused1]')] + tokens_output['input_ids']
    # sample = {'token_ids': token_ids, 'labels': [], 'sample_id': sample_id}
    tokens_len = len(tokenizer.tokenize(text))
    all_tokens_len = tokens_len + 3

    file_size = os.stat(file_name).st_size
    if file_size == 0: # 增加一个判断条件，判断是否为空
        raise Exception("Error:File is empty.")
    else:
        fgraph = open(file_name, "rb") # "rb"是文件打开模式。"r"表示读取模式，"b"表示二进制模式。
        idx2graph = pickle.load(fgraph)
        fgraph.close() # very import, sure added
  
    sample = {'token_ids': token_ids, 'labels': [], 'sample_id': sample_id} #,'dependency_matrix':[]}

    if have_triples == 0:
        # sample['dependency_matrix'] = torch.eye(args.max_text_length).float() # if no comparative, 则赋值是主对角线是1的max_len*max_len的矩阵
        init_adj = np.eye(all_tokens_len)
        cur_adj = torch.tensor(np.pad(init_adj,((0,args.max_text_length - all_tokens_len),(0,args.max_text_length - all_tokens_len))))
        sample['dependency_matrix'] = cur_adj
        return sample

    for tri in raw_labels:
        # tri: [sub, obj, aspect, opinion, sentiment]
        # data_path:主要是为了区分中英文数据集，原始数据中，存放格式不一致
        sub_offset = proc_raw_offset(tri[0], text, args.data_path) # pro_raw_offset: obtain the start offset and end offset for each element
        obj_offset = proc_raw_offset(tri[1], text, args.data_path)
        aspect_offset = proc_raw_offset(tri[2], text, args.data_path)
        view_offset = proc_raw_offset(tri[3], text, args.data_path)

        sentiment_label = have_triples * EMO_MAP[int(tri[4])]

        if 'Camera' in args.data_path:
            sample['labels'].append({
                'sub_start_index': tokens_output.word_to_tokens(sub_offset[0]).start,
                'sub_end_index': tokens_output.word_to_tokens(sub_offset[1]).end - 1,
                'obj_start_index': tokens_output.word_to_tokens(obj_offset[0]).start,
                'obj_end_index': tokens_output.word_to_tokens(obj_offset[1]).end - 1,
                'aspect_start_index': tokens_output.word_to_tokens(aspect_offset[0]).start,
                'aspect_end_index': tokens_output.word_to_tokens(aspect_offset[1]).end - 1,
                'opinion_start_index': tokens_output.word_to_tokens(view_offset[0]).start,
                'opinion_end_index': tokens_output.word_to_tokens(view_offset[1]).end - 1,
                'relation': sentiment_label
            })
            try:
                dependency_adj_matrix = idx2graph[text] # obtain the sen_len adj_matrix
                dependency_adj_matrix_paded = np.pad(dependency_adj_matrix,((2,args.max_text_length-len(dependency_adj_matrix)-2),(2,args.max_text_length-len(dependency_adj_matrix)-2)), 'constant') # pad the seq_len to args.max_text_length
                dependency_adj_matrix_paded[0][0] = 1 #针对unused,主对角线是1 
                dependency_adj_matrix_paded[1][1] = 1 #针对cls,主对角线是1 
                dependency_adj_matrix_paded[2+len(dependency_adj_matrix)][2+len(dependency_adj_matrix)]=1 #针对sep,主对角线是1 
                sample['dependency_matrix']=torch.tensor(dependency_adj_matrix_paded) 
            except:
                print(text)
                stop()
        else:
            sample['labels'].append({
                'sub_start_index': tokens_output.char_to_token(sub_offset[0]) + 1,
                'sub_end_index': tokens_output.char_to_token(sub_offset[1]) + 1,
                'obj_start_index': tokens_output.char_to_token(obj_offset[0]) + 1,
                'obj_end_index': tokens_output.char_to_token(obj_offset[1]) + 1,
                'aspect_start_index': tokens_output.char_to_token(aspect_offset[0]) + 1,
                'aspect_end_index': tokens_output.char_to_token(aspect_offset[1]) + 1,
                'opinion_start_index': tokens_output.char_to_token(view_offset[0]) + 1,
                'opinion_end_index': tokens_output.char_to_token(view_offset[1]) + 1,
                'relation': sentiment_label
            })
            try:
                dependency_adj_matrix = idx2graph[text] # obtain the sen_len adj_matrix
                dependency_adj_matrix_paded = np.pad(dependency_adj_matrix,((2,args.max_text_length-len(dependency_adj_matrix)-2),(2,args.max_text_length-len(dependency_adj_matrix)-2)), 'constant') # pad the seq_len to args.max_text_length
                dependency_adj_matrix_paded[0][0] = 1 #针对unused,主对角线是1 
                dependency_adj_matrix_paded[1][1] = 1 #针对cls,主对角线是1 
                dependency_adj_matrix_paded[2+len(dependency_adj_matrix)][2+len(dependency_adj_matrix)]=1 #针对sep,主对角线是1 
                sample['dependency_matrix']=torch.tensor(dependency_adj_matrix_paded)  
            except:
                print(text)
                stop()
                pass
    
    return sample
        
def load_data_GCN(args, mode: str, dep_mode_path:str):
    # English: f'{dep_mode_path}_only_sentence.txt.graph'
    # Chinese: f'{dep_mode_path}_sentences_labels.txt.graph'
    if 'Camera' in args.data_path:
        cur_path = f'{dep_mode_path}_only_sentence.txt.graph'
    elif 'Car' in args.data_path or 'Ele' in args.data_path:
        cur_path = f'{dep_mode_path}_sentences_labels.txt.graph'
        
    dependency_mode_path = os.path.join(args.data_path, cur_path)
    # dependency_mode_path = os.path.join(args.data_path, f'{dep_mode_path}_only_sentence.txt.graph') # dependency_mode_path demotes the path to save the adjacency matirx of each sentence
    raw_data = []
    with open(os.path.join(args.data_path, f'{mode}.txt'), 'r') as f:
        for line in f:
            raw_data.append(line)
    all_samples = []
    line_id, i = 0, 0
    text_line, label_line = '', ''
    for line_id in trange(len(raw_data), desc=f'processing data for mode {mode}'): # load all dataset, include sentence and quintuple
        cur_line = raw_data[line_id]

        if len(cur_line.split('\t')) != 2: # denotes quintuple, not sentence and corresponding label
            label_line += '\n' + cur_line
        else:
            # a new text line, so push the last text and update text_line
            if text_line != '':
                all_samples.append(process_line_GCN(args, text_line, label_line, args.tokenizer, i, dependency_mode_path))
                i += 1
            text_line = cur_line
            label_line = ''

    all_samples.append(process_line_GCN(args, text_line, label_line, args.tokenizer, i, dependency_mode_path))

    return all_samples

def build_collate_fn_GCN(args):
    def collate_fn_GCN(batch):
        input_ids = torch.tensor([sample['token_ids'] for sample in batch], device=args.device, dtype=torch.long)
        seq_ids = [sample['sample_id'] for sample in batch]
        dependency_adj_matrix = torch.cat([sample['dependency_matrix'].unsqueeze(0) for sample in batch],0).float()
        labels = []

        for sample in batch:
            target = {
                'sub_start_index': [],
                'sub_end_index': [],
                'obj_start_index': [],
                'obj_end_index': [],
                'aspect_start_index': [],
                'aspect_end_index': [],
                'opinion_start_index': [],
                'opinion_end_index': [],
                'relation': [],
            }
            for tri in sample['labels']:
                for k in tri:
                    target[k].append(tri[k])

            for k in target: # target: dict,  dict_keys(['sub_start_index', 'sub_end_index', 'obj_start_index', 'obj_end_index', 'relation'])
                # assert len(target[k]) <= args.num_generated_triples  # num_generated_triples: default=10
                try:
                    assert len(target[k]) <= args.num_generated_triples # args.num_generated_triples 最小值是17
                except AssertionError:
                    stop()

                target[k] = torch.tensor(target[k], device=args.device, dtype=torch.long)
            labels.append(target)
        return input_ids, labels, seq_ids, dependency_adj_matrix
    return collate_fn_GCN

