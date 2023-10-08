# Function:实现对中文文本单个字级别的依存关系邻接矩阵的构建
import spacy
from transformers import BertTokenizerFast
import numpy as np
from pdb import set_trace as stop
import pickle
import re


def spacy_result_dict(text):
    nlp = spacy.load("zh_core_web_sm") # 加载spaCy模型
    doc = nlp(text)
    spacy_tokens = []
    for token in doc:
        spacy_tokens.append(token)
    # 依据spacy的分词解析结果，存放开始的index
    test_dict = {}
    count = 0
    for word in spacy_tokens:
        if word.like_num: # 匹配纯数字的情况，长度为1
            test_dict[count] = str(word)
            count += 1  
        elif re.findall(r"[a-zA-Z]", word.text): # 匹配纯字母的情况，长度为1
            test_dict[count] = str(word)
            count += 1
        else: # 正常的情况
            test_dict[count] = str(word)
            count+=len(word)
    return test_dict

def get_split(spacy_results, bert_results):
    not_complete_in_bert_results = {key:value for key, value in spacy_results.items() if 
    value not in bert_results and value not in ''.join(bert_results)} # 获取在bert中完全未出现的词语
    concatenated_values = []
    current_list = []

    previous_key = None
    for key in sorted(not_complete_in_bert_results.keys()):
        if previous_key is not None and key == previous_key + 1:
            current_list.append(not_complete_in_bert_results[key])
        else:
            if current_list:
                concatenated_values.append(''.join(current_list))
            current_list = [not_complete_in_bert_results[key]]
        previous_key = key

    if current_list:
        concatenated_values.append(''.join(current_list))
    return concatenated_values


def dependency_adj_matrix_zh(text, sent_tokened):
    # sen_tokens_text_list = tokenizer.tokenize(text)
    nlp = spacy.load("zh_core_web_sm") # 加载spaCy模型
    doc = nlp(text) # 解析文本

    spacy_tokens = []
    for token in doc:
        spacy_tokens.append(token) # ['油耗', '比', '骑车', '要', '高', '。']
    # print(spacy_tokens)

    # 依据spacy的分词解析结果，存放开始的index
    test_dict = {}
    count = 0
    for word in spacy_tokens:
        # print(type(word.text))
        if word.like_num:
            test_dict[count] = word
            count += 1
        elif word.text == "DX":
            test_dict[count] = word
            count += 1  
        elif len(re.findall(r"\d+[a-zA-Z]+[\u4e00-\u9fff]+", word.text)) != 0: # 匹配数字字母汉字格式的str, eg 4s站
            test_dict[count] = word
            count += 1
        elif len(re.findall(r"\d+[\u4e00-\u9fff]+", word.text)) != 0: # 匹配数字汉字格式， eg:
            test_dict[count] = word
            count +=1
        elif len(re.findall(r"[a-zA-Z]+\d+", word.text)) != 0:  # 匹配字母数字格式， eg:5W
            test_dict[count] = word
            count += 1
        elif len(re.findall(r"[\u4e00-\u9fff]+\d", word.text)) != 0: #匹配汉字数字格式，eg:
            test_dict[count] = word
            count += 1   
        elif re.findall(r"[a-zA-Z]", word.text):
            test_dict[count] = word
            count += 1         
        else: # 正常的情况
            test_dict[count] = word
            count+=len(word)
    # print("spacy results:", test_dict)
    # print(test_dict) # {0: '油耗', 2: '比', 3: '骑车', 5: '要', 6: '高', 7: '。'}

    number = 0
    adj_matrix = np.eye(len(sent_tokened))
    i = 0
    while i < len(sent_tokened):
        word = spacy_tokens[number]
        word_sp = list(word.text)
        for child in word.children:
            adj_word_list = list(child.text) # 具有依存关系的词语
            word_list = list(word.text) # spacy分词的结果中的某一个词
            child_key = next(key for key, val in test_dict.items() if val == child) # obtain the start index of child
            word_key = next(key for key, val in test_dict.items() if val == word) # obtain the start index of spacy_word
            # print("child:{}, word:{}".format(child, word))
            for m in range(child_key, len(adj_word_list) + child_key):
                for n in range(word_key, len(word_list) + word_key):
                    try:
                        adj_matrix[m][n] = 1
                        adj_matrix[n][m] = 1
                    except:
                        print(text)
                        # stop()
        i += len(word_sp)
        number += 1
    return adj_matrix

def compara_spacy_bert_len(sent_tokened, spacy_test_dict):
    sent_bert_len = len(sent_tokened)
    last_key =  sorted(spacy_test_dict.keys())[-1]
    last_values = spacy_test_dict[last_key]
    spacy_len = last_key + len(last_values)
    if sent_bert_len == spacy_len:
        return "equal"
    elif sent_bert_len > spacy_len:
        return "longer"
    elif sent_bert_len < spacy_len:
        return "shorter"
    

def process(filename, tokenizer):
    fin = open(filename, 'r', encoding='utf-8')
    lines = fin.readlines()
    fin.close()
    idx2graph = {}
    fout = open(filename+'.graph', 'wb')
    for i in lines:
        sentence, label = i.strip().split("\t")
        if "100K" in sentence:
            sentence = sentence.replace("100K", "#")
        sent_tokened = tokenizer.tokenize(sentence) #obtain bert解析的结果 type:list
        if label == "1":
            spacy_test_dict = spacy_result_dict(sentence) # obtain spacy解析的结果 type:dict
            compared_result = compara_spacy_bert_len(sent_tokened, spacy_test_dict) # 判断两个长度是否相等，如果相等则继续，如果不相等则需要进行一定的处理
            if compared_result == "equal": 
                adj_matrix = dependency_adj_matrix_zh(sentence, sent_tokened)
            elif compared_result == "shorter": # len(bert_token) < len(spacy)
                different_results = get_split(spacy_test_dict, sent_tokened) # different_results type: list
                for i in range(len(different_results)):
                    pattern = '^[\u4e00-\u9fa5a-zA-Z]+$'
                    char_pattern = '[a-zA-Z]'
                    try:
                        if bool(re.match(pattern, different_results[i])): # 如果是汉字字母形式，e.g., 呈V，仅将字母替换成#，汉字不变
                            replace_feature = re.findall(char_pattern,different_results[i])
                            sentence = sentence.replace("".join(replace_feature),"#")
                        else:
                            sentence = sentence.replace(different_results[i], "#")
                    except:
                        print(sentence)
                        stop()
                adj_matrix = dependency_adj_matrix_zh(sentence, sent_tokened)

            elif compared_result == "longer":
                adj_matrix = dependency_adj_matrix_zh(sentence, sent_tokened)

        elif label == "0":
            adj_matrix = np.eye(len(sent_tokened))
        idx2graph[sentence] = adj_matrix
    pickle.dump(idx2graph, fout)
    fout.close()        

filename = "/home/qtxu/Sentiment-SPN/data/Car-COQE/dev_sentences_labels.txt"
bert_path = "/home/qtxu/PLM/bert-base-chinese"
tokenizer = BertTokenizerFast.from_pretrained(bert_path)
process(filename, tokenizer)