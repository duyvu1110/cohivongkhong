import numpy as np
import spacy
import pickle
from transformers import BertTokenizerFast, BertTokenizer, AutoTokenizer
from spacy.tokens import Doc
import jieba

from pdb import set_trace as stop

# Function： 构建英文数据集的邻接矩阵，比较句依据依存关系构图，非比较句则直接构建tokenize长度的主对角线为1的矩阵

class WhitespaceTokenizer(object):
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words = text.split()
        # All tokens 'own' a subsequent space character in this tokenizer
        spaces = [True] * len(words)
        return Doc(self.vocab, words=words, spaces=spaces)

nlp = spacy.load('en_core_web_sm')  # zh_core_web_sm or en_core_web_sm
nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)
tokenizer = BertTokenizerFast.from_pretrained("/home/qtxu/PLM/bert-base-uncased") # /home/qtxu/PLM/bert-base-chinese or bert-base-uncased

def dependency_adj_matrix(text):
    # https://spacy.io/docs/usage/processing-text
    tokens = nlp(text)
    tokenized = tokenizer(text.split(" "), is_split_into_words=True, add_special_tokens=False)
    # stop()
    word_ids = tokenized.word_ids()
    words = text.split()
    matrix1 = np.zeros((len(word_ids), len(word_ids))).astype('float32')
    assert len(words) == len(list(tokens)) # make sure the len is same
    assert (len(tokens) - 1) == max(word_ids)
    
    for i, idx in enumerate(word_ids):
        # print("i{}, idx{}".format(i,idx))
        matrix1[i][i] = 1 # 主对角线是1， 保留自己本身的特征信息
        for j, id in enumerate(word_ids):
            if tokens[id] in tokens[idx].children or word_ids[j] == word_ids[i]: 
                # tokens[id] in tokens[idx].children：检查是否存在一个语法依赖关系，即 id 单词是否是 idx 单词的子节点
                # word_ids[j] == word_ids[i]：检查是否是同一个单词，因为语法依赖关系还保留自己本身的特征信息，所以要将矩阵对角线设置为 1
                matrix1[i][j] = 1
                matrix1[j][i] = 1
    return matrix1

def softmax(x):
    if len(x.shape) > 1:
        # matrix
        tmp = np.max(x, axis=1)
        x -= tmp.reshape((x.shape[0], 1))
        x = np.exp(x)
        tmp = np.sum(x, axis=1)
        x /= tmp.reshape((x.shape[0], 1))
    else:
        # vector
        tmp = np.max(x)
        x -= tmp
        x = np.exp(x)
        tmp = np.sum(x)
        x /= tmp
    return x

def process(filename):
    fin = open(filename, 'r', encoding='utf-8')
    lines = fin.readlines()
    fin.close()
    idx2graph = {}
    fout = open(filename+'.graph', 'wb')
    for i in lines:
        sentence, label = i.strip().split("\t")
        if label == "1":
            adj_matrix = dependency_adj_matrix(sentence)
        elif label == "0":
            adj_matrix = np.eye(len(tokenizer.tokenize(sentence))) # 可能存在bert分词的时候，将一个单词拆分成多少
            # adj_matrix = np.eye(len(sentence.split(" "))) # 如果是非比较句，则构建的邻接矩阵是主对角线为1
            # 英文有空格作为分隔符
        idx2graph[sentence] = adj_matrix
    pickle.dump(idx2graph, fout)
    fout.close()

filename = "/home/qtxu/Sentiment-SPN/data/Camera-COQE/dev_sentences_labels.txt"
process(filename)
# text = 'By the way you can use the same battery the S50 comes with .'
# dependency_adj_matrix(text)
