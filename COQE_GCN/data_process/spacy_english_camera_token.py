import numpy as np
import spacy
import pickle
from transformers import BertTokenizerFast, BertTokenizer, AutoTokenizer
from spacy.tokens import Doc

from pdb import set_trace as stop

# Function： 构建英文数据集的邻接矩阵，比较句依据依存关系构图，非比较句则直接构建tokenize长度的主对角线为1的矩阵

parser_dict = {'empty': 0, 'self':1}
nlp = spacy.load('en_core_web_sm')
parser_tuple = nlp.get_pipe("parser").labels
count = 2
for parser in parser_tuple:
    parser_dict[parser] = count
    count += 1
   

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
    tokens = nlp(text)
    tokenized = tokenizer(text.split(" "), is_split_into_words=True, add_special_tokens=False)
    word_ids = tokenized.word_ids()
    words = text.split()
    matrix1 = np.zeros((len(word_ids), len(word_ids))).astype('float32')
    assert len(words) == len(list(tokens))
    assert (len(tokens) - 1) == max(word_ids)

    for i, idx in enumerate(word_ids):
        matrix1[i][i] = 1
        for j, id in enumerate(word_ids):
            if i == j:
                continue
            if tokens[id].is_ancestor(tokens[idx]):
                for token in tokens[id].subtree:
                    if token == tokens[idx]:
                        relation_type = token.dep_
                        matrix1[i][j] = parser_dict[relation_type]
                        # print(f" **{words[i]} ({tokens[idx]}) --{relation_type}--> {words[j]} ({tokens[id]})")
                        break
            elif tokens[idx].is_ancestor(tokens[id]):
                for token in tokens[idx].subtree:
                    if token == tokens[id]:
                        relation_type = token.dep_
                        matrix1[i][j] = parser_dict[relation_type]
                        # print(f" ##{words[j]} ({tokens[id]}) --{relation_type}--> {words[i]} ({tokens[idx]})")
                        break
            else:
                matrix1[i][j] = 0 # 若两个token之间没关系，则对应元素赋值0

    return matrix1

def process(filename):
    fin = open(filename, 'r', encoding='utf-8')
    lines = fin.readlines()
    fin.close()
    idx2graph = {}
    fout = open('/home/qtxu/Sentiment-SPN/data/Camera-COQE/dev_diff_'+'.graph', 'wb')
    for i in lines:
        sentence, label = i.strip().split("\t")
        sentence = sentence.strip() # add strip() for English 
        label = label.strip()
        if label == "1":
            adj_matrix = dependency_adj_matrix(sentence)
        elif label == "0":
            adj_matrix = np.eye(len(tokenizer.tokenize(sentence))) # 可能存在bert分词的时候，将一个单词拆分成多少
            # adj_matrix = np.eye(len(sentence.split(" "))) # 如果是非比较句，则构建的邻接矩阵是主对角线为1
            # 英文有空格作为分隔符
        # stop()
        idx2graph[sentence] = adj_matrix
    pickle.dump(idx2graph, fout)
    fout.close()

filename = "/home/qtxu/Sentiment-SPN/data/Camera-COQE/dev_sentences_labels.txt"
# filename = "/home/qtxu/Sentiment-SPN/data/Camera-COQE/try.txt"
process(filename)
# text = 'By the way you can use the same battery the S50 comes with .'
# dependency_adj_matrix(text)

