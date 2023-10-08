import spacy
import numpy as np
import pickle
import pdb
from transformers import BertTokenizerFast


def adj_dependcy_tree(argments, max_length):
    nlp = spacy.load('en_core_web_sm')
    depend = []
    depend1 = []
    doc = nlp(str(argments))
    d = {}
    i = 0
    for (_, token) in enumerate(doc):
        if str(token) in d.keys():
            continue
        d[str(token)] = i
        i = i+1
    for token in doc:
        depend.append((token.text, token.head.text))
        depend1.append((d[str(token)], d[str(token.head)]))

    ze = np.identity(max_length)
    for (i, j) in depend1:
        if i>=max_length or j>=max_length:
            continue
        ze[i][j] = 1

    return ze

def process(filename, max_length):
    # print("just try if run this function")
    fin = open(filename, 'r', encoding='utf-8')
    lines = fin.readlines()
    fin.close()
    idx2graph = {}
    fout = open(filename+'.graph', 'wb')
    
    for i in lines:
        sentence = i.strip()
        try:
            adj_matrix = adj_dependcy_tree(sentence, max_length) # array.shape: 128*128
            pdb.set_trace()
        except:
            print(filename)
            raise
        idx2graph[sentence] = adj_matrix
    pickle.dump(idx2graph, fout)
    fout.close()

# process("/home/qtxu/SPN/data/Camera-COQE/dev_only_sentence_test.txt", 128) # spend much time
