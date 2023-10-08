import logging
from stanfordcorenlp import StanfordCoreNLP
import numpy as np
import re
from pdb import set_trace as stop
# from transformers import BertTokenFast
from transformers import BertTokenizerFast, BertTokenizer, AutoTokenizer
import pickle

# Function:判断bert解析结果与stanford解析结果的长度是否一致
def compara_stanf_bert_len(sent_tokened, stanf_test_dict):
    sent_bert_len = len(sent_tokened)
    last_key =  sorted(stanf_test_dict.keys())[-1] # 获取stanford的解析结果
    last_values = stanf_test_dict[last_key].rsplit("-")[0]
    spacy_len = int(last_key.rsplit("-")[1]) + len(last_values)
    if sent_bert_len == spacy_len:
        return "equal"
    elif sent_bert_len > spacy_len:
        return "longer"
    elif sent_bert_len < spacy_len:
        return "shorter"

def get_stan_dict(sen):
    words = list(zh_model.word_tokenize(sen)) #  ['这', '款', '车', '的', '发动机', '异于', '同类', '。']
    arcs = list(zh_model.dependency_parse(sen)) #使用stanford解析得到的结果, 包含root

    arcs.sort(key=lambda x: x[2]) # 按照在sentence中出现的次序排序
    # print("sorted resluts:",arcs) # 得到排序之后的arcs
    words = [w + "-" + str(idx) for idx, w in enumerate(words)]  #从0开始给使用stanford解析之后的结果排序

    rely_id = [arc[1] for arc in arcs]
    relation = [arc[0] for arc in arcs]

    heads = ['Root' if id == 0 else words[id - 1] for id in rely_id]

    stan_dict = {}
    for i in range(len(words)):
        stan_dict[words[i]] = heads[i]
        # print(relation[i] + '(' + words[i] + ', ' + heads[i] + ')')
    return stan_dict

def dependency_adj_matrix_zh(sen, sent_tokened):
    stanford_words = list(zh_model.word_tokenize(sen))
    stan_dict = get_stan_dict(sen)
    test_dict = {}
    count = 0
    for word in stanford_words:
        if word.isdigit():
            test_dict[count] = word
            count += 1
        elif len(re.findall(r"\d+[a-zA-Z]+[\u4e00-\u9fff]+", word)) != 0: # 匹配数字字母汉字格式的str, eg 4s站
            test_dict[count] = word
            count += 1
        elif len(re.findall(r"\d+[\u4e00-\u9fff]+|\d+[.]+\d+[a-zA-Z]+", word)) != 0: # 匹配数字汉字格式， eg:呈V or 1.8T
            test_dict[count] = word
            count +=1
        elif len(re.findall(r"[a-zA-Z]+\d+|\d+[.,-，]+\d+|\d+[+]+", word)) != 0:  # 匹配字母数字格式，或数字标点符号数字格式 eg:5W， 25.4 or 11-13
            test_dict[count] = word
            count += 1
        elif len(re.findall(r"[\u4e00-\u9fff]+\d|\d+[-]", word)) != 0: #匹配汉字数字格式，eg:
            test_dict[count] = word
            count += 1   
        elif re.findall(r"[a-zA-Z]|[a-zA-Z]+[-]+[a-zA-Z]+|[0-9]+[-]+", word):
            test_dict[count] = word
            count += 1   
        elif word == 'Ｗ８００Ｃ': # Ele test 特例情况
            test_dict[count] = word 
            count += 1
        elif re.findall(r'\d+[%.％]|\d+[,/]+\d+|[～]+[0-9]+', word): # 15%
            test_dict[count] = word
            count += 2   
        elif word == '～～～赞': # 特例情况
            test_dict[count] = word
            count += 2
        elif word == '昂达的-850':
            test_dict[count] = word
            count += 4
        else: # 正常的情况
            test_dict[count] = word
            count+=len(word)
    # print("test_dict", test_dict)

    # number = 0
    adj_matrix = np.eye(len(sent_tokened)).astype('float32') # sent_tokened:表示的是经过berttokenfast分词之后的结果
    for key_i, value_i in stan_dict.items():
        try:
            key_word, key_id = key_i.rsplit("-", 1)
        except:
            assert "key_i 错误"
        try:
            value_word, value_id = value_i.rsplit('-', 1) # 不同于split("-"), 其可以处理11-13-13这样的特殊情况
        except:
            continue

        key_start = next(key for key, value in test_dict.items() if value == key_word)
        value_start = next(key for key, value in test_dict.items() if value == value_word)

        if re.findall(r'\d+[.]+\d+[a-zA-Z]+|\d+|\d+[.]\d+[a-zA-Z]+|[a-zA-Z]+', key_word):
            len_key_word = 1
        else:
            len_key_word = len(key_word)

        if re.findall(r'\d+|\d+[.]\d+[a-zA-Z]+|[a-zA-Z]+', value_word):
            len_value_word = 1
        else:
            len_value_word = len(value_word)

        for i in range(key_start, key_start + len_key_word): # 注意此处的id是test_dict中的id,而不是stan_dict中的id
            for j in range(value_start, value_start + len_value_word):    
                try:
                    adj_matrix[i][j] = 1
                except:
                    print(key_start, key_word)
                    print(value_start, value_word)
                    stop()
                adj_matrix[j][i] = 1

    return adj_matrix

def count_len(stanford_words):
    sum_len = 0
    for i in range(len(stanford_words)):
        if re.findall(r"\d+[a-zA-Z]+[\u4e00-\u9fff]+|\d|[a-zA-Z]|\d+[a-zA-Z]+[\u4e00-\u9fff]+|[\u4e00-\u9fff]+\d", stanford_words[i]):
            sum_len += 1
        else:
            sum_len += len(stanford_words[i])
    return sum_len

def get_split(stanfor_results, bert_results):
    not_complete_in_bert_results = {key:value for key, value in stanfor_results.items() if 
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

#Function:将stanford解析得到的list，转换成dict的形式
def stan_list_to_dict(list1):
    list1_dict = {}
    for i in range(len(list1)):
        list1_dict[i]=list1[i]
    return list1_dict

def process(filename, tokenizer):
    fin = open(filename, 'r', encoding='utf-8')
    lines = fin.readlines()
    fin.close()
    idx2graph = {}
    fout = open(filename+'.graph', 'wb')
    num = 0 
    for i in lines:
        sen, label = i.strip().split("\t")
        sen = sen.strip()
        sent_tokened = tokenizer.tokenize(sen)
        stanford_words = list(zh_model.word_tokenize(sen))
        len_stanford_words = count_len(stanford_words)

        if label == "1":
            try:
                if len(sent_tokened) < len_stanford_words:
                    stan_convert_dict = stan_list_to_dict(stanford_words)
                    diff_results = get_split(stan_convert_dict, sent_tokened) # diff_results: 表示比较之后得到的不同的str
                    sen_new = sen
                    for diff in range(len(diff_results)):
                        sen_new = sen_new.replace(diff_results[diff], "#") # 将解析的不同词语使用特殊符号#替换， 对应bert解析的unk
                    # sen_new = sen
                    if "8XDL" in sen:
                        sen_new = sen.replace("8XDL", "#")
                    if "40XCd" in sen and "SHB022a" in sen and "12Xdvdr" in sen and "SEB026a" in sen:
                        sen_new = sen.replace("40XCd", "#")
                        sen_new = sen_new.replace("SHB022a", "#")
                        sen_new = sen_new.replace("12Xdvdr", "#")
                        sen_new = sen_new.replace("SEB026a", "#")
                    if "1g" in sen:
                        sen_new = sen.replace("1g", "#") # Ele dev 特殊情况
                    if "MX360" in sen and "C770" in sen :
                        sen_new = sen.replace("MX360", "#")

                    adj_matrix = dependency_adj_matrix_zh(sen_new, sent_tokened)
                    num += 1
                else:
                # elif len(sent_tokened) > len_stanford_words:
                    adj_matrix = dependency_adj_matrix_zh(sen, sent_tokened)
                    num += 1
                print("处理了{}条比较句".format(num))
            except:
                print(sen)
                stop()

        elif label == "0":
            adj_matrix = np.eye(len(sent_tokened)) # 可能存在bert分词的时候，将一个单词拆分成多少
            # 如果是非比较句，则构建的邻接矩阵是主对角线为1

        idx2graph[sen] = adj_matrix
    pickle.dump(idx2graph, fout)
    fout.close()


filename = "/home/qtxu/Sentiment-SPN/data/Ele-COQE/test_ele.txt"
# filename = "/home/qtxu/Sentiment-SPN/data/Car-COQE/train_test.txt"
bert_path = "/home/qtxu/PLM/bert-base-chinese"
tokenizer = BertTokenizerFast.from_pretrained(bert_path)
model_path = "/home/qtxu/PLM/stanford-corenlp-full-2018-02-27"
zh_model = StanfordCoreNLP(model_path, lang='zh', quiet=True, logging_level=logging.DEBUG)
process(filename, tokenizer)


# sentence = "想不到还带着一节电池,雷柏真是业良心啊!用了两天,感觉还不错,至于精确度,因为我不怎么玩游戏,所以不能提供这类信息.人个感觉与有线鼠标一样,只是少了一根线.另外,在Win7系统下,插上接收器打开鼠标后,系统自动安装的驱动吧,等待了大约一分钟左右就可以使用了,非常方便.终于听不到上一个有线鼠标的线与鼠标接头那里的线有些断开而每天听"噔噔噔噔"的声音了.感叹一下,冬天真是一个淘汰有线鼠标的季节啊!包装上说电池是可以用三个月,也不知道会不会,反正鼠标下面有个开关,平时不用和晚上睡觉的时候关掉可以省电.还有鼠标是不带灯的,这就是为什么可以续航三个月么?好吧,我这就开始实验,使用像我这么不算频繁的,到底可以用多长时间."    
# sent_tokened = tokenizer.tokenize(sentence)
# adj_matrix = dependency_adj_matrix_zh(sen_new, sent_tokened)
# print(adj_matrix)



