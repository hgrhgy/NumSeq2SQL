import json

import numpy as np
import datetime

LOCAL_TEST=True
BERT = True
N_word = 300



word_to_idx = {'<UNK>':0, '<BEG>':1, '<END>':2}
word_num = 3
embs = [np.zeros(N_word,dtype=np.float32) for _ in range(word_num)]

with open('G:/BaiduNetdiskDownload/sgns.target.word-ngram.1-2.dynwin5.thr10.neg5.dim300.iter5/sgns.target.word-ngram.1-2.dynwin5.thr10.neg5.dim300.iter5', encoding='UTF-8') as inf:
    lines = inf.readline()
    cnt, dim = [int(a.strip()) for a in lines.split(' ')]
    for i in range(cnt):
        line = inf.readline()
        arr = [a.strip() for a in line.split(' ')]
        tok = arr.pop(0)
        word_to_idx[tok] = word_num
        embs.append([float(d) for d in arr[0:300]])
        word_num += 1

print ("Length of used word vocab: %s"%len(word_to_idx))
emb_array = np.stack(embs, axis=0)
with open('./baidu/word2idx.json', 'w', encoding="UTF-8") as outf:
    json.dump(word_to_idx, outf)
np.save(open('./baidu/usedwordemb.npy', 'wb'), emb_array)
