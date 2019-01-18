import json
with open('baidu/word2idx.json', encoding='UTF-8') as inf:
    w2i = json.load(inf)

with open('dict/custom.dict', encoding='UTF-8', mode='w') as out:
    for key in w2i.keys():
        out.write(key + "\n")
