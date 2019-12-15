
from nltk.tokenize import RegexpTokenizer
from nltk.probability import FreqDist
import pandas as pd
import csv
import re

tokenizer = RegexpTokenizer(r'\w+')

data = pd.read_csv('data/dataset.csv', encoding='ISO-8859-1')
tag1 = pd.read_csv('data/tags_list.csv', encoding='ISO-8859-1')
data1_content = []
data1_tag = []
for i in range(10000):
    data["Content"][i] = tokenizer.tokenize(data["Content"][i])
    data["Tags"][i] = re.sub(r'[\[,\],\']', '', data["Tags"][i]).split(" ")
    data1_content.append(data["Content"][i])
    data1_tag.append(data["Tags"][i])
# generate network
# tag_dict
k = 9000
x_train_freq = data1_content[:k]
y_train = data1_tag[:k]
dict_tag_tag = {}
tag_f = []
for obj in y_train:
    for i in obj:
        tag_f.append(i)
tag_freq = FreqDist(tag_f)

# duplicate
for tag in tag1["Tags"]:
    tag_tag = []
    for i in range(len(x_train_freq)):
        if tag in y_train[i]:
            for item in y_train[i]:
                if item != tag:
                    tag_tag.append(item)
    dict_tag_tag[tag] = tag_tag

    # not duplicate
dict_tag_nd = {}
for tag in tag1["Tags"]:
    tag_tag = []
    for i in range(len(x_train_freq)):
        if tag in y_train[i]:
            for item in y_train[i]:
                if item != tag and item not in tag_tag:
                    tag_tag.append(item)
    dict_tag_nd[tag] = tag_tag

    # network
tag_index = 0
head = ['']
tag_dict = {}
for tag in tag1["Tags"]:
    head.append(tag)
    tag_dict[tag] = tag_index
    tag_index += 1
network = [[0 for i in range(len(tag_dict))] for i in range(len(tag_dict))]
for tag in tag1["Tags"]:
    for item in dict_tag_nd[tag]:
        # print(item)
        x_index = tag_dict[tag]
        try:
            y_index = tag_dict[item]
        except:
            y_index = 268
        intersaction = FreqDist(dict_tag_tag[tag])[item]
        union = tag_freq[tag] + tag_freq[item] - intersaction
        network[x_index][y_index] = intersaction / union

    # write csv
f = open('data/network_matrix.csv', 'w', newline='')
write = csv.writer(f)
write.writerow(head)
for i in range(len(head) - 1):
    w = [head[i + 1]]
    w.extend(network[i])
    write.writerow(w)

