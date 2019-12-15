
from nltk.tokenize import RegexpTokenizer
import numpy as np
import pandas as pd
import re

tokenizer = RegexpTokenizer(r'\w+')

# Frequentist Inference Component

# tag-term in training dataset
data = pd.read_csv('data/dataset.csv', encoding='ISO-8859-1')
tag1 = pd.read_csv('data/tags_list.csv', encoding='ISO-8859-1')
data1_content = []
data1_tag = []
for i in range(10000):
    data["Content"][i] = tokenizer.tokenize(data["Content"][i])
    data["Tags"][i] = re.sub(r'[\[,\],\']', '', data["Tags"][i]).split(" ")
    data1_content.append(data["Content"][i])
    data1_tag.append(data["Tags"][i])
k = 9000
x_train_freq = data1_content[:k]
y_train = data1_tag[:k]
dict_tag = {}

for tag in tag1["Tags"]:
    tag_term = []
    for i in range(len(x_train_freq)):
        if tag in y_train[i]:
            for term in x_train_freq[i]:
                if term not in tag_term:
                    tag_term.append(term)
    dict_tag[tag] = tag_term

# predicting phase
data1_content = np.array(data1_content)
data1_tag = np.array(data1_tag)
k = 9000
x_train_freq = data1_content[:k]
x_test_freq = data1_content[k:]
y_train = data1_tag[:k]
y_test = data1_tag[k:]
xtr = []
xte = []
ytr = []
obj_tag_score = []
obj_tag_score_freq = []
final_score = []
xtr_freq = []
xte_freq = []
for i in range(len(x_train_freq)):
    d = x_train_freq[i]
    c = y_train[i]
    xtr_freq.append(d)
    ytr.append(c)
xtr = np.array(xtr)
xtr_freq = np.array(xtr_freq)
ytr = np.array(ytr)
for i in range(len(x_test_freq)):
    tag_score = []
    tag_score_freq = []
    for tag in tag1["Tags"]:
        count = len([i for i in x_test_freq[i] if i in dict_tag[tag]])
        tag_score_freq.append([tag, count])
    obj_tag_score_freq.append(tag_score_freq)

