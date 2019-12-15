
from nltk.probability import FreqDist
from nltk.tokenize import RegexpTokenizer
import numpy as np
import pandas as pd
import csv
import re

tokenizer = RegexpTokenizer(r'\w+')

# Similarity Based Ranking Component
def ComputeIDF(document, obj, item, content):
    import math
    f1 = FreqDist(item)
    f2 = FreqDist(obj)
    se = []
    for i in range(len(content)):
        DF = math.log10(len(document) / f1[content[i]])
        TF = f2[content[i]] / len(obj)
        w = DF * TF
        se.append(w)
    return se


def SimScore(obj1, obj2):
    a = np.array(obj1)
    b = np.array(obj2)
    score = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return score


def SimRank(x_train_set, x_test):
    score = []
    for i in range(len(x_train_set)):
        a = SimScore(x_train_set[i], x_test)
        score.append([i, a])
    score.sort(key=(lambda x: x[1]), reverse=True)
    return score


def Result(tag, score, y_train):
    total = 0
    count = 0
    for m in range(50):
        if tag in y_train[score[m][0]]:
            count = count + 1
        total = total + len(y_train[score[m][0]])
    simscore = count / total
    return simscore


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
# Frequentist Inference Component
# tag-term in training dataset
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

# to obtain document frequency
c = []
for i in range(len(data1_content)):
    l = []
    for j in range(len(data1_content[i])):
        if data1_content[i][j] not in l:
            l.append(data1_content[i][j])
    c.append(l)
item1 = []
for i in range(len(c)):
    item1.extend(c[i])

# to obtain item list
content = []
for i in range(len(data1_content)):
    for j in range(len(data1_content[i])):
        if data1_content[i][j] not in content:
            content.append(data1_content[i][j])

#Transfer IDF
data1_content=np.array(data1_content)
idf=[]
for i in range(len(data1_content)):
    a=ComputeIDF(data1_content,data1_content[i],item1,content)
    idf.append(a)

# predicting phase
data1_content = np.array(data1_content)
data1_tag = np.array(data1_tag)
idf = np.array(idf)
x_train = idf[:k]
x_test = idf[k:]
y_train = data1_tag[:k]
y_test = data1_tag[k:]
xtr = []
xte = []
ytr = []
obj_tag_score_sim = []
for i in range(len(x_train)):
    a = x_train[i]
    c = y_train[i]
    xtr.append(a)
    ytr.append(c)
xtr = np.array(xtr)
ytr = np.array(ytr)
for i in range(len(x_test)):
    b = x_test[i]
    xte.append(b)
    tag_score = []
    obj_rank = SimRank(xtr, xte[i])
    for tag in tag1["Tags"]:
        SimScore_i_t = Result(tag, obj_rank, y_train)
        tag_score.append([tag, SimScore_i_t])
    obj_tag_score_sim.append(tag_score)


# Frequentist Inference Component
# predicting phase
x_train_freq = data1_content[:k]
x_test_freq = data1_content[k:]
y_train = data1_tag[:k]
y_test = data1_tag[k:]
xtr = []
xte = []
ytr = []
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


# sum combination with normalized frequentist score
sum_list = []
for i in range(len(obj_tag_score_freq)):
    s = 0
    for j in range(len(obj_tag_score_freq[i])):
        s = s + obj_tag_score_freq[i][j][1]
    sum_list.append(s)

com_list = []
for i in range(len(obj_tag_score_freq)):
    com_s = []
    for j in range(len(obj_tag_score_freq[i])):
        obj_tag_score_freq[i][j][1] = obj_tag_score_freq[i][j][1] / sum_list[i]
        score = obj_tag_score_sim[i][j][1] + obj_tag_score_freq[i][j][1] / sum_list[i]
        com_s.append([obj_tag_score_freq[i][j][0], score])
    com_list.append(com_s)

# Combination: intesection of two methods
f = open('data/output_tag.csv', 'w', newline='')
write = csv.writer(f)
result1 = []
result2 = []
for i in range(len(x_test)):
    obj_tag_score_freq[i].sort(key=(lambda x: x[1]), reverse=True)
    obj_tag_score_sim[i].sort(key=(lambda x: x[1]), reverse=True)
    obj_top_tag_sim = []
    obj_top_tag_freq = []
    for j in range(20):
        result1.extend([obj_tag_score_sim[i][j][0],obj_tag_score_freq[i][j][0]])
        result2.extend([obj_tag_score_sim[i][j][1],obj_tag_score_freq[i][j][1]])
    #     obj_top_tag_sim.append(obj_tag_score_sim[i][j][0])
    #     obj_top_tag_freq.append(obj_tag_score_freq[i][j][0])
    # intersection = [i for i in obj_top_tag_sim if i in obj_top_tag_freq]
    # result.append(intersection)
    write.writerow(result1)
    write.writerow(result2)

#recall@k score
score=0
k=10
for i in range(len(x_test)):
    count=0
    com_list[i].sort(key=(lambda x:x[1]),reverse=True)
    for j in range(k):
        if com_list[i][j][0] in y_test[i]:
            count=count+1
    score=score+count/len(y_test[i])
score=score/len(x_test)
print("sum combination",score)