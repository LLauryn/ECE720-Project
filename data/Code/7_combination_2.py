
from nltk.tokenize import RegexpTokenizer
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
x_test = data1_content[9000:]
y_test = data1_tag[9000:]
network = pd.read_csv('data/network_matrix.csv')
f = open('data/output_tag.csv', 'r')
reader = csv.reader(f)
tag_index = 0
tag_dict = {}
result=[]
result_score=[]
for tag in tag1["Tags"]:
    tag_dict[tag] = tag_index
    tag_index += 1

#obtain intersection set
for obj in reader:
    seen1=[]
    seen2=[]
    temp1=[]
    temp2=[]
    score = next(reader)
    for i in range(len(obj)):
        if obj[i] not in seen1:
            seen1.append(obj[i])
            seen2.append(float(score[i]))
        else:
            temp1.append(obj[i])
            x=seen1.index(obj[i])
            temp2.append(seen2[x]+float(score[i]))

    result.append(temp1)
    result_score.append(temp2)

#reduce more add less based on network
k=10
tag_score_net_b=[]
for i in range(len(result)):
    obj_tag_b = []
    if len(result[i])>=k:
        for tag in result[i]:
            b=0
            for j in range(len(result[i])):
                # x_index=tag_dict[tag]
                y_index=tag_dict[result[i][j]]
                # a=a+float(network[tag][y_index])*result_score[i][j]
                b = b + float(network[tag][y_index])
            obj_tag_b.append([tag, b])
    else:
        for tag in tag1["Tags"]:
            b=0
            if tag not in result[i]:
                for j in range(len(result[i])):
                    # x_index=tag_dict[tag]
                    y_index = tag_dict[result[i][j]]
                    # a = a + float(network[tag][y_index]) * result_score[i][j]
                    b = b + float(network[tag][y_index])
                obj_tag_b.append([tag, b])
    tag_score_net_b.append(obj_tag_b)


#reduce more add less score
score_b=0
for i in range(len(x_test)):
    count_b = 0
    if k-len(result[i])>0:
        tag_score_net_b[i].sort(key=(lambda x: x[1]), reverse=True)
        for m in range(len(result[i])):
            if result[i][m] in y_test[i]:
                count_b = count_b + 1
        for n in range(k-len(result[i])):
            if tag_score_net_b[i][n][0] in y_test[i]:
                count_b = count_b + 1
    else:
        tag_score_net_b[i].sort(key=(lambda x: x[1]), reverse=True)
        for n in range(k):
            if tag_score_net_b[i][n][0] in y_test[i]:
                count_b = count_b + 1
    score_b = score_b + count_b / len(y_test[i])
score_b=score_b/len(x_test)
print("network",score_b)