import csv
import pandas as pd
import nltk
from nltk.tokenize import RegexpTokenizer
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.probability import FreqDist
import re

#Preprocess data
tdata=pd.read_csv('data/traindata1.csv')
file=open('data/ProcessedData.csv','w',encoding='utf-8')
csv_write=csv.writer(file)
csv_head=["Tags","Content"]
csv_write.writerow(csv_head)

stopword=set(stopwords.words('english'))
stopword.add('p')
stopword.add('n')
tokenizer = RegexpTokenizer(r'\w+')
for i in range(len(tdata)):
    tdata["Content"][i]=tokenizer.tokenize(tdata["Content"][i])
    t = nltk.pos_tag(tdata["Content"][i])
    nouns = [word for (word, pos) in t if pos in ['NN','NNP','NNS','NNPS']]
    item=[]
    for j in range(len(nouns)):
        if nouns[j] not in stopword:
            item.append(nouns[j])
    csv_write.writerow([tdata["Tags"][i],item])

# filtering tags and terms
tdata = pd.read_csv('data/ProcessedData1.csv', encoding='ISO-8859-1')
tokenizer = RegexpTokenizer(r'\w+')
for i in range(len(tdata)):
    tdata["Content"][i] = tokenizer.tokenize(tdata["Content"][i])

file = open('data/content1.csv', 'w', encoding='utf-8')
csv_write = csv.writer(file)
csv_head = ["Content"]
csv_write.writerow(csv_head)
f = FreqDist(item)
l = []
for i in range(len(tdata["Content"])):
    elem = []
    for j in range(len(tdata["Content"][i])):
        if f[tdata["Content"][i][j]] >= 20:
            elem.append(tdata["Content"][i][j])
    l.append(elem)
    csv_write.writerow([elem])

file = open('data/tags.csv', 'w', encoding='utf-8')
csv_write = csv.writer(file)
csv_head = ["Tags"]
csv_write.writerow(csv_head)
f = FreqDist(item)
l = []
for i in range(len(tdata["Tags"])):
    elem = []
    for j in range(len(tdata["Tags"][i])):
        if f[tdata["Tags"][i][j]] >= 50:
            elem.append(tdata["Tags"][i][j])
    l.append(elem)
    csv_write.writerow([elem])

#finding tag_list
item=[]
for i in range(len(tdata["Tags"])):
    for j in range(len(tdata["Tags"][i])):
        item.append(tdata["Tags"][i][j])
file=open('data/tags_list.csv','w',encoding='utf-8')
csv_write=csv.writer(file)
csv_head=["Tags"]
csv_write.writerow(csv_head)
m=[]
for i in range(len(item)):
    if f[item[i]]>=50:
        if item[i] not in m:
            m.append(item[i])
            csv_write.writerow([item[i]])