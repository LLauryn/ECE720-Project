
# Extract 50,000 questioins(body+title)+tags
import xml.etree.ElementTree as ET
import csv

tree=ET.parse('data/posts.xml')
csvfile = open('data/traindata1.csv', 'w', newline='')
writer = csv.writer(csvfile)
keys = ['Tags','Content']
writer.writerow(keys)
Content=[]
Tags=[]
root = tree.getroot()
count=0
for i in root.iter('row'):
    d = i.get('PostTypeId')
    if d == '1':
        count=count+1
        a = i.get('Body')
        b = i.get('Title')
        c = i.get('Tags')
        writer.writerow([c,a+b])
    if count==50000:
        break