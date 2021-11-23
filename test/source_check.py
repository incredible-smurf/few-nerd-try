#this file is used to check data shape
#need to be removed after check
import os 
data_root = './data/few_ner/'
inter_path = 'inter'

my_map=[]
path = os.path.join(data_root,inter_path)
for file in os.listdir(path):
    with open(os.path.join(path,file),encoding='UTF-8') as f :
        line = f.readline()
        while line:
            data =line.split('	')
            
            if(len(data)>1 and data[1] not in my_map):
                my_map.append(data[1])
            line=f.readline()
for i in my_map:
    tt= i.split('/')
    print(tt)
            
        

