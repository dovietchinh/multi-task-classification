import pandas as pd
import numpy as np
import json 
import os
paths = []
label = []
dict_ = json.load(open('1103.json','r'))
for i in dict_:
    label_index = int(i)
    list_folder =dict_[i]
    if len(list_folder)>2:
        list_folder = list_folder[:-1]
    elif len(list_folder)==2:
        list_folder = list_folder[:-1]
    # print(list_folder)
    for folder in list_folder:
        files = os.listdir(os.path.join('/u01/Intern/chinhdv/DATA/milk/all_crop_data_origin',folder))
        files = [os.path.join(folder,x) for x in files]
        paths += files
        label += [label_index]*len(files)
df = pd.DataFrame({'path':paths, 'label':label})
df = df.sample(frac=1).reset_index(drop=True)
len_ = len(df)
print(len_)
df_train = df[:12000]
df_val = df[12000:]
df.to_csv('76classes_1103.csv')
df_train.to_csv('76classes_train_1103.csv')
df_val.to_csv('76classes_val_1103.csv')