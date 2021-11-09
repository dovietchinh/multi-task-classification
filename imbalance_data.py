import os 
import pandas as pd

def main():
    df = pd.read_csv('/u01/Intern/chinhdv/code/M_classification_torch/76classes_1103.csv')
    len_ = []
    for i in range(80):
        len_.append(len(df[df.label==i]))
    print(max(len_))

def balance_data(csv=None,image_per_epoch=200):
    if isinstance(csv,str):
        csv = pd.read_csv(csv)
    labels = set(csv.label)
    # print(label)
    dfs = []
    for label in labels:
        df = csv[csv.label==label].sample(n=image_per_epoch,replace=True)
        dfs.append(df)
    df = pd.concat(dfs,axis=0)
    for i in labels:
        print(len(df[df.label==i]))
    print((df))
    # df = pd.DataFrame({'path':paths,'label':labels})
    # df_choices = df.sample(n=200, replace=True)


if __name__ =='__main__':
    balance_data('/u01/Intern/chinhdv/code/M_classification_torch/76classes_1103.csv')