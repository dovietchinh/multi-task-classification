import os 
import pandas as pd 
import shutil
from tqdm import tqdm
df1 = pd.read_csv('/u01/DATA/HUMAN_ATTRIBUTE/PA_100K/annotation/csv/train.csv')
df2 = pd.read_csv('/u01/DATA/HUMAN_ATTRIBUTE/PA_100K/annotation/csv/test.csv')
df3 = pd.read_csv('/u01/DATA/HUMAN_ATTRIBUTE/PA_100K/annotation/csv/val.csv')
df = pd.concat([df1,df2,df3], axis=0)
# print(df.keys())
keys = [
    "Female",
    "AgeOver60",
    "Age18-60",
    "AgeLess18",
    "Front",
    "Side",
    "Back",
    "Hat",
    "Glasses",
    "HandBag",
    "ShoulderBag",
    "Backpack",
    "HoldObjectsInFront",
    "ShortSleeve",
    "LongSleeve",
    "UpperStride",
    "UpperLogo",
    "UpperPlaid",
    "UpperSplice",
    "LowerStripe",
    "LowerPattern",
    "LongCoat",
    "Trousers",
    "Shorts",
    "Skirt&Dress",
    "boots"
]
df = df.sample(frac=1).reset_index(drop=True)
# for key in tqdm(keys):
#     df_ = df[df[key]==0].sample(n=500, replace=True)
#     os.makedirs(f'test_PA100K_0/{key}', exist_ok=True)
#     for i in df_.iloc:
#         old_path = os.path.join('/u01/DATA/HUMAN_ATTRIBUTE/PA_100K/data', i.path)
#         new_path  = (f'test_PA100K_0/{key}/{i.path}') 
#         # print(new_path,old_path)
#         shutil.copy(old_path,new_path)

# df_ = df[df['ShoulderBag']==1]
# df_ = df_[df_['HandBag']==1]
# print(len(df_))
# os.makedirs('testasd', exist_ok=True)
# for i in df_.iloc:
#     old_path = os.path.join('/u01/DATA/HUMAN_ATTRIBUTE/PA_100K/data', i.path)
#     new_path = os.path.join('testasd',i.path)
#     shutil.copy(old_path,new_path)
import cv2
heights = []
widths = []
sizes = []
for i in tqdm(df.iloc, total=df.shape[0]):
    path = os.path.join('/u01/DATA/HUMAN_ATTRIBUTE/PA_100K/data', i.path)
    # print(path)
    height,width,_ = cv2.imread(path, cv2.IMREAD_COLOR).shape
    size = height + width
    heights.append(height)
    widths.append(width)
    sizes.append(size)
df['width'] = widths
df['height'] = heights 
df['size'] = sizes
df= df.sort_values(by=['size'],ascending=False)#,key= (lambda x,y: x+y ))
df.to_csv('PA100K.csv')
