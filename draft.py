import os
import torch
from source.models.mobilenetv2 import MobileNetV2
import cv2
import numpy as np
import json
import shutil
def preprocess(img,img_size,padding=True):
    if padding:
        height,width,_ = img.shape 
        delta = height - width 
        
        if delta > 0:
            img = np.pad(img,[[0,0],[delta//2,delta//2],[0,0]], mode='constant',constant_values =255)
        else:
            img = np.pad(img,[[-delta//2,-delta//2],[0,0],[0,0]], mode='constant',constant_values =255)
    if isinstance(img_size,int):
        img_size = (img_size,img_size)
    return cv2.resize(img,img_size)
		
# folder = '/u01/Intern/Freelance_DaNang/res'

model = MobileNetV2(76)
model.load_state_dict(torch.load('result/runs_m/last.pt')['state_dict'])
model.eval()
index = 0
with open('1103.json') as f:
	labels= json.load(f)
for root,dirs,files in os.walk(folder):
	for file in files:
		path = os.path.join(root,file)
		img = cv2.imread(path)
		img = preprocess(img,224,True)
		img = np.transpose(img,[2,0,1])
		img = img.astype('float')/255.
		img = np.expand_dims(img,axis=0)
		img = torch.Tensor(img)
		out = model.predict(img)
		out = torch.argmax(out,axis=-1)
		name = labels[str(out[0].item())][-1]
		name = name.replace('/','_')
		index +=1
		new_name = os.path.join('result3',name,'{:04d}.png'.format(index))
		os.makedirs(os.path.dirname(new_name), exist_ok=True)
		shutil.copy(path,new_name)
		print(index,end='\r')



  