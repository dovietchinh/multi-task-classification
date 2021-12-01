from models.mobilenetv2 import MobileNetV2
from utils.dataset import LoadImagesAndLabels, preprocess
import torch
import cv2
import numpy as np

def inference():
    pass

if __name__ =='__main__':
    model = MobileNetV2(model_config = [2])
    ckp = torch.load('/u01/Intern/chinhdv/code/multi-task-classification/result/runs_elevator_4/last.pt')
    model.load_state_dict(ckp['state_dict'])
    img = cv2.imread('/u01/Intern/chinhdv/code/multi-task-classification/test9.jpg')
    device = 'cuda:0'
    img = cv2.resize(img,(224,224))
    img = np.transpose(img,(2,0,1))
    img = img[None]
    img = img.astype('float')/255.
    img = torch.Tensor(img).to(device)
    model = model.to(device)
    model.eval()
    output = model.predict(img)
    print(output)