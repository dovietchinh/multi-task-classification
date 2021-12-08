from models.mobilenetv2 import MobileNetV2
from utils.dataset_elevator import LoadImagesAndLabels, preprocess
import torch
import cv2
import numpy as np
import os
def inference():
    pass

if __name__ =='__main__':
    model = MobileNetV2(model_config = [2])
    ckp = torch.load('/u01/Intern/chinhdv/code/multi-task-classification/result/runs_elevator_5/best.pt')
    model.load_state_dict(ckp['state_dict'])
    device = 'cuda:0'
    model = model.to(device)
    model.eval()
    for i in os.listdir('/u01/Intern/chinhdv/code/multi-task-classification/mount_point/truck_classification/test_img'):
        path = os.path.join('/u01/Intern/chinhdv/code/multi-task-classification/mount_point/truck_classification/test_img',i)
        # img = cv2.imread('/u01/Intern/chinhdv/code/multi-task-classification/1_test7.jpg')
        img_raw = cv2.imread(path)
        
        img = cv2.resize(img_raw,(224,224))
        img = np.transpose(img,(2,0,1))
        img = img[None]
        img = img.astype('float')/255.
        img = torch.Tensor(img).to(device)

        output = model.predict(img)
        output = output[0].detach().cpu()
        score = torch.max(output).numpy()
        class_index = torch.argmax(output).numpy()
        
        print(score,class_index)
        cv2.imshow('a',img_raw)
        k = cv2.waitKey(0)
        if k ==ord('q'):
            exit()