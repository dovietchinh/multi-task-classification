import cv2
import os 
import numpy as np
import matplotlib.pyplot as plt
root = '/u01/Intern/chinhdv/github/yolov5_0509/runs/MILK2/crops/Sua'

def main():
    images = os.listdir(root)
    # with open('size.txt','a') as f:
    #     f.write(f'height width\n')
    for image in images:
        path = os.path.join(root,image)
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        height,width,_ = img.shape
        ratio = height/width
        with open('size2.txt','a') as f:
            f.write(f'{height} {ratio}\n')
def plot():
    with open('size2.txt') as f:
        lines = f.readlines()
    ratios = []
    heights = []
    for line in lines:
        height,ratio = line.strip().split()
        height = int(height)
        ratio = float(ratio)
        ratios.append(ratio)
        heights.append(height)
    # bins = np.linspace(min(heights),max(heights),int((max(heights)-min(heights))/30))
    bins = np.linspace(50,300,10)
    plt.hist(heights,bins)
    plt.show()

def create_csv():
    pass

if __name__ =='__main__':
    # main()
    plot()