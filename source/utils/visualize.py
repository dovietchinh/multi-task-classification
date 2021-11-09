import matplotlib.pyplot as plt
import pandas as pd
import numyp as np
from sklearn.metrics import confusion_matrix
def visualize(csv,classes):
    if isinstance(csv,str):
        csv = pd.read_csv(csv)
    for label_name,class_name in classes.items():
        len_ = len(class_name)
        ax = plt.axes        


def confusion_matrix(out1,out2):

    