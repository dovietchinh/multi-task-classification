import matplotlib.pyplot as plt
import os 
import pandas as pd
def main():
    df = pd.read_csv('76classes_1103.csv')
    
    df['label'].value_counts().plot(kind='bar')
    plt.show()
    # label = (df.label.values)
    # print(label.shape)

if __name__ =='__main__':
    main()