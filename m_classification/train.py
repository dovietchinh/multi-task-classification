import torch 
import yaml 
import pandas as pd
import argparse
def train(opt):
    df_train = pd.read_csv(opt.train_csv)
    df_val = pd.read_csv(opt.val_csv)
    device = 

def parse_opt(know=True):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='',help = 'weight path')
    parser.add_argument('--cfg',type=str,default='/u01/Intern/chinhdv/code/M_classification_torch/config/train_config.yaml')
    parser.add_argument('--data',type=str,default='/u01/Intern/chinhdv/code/M_classification_torch/config/data_config.yaml')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epoch', type=int, default=100)
    opt = parser.parse_known_args()[0] if know else parser.parse_arg()
    return opt 

if __name__ =='__main__': 
    opt = parse_opt(True)
    with open(opt.cfg) as f:
        cfg = yaml.safe_load(f)
    with open(opt.data) as f:
        data = yaml.safe_load(f)
    for k,v in cfg.items():
        setattr(opt,k,v)    
    for k,v in data.items():
        setattr(opt,k,v) 
    train(opt)
