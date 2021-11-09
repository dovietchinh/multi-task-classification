import torch 
import logging

LOGGER = logging.getLogger(__name__)

class EarlyStoping:

    def __init__(self, best_epoch=0,best_fitness=0,patience=30):
        self.best_epoch = best_epoch
        self.best_fitness = best_fitness
        self.patience = patience
    def __call__(self,epoch,fi):
        if fi >= self.best_fitness:
            self.best_epoch = epoch
            self.best_fitness = fi
        stop =  (epoch - self.best_epoch) >= self.patience
        if stop:
            LOGGER.info(f'EarlyStopping patience {self.patience} exceeded, stopping training')
        return stop

def loadingImageNetWeight(model,weight=None):
    return model



        

