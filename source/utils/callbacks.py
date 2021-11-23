import torch
from torch.utils.tensorboard import SummaryWriter
import os
class CallBack:
    """
    Handles all registers callbacks for model.
    Many features in developing progress. please update at https://github.com/dovietchinh/multi-task-classification.
    """

    # _callbacks = {
    #     'on_pretrain_routine_start' : [],
    #     'on_pretrain_routine_end' : [],

    #     'on_training_start' : [],
    #     'on_training_end' : [],
    #     'on_epoch_start' : [],
    #     'on_epoch_end' : [],
    #     'on_bestfit_epoch_end': [],
    #     'on_model_save' : [],
    # }

    def __init__(self,save_dir):

        self.writer_train = SummaryWriter(os.path.join(save_dir,'tensorboard_log','train'))
        self.writer_val = SummaryWriter(os.path.join(save_dir,'tensorboard_log','train'))
        os.makedirs(os.path.join(save_dir,'tensorboard_log'), exist_ok=True)   

    def __call__(self,loss_train,loss_val,epoch):
        self.writer_train.add_scalar('train',loss_train,epoch)
        self.writer_val.add_scalar('val',loss_val,epoch)

        

