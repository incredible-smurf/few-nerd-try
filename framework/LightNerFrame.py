import torch
import numpy as np

class LightNerFrame:
    def __init__(self,model,device,args) -> None:
        self.model = model
        self.device = device
        self.args=args

    
    def train(self,train_dataloader,eval_dataloader=None,optimizer=None,epoch=100):
        pass