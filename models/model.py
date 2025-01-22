import os

import torch
import torch.nn as nn

class BaseModel(nn.Module):

    def __init__(self):
        super(BaseModel, self).__init__()

    @staticmethod
    def save_pretrained(path, model):
        save = os.path.join(path, "/model/")
        os.makedirs(save, exist_ok=True)
        torch.save(model.state_dict(), save)
