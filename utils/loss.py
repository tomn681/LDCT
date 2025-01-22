import torch
import torch.nn as nn

''' 
variational_loss function:
    Inputs:
        - input_img:    (torch.Tensor)  Image
        - pred_sample:  (VAE DecoderOutput) Predicted Output with KL loss
        
    Output:
        - loss: () Variational Loss
'''
class VariationalLoss():

    def __init__(self, device):
        self.loss = nn.BCEWithLogitsLoss().to(device)

    def __call__(self, input_img, pred_sample):
        x_hat = pred_sample.sample
        KLD = pred_sample.commit_loss

        reproduction_loss = self.loss(x_hat, input_img)
        
        return reproduction_loss + KLD.mean()
