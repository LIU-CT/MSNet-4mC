
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

import numpy as np


# ce loss with weights
class CE_Loss(nn.Module):

    ### init
    def __init__(self, num_class=2, ignore_index=None, classes=6, class_weights=None):
        super(CE_Loss, self).__init__()


        self.ignore_index = ignore_index
        self.num_class = num_class
        self.class_weights = class_weights.clone().detach().cuda()

        
    def forward(self, x, target, sign):
        
        # label vector and reverse label vector
        labelvec = torch.zeros_like(x).scatter(1, target.view(-1, 1), 1) # one-hot vector

        # computing probability (approximate) based on rectified distance
        p_pred = ( ( (labelvec * torch.exp(x)) ).sum(dim=1, keepdim=True)  ) / ( torch.exp(x).sum(dim=1, keepdim=True) ) 
        loss_ce = -torch.log(p_pred + 1e-08) 
        
        ## reweighting based on the class freq
        sign_vec = sign[:,0]
        weight_mask = self.class_weights[list(sign_vec.long())]
        # norm
        weight_mask = weight_mask/weight_mask.sum()
        weight_mask = weight_mask.view(loss_ce.shape[0], 1)
        
        # loss reweighting
        losses = (loss_ce * weight_mask).sum()
        #losses = loss_ce.mean()
        

        return losses
    




