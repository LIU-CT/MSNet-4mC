"""
@author: liu chunting
Department of IST, Kyoto University
"""
from torch import nn
from torch.nn.parameter import Parameter
import torch
import math

# ver1
class modeltest(nn.Module):
    def __init__(self, load_pretrain = False, load_path = None):
        super(modeltest, self).__init__()
        self.expand = 4
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.sig = nn.Sigmoid()
        
        self.conv_op = nn.Conv1d(in_channels=4, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_op = nn.BatchNorm1d(16)
        self.nl = nn.GELU()
        # block 1
        self.conv1_1 = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv1_2 = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1_2p = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=3, stride=1, dilation=2, padding=2, bias=False)
        self.conv1_2w = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=3, stride=1, dilation=4, padding=4, bias=False)
        self.conv1_3 = nn.Conv1d(in_channels=48, out_channels=96, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1_1 = nn.BatchNorm1d(16)
        self.bn1_2 = nn.BatchNorm1d(16)
        self.bn1_2p = nn.BatchNorm1d(16)
        self.bn1_2w = nn.BatchNorm1d(16)
        self.bn1_3 = nn.BatchNorm1d(96)
        
        # ch gate 1
        self.gamma_b1_1 = Parameter(torch.ones(1,96,1))
        self.beta_b1_1 = Parameter(torch.ones(1,96,1))
        self.mlp_b1 = nn.Conv1d(in_channels=96, out_channels=96, kernel_size=1, stride=1, padding=0, bias=False)
        self.gamma_b1_2 = Parameter(torch.ones(1,96,1))
        self.beta_b1_2 = Parameter(torch.ones(1,96,1))

        self.shortcut_b1 = nn.Sequential(
            nn.Conv1d(16, 96, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm1d(96)
            )
        
        # block 2
        self.conv2_1 = nn.Conv1d(in_channels=96, out_channels=32, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv2_2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2_2p = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=1, dilation=2, padding=2, bias=False)
        self.conv2_2w = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=1, dilation=4, padding=4, bias=False)
        self.conv2_3 = nn.Conv1d(in_channels=96, out_channels=192, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2_1 = nn.BatchNorm1d(32)
        self.bn2_2 = nn.BatchNorm1d(32)
        self.bn2_2p = nn.BatchNorm1d(32)
        self.bn2_2w = nn.BatchNorm1d(32)
        self.bn2_3 = nn.BatchNorm1d(192)

        self.shortcut_b2 = nn.Sequential(
            nn.Conv1d(96, 192, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm1d(192)
            )
        
        # ch gate 2
        self.gamma_b2_1 = Parameter(torch.ones(1,192,1))
        self.beta_b2_1 = Parameter(torch.ones(1,192,1))
        self.mlp_b2 = nn.Conv1d(in_channels=192, out_channels=192, kernel_size=1, stride=1, padding=0, bias=False)
        self.gamma_b2_2 = Parameter(torch.ones(1,192,1))
        self.beta_b2_2 = Parameter(torch.ones(1,192,1))


        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(192*41, 192)
        self.linear2 = nn.Linear(192, 2)
        
        self.drop_ss = nn.Dropout(0)#default:0.2
        self.drop_s = nn.Dropout(0.25)#default:0.8
        self.drop = nn.Dropout(0.5)#default:0.8

        # activate pretraining
        if load_pretrain:
            self._load_pretrained_model(load_path)           
    
    def forward(self, x):
        eps = 1e-12
        
        # openning conv
        x = self.conv_op(x)
        x = self.bn_op(x)
        x = self.nl(x)
        
        # identity mapping
        residual = x
        
        # block1
        x = self.conv1_1(x)
        x = self.bn1_1(x)
        x = self.nl(x)  
        x = self.drop_ss(x)
        
        # multi-scale 
        xp1 = self.conv1_2(x)
        xp1 = self.bn1_2(xp1)
        
        xp2 = self.conv1_2p(x)
        xp2 = self.bn1_2p(xp2)
        
        xp3 = self.conv1_2w(x)
        xp3 = self.bn1_2w(xp3)
        
        x = torch.cat([xp1, xp2, xp3], dim=1)
        
        x = self.nl(x)  
        x = self.drop_ss(x)

        x = self.conv1_3(x)
        x = self.bn1_3(x)
        
        # gating b1
        avg = self.avgpool(x)
        avg = (avg - avg.mean(dim=1, keepdim=True)) / (avg.std(dim=1, keepdim=True) + eps)
        avg = avg * self.gamma_b1_1 + self.beta_b1_1
        avg = self.mlp_b1(avg)
        avg = (avg - avg.mean(dim=1, keepdim=True)) / (avg.std(dim=1, keepdim=True) + eps)
        avg = avg * self.gamma_b1_2 + self.beta_b1_2
        avg = self.sig(avg)
        
        #
        x = x * avg + self.shortcut_b1(residual)
        
        x = self.nl(x)  
        x = self.drop_ss(x)
        
        
        # block2
        residual = x
        
        x = self.conv2_1(x)
        x = self.bn2_1(x)
        x = self.nl(x)    
        x = self.drop_s(x)


        # multi-scale 
        xp1 = self.conv2_2(x)
        xp1 = self.bn2_2(xp1)
        
        xp2 = self.conv2_2p(x)
        xp2 = self.bn2_2p(xp2)
        
        xp3 = self.conv2_2w(x)
        xp3 = self.bn2_2w(xp3)
        
        x = torch.cat([xp1, xp2, xp3], dim=1)

        x = self.nl(x)  
        x = self.drop_s(x)

        x = self.conv2_3(x)
        x = self.bn2_3(x)
        
        
        # gating b1
        avg = self.avgpool(x)
        avg = (avg - avg.mean(dim=1, keepdim=True)) / (avg.std(dim=1, keepdim=True) + eps)
        avg = avg * self.gamma_b2_1 + self.beta_b2_1
        avg = self.mlp_b2(avg)
        avg = (avg - avg.mean(dim=1, keepdim=True)) / (avg.std(dim=1, keepdim=True) + eps)
        avg = avg * self.gamma_b2_2 + self.beta_b2_2
        avg = self.sig(avg)
        
        #
        x = x * avg + self.shortcut_b2(residual)
        
        x = self.nl(x) 
        x = self.drop(x)
        

        # mlp
        x = x.view(x.shape[0], -1)
        x = self.linear1(x) 
        #x = self.drop(x)
        x = self.linear2(x)
        
        return x
    
    def _load_pretrained_model(self, load_path):
        
        pretrain_dict = torch.load(load_path)
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)
        # sign
        print('Model_best loaded')
########

