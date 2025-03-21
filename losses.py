import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class SmoothBCEFocalLoss(nn.Module):
    def __init__(self,smoothing=0.) -> None:
        super().__init__()
        self.smoothing = smoothing

    def forward(self,input,targets,weights=None):
        # targets bs,numclasses
        targets[torch.where(targets==1)] = 1-self.smoothing
        input = torch.clamp(input,1e-3,0.999)
        # input_sigmoid = torch.sigmoid(input)
        if weights is None:
            weights = torch.ones_like(input).cuda()
        loss = -torch.mean(targets*torch.log(input)*weights*((1-input)**2) + 
                           (1-targets)*torch.log(1-input)*weights*(input**2),dim=(0,1))
        return loss

class MultiLoss(nn.Module):
    def __init__(self,smoothing=0.) -> None:
        super().__init__()
        self.smoothing = smoothing

    def forward(self,input,targets,weights=None):
        """
        pred : clipwise_pred and framewise_pred b num_class
        target: b num_class
        """
        # targets bs,numclasses
        targets[torch.where(targets==1)] = 1-self.smoothing
        input1 = torch.clamp(input['clipwise_output'],1e-3,0.999)
        input2 = torch.clamp(input['maxframewise_output'],1e-3,0.999)
        # input_sigmoid = torch.sigmoid(input)
        if weights is None:
            weights = torch.ones_like(input1).cuda()
        loss1 = -torch.mean(targets*torch.log(input1)*weights*((1-input1)**2) + (1-targets)*torch.log(1-input1)*weights*(input1**2),dim=(0,1))
        loss2 = -torch.mean(targets*torch.log(input2)*weights*((1-input2)**2) + (1-targets)*torch.log(1-input2)*weights*(input2**2),dim=(0,1))
        return loss1*0.666 + loss2 * 0.333

class MultiLossWeighting(nn.Module):
    def __init__(self,smoothing=0.) -> None:
        super().__init__()
        self.smoothing = smoothing

    def forward(self,input,targets,weights=None):
        """
        pred : clipwise_pred and framewise_pred b num_class
        target: b num_class
        """
        # targets bs,numclasses
        targets[torch.where(targets==1)] = 1-self.smoothing
        input1 = torch.clamp(input['clipwise_output'],1e-3,0.999)
        input2 = torch.clamp(input['maxframewise_output'],1e-3,0.999)
        # input_sigmoid = torch.sigmoid(input)
        if weights is None:
            weights = torch.ones_like(input1).cuda()
        loss1 = -torch.mean(targets*torch.log(input1)*((1-input1)**2) + (1-targets)*torch.log(1-input1)*(input1**2),dim=1)
        loss2 = -torch.mean(targets*torch.log(input2)*((1-input2)**2) + (1-targets)*torch.log(1-input2)*(input2**2),dim=1)
        loss1 = torch.mean(weights*loss1)
        loss2 = torch.mean(weights*loss2)
        return loss1*0.666 + loss2 * 0.333

class PANNsLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.bce = nn.BCELoss()

    def forward(self, input, target):
        input_ = input
        input_ = torch.where(torch.isnan(input_),
                             torch.zeros_like(input_),
                             input_)
        input_ = torch.where(torch.isinf(input_),
                             torch.zeros_like(input_),
                             input_)

        target = target.float()

        return self.bce(input_, target)
    
if __name__ == '__main__':
    loss = SmoothBCEFocalLoss(0.05)
    data = torch.randn((4,10))
    label = torch.ones((4,))
    res = loss.forward(data,label,weights=1)
    print(1)