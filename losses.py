import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def pdist(e, squared=False, eps=1e-12):
    e_square = e.pow(2).sum(dim=1)
    prod = e @ e.t()
    res = (e_square.unsqueeze(1) + e_square.unsqueeze(0) - 2 * prod).clamp(min=eps)

    if not squared:
        res = res.sqrt()

    res = res.clone()
    res[range(len(e)), range(len(e))] = 0
    return res

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

class MLDLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.kl_div_loss = torch.nn.KLDivLoss(reduction="none")

    def forward(self,prob_teacher,prob_student):
        loss = self.kl_div_loss(torch.log(prob_student), prob_teacher) + self.kl_div_loss(torch.log(1 - prob_student), 1 - prob_teacher)
        loss = loss.mean()
        return loss
    
class BCELoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.loss = torch.nn.BCEWithLogitsLoss()

    def forward(self,input,targets,weights=None):
        # targets bs,numclasses
        input1 = torch.clamp(input['clipwise_output'],1e-3,0.999)
        input2 = torch.clamp(input['maxframewise_output'],1e-3,0.999)

        # targets_mask = (targets != 0.5) 

        loss1 = -torch.mean(targets*torch.log(input1) + 
                           (1-targets)*torch.log(1-input1),dim=1)

        loss2 = -torch.mean(targets*torch.log(input2) + 
                           (1-targets)*torch.log(1-input2),dim=1)
        if weights.sum() == 0:
            return 0.001*(loss1 + loss2)
        loss1 = torch.mean(loss1[torch.where(weights)])
        loss2 = torch.mean(loss2[torch.where(weights)])

        # loss1 = -(targets*torch.log(input1) + 
        #                    (1-targets)*torch.log(1-input1))

        # loss2 = -(targets*torch.log(input2) + 
        #                    (1-targets)*torch.log(1-input2))
        
        # loss1 = torch.sum(loss1*targets_mask)/torch.sum(targets_mask)
        # loss2 = torch.sum(loss2*targets_mask)/torch.sum(targets_mask)

        loss = loss1 + loss2
        return loss

class Splitloss(nn.Module):
    def __init__(self,split1,split2,device='cuda',focal_weight = 2) -> None:
        super().__init__()
        self.split1 = torch.tensor(split1).to(device)
        self.split2 = torch.tensor(split2).to(device)
        self.focal_weight = focal_weight
        self.alpha = 0.25


    def forward(self,input,targets,weights=None):
        # targets bs,numclasses
        input1,input2 = input[0],input[1]
        split1input1 = torch.clamp(input1['clipwise_output'],1e-3,0.999)
        split1input2 = torch.clamp(input1['maxframewise_output'],1e-3,0.999)

        split2input1 = torch.clamp(input2['clipwise_output'],1e-3,0.999)
        split2input2 = torch.clamp(input2['maxframewise_output'],1e-3,0.999)

        split1targets = targets[:,self.split1]
        split2targets = targets[:,self.split2]

        split1loss1 = -torch.mean(split1targets*torch.log(split1input1) + 
                           (1-split1targets)*torch.log(1-split1input1),dim=(0,1))

        split1loss2 = -torch.mean(split1targets*torch.log(split1input2) + 
                           (1-split1targets)*torch.log(1-split1input2),dim=(0,1))


        split2loss1 = -torch.mean(split2targets*torch.log(split2input1)*((1-split2input1)**2) + 
                           (1-split2targets)*torch.log(1-split2input1)*(split2input1**2)*self.alpha,dim=(0,1))

        split2loss2 = -torch.mean(split2targets*torch.log(split2input2)*((1-split2input2)**2) + 
                           (1-split2targets)*torch.log(1-split2input2)*(split2input2**2)*self.alpha,dim=(0,1))
        loss = (split1loss1+split1loss2+self.focal_weight*split2loss1+self.focal_weight*split2loss2)/4
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

def phi(e:torch.Tensor, eps=1e-12):
    e_square = e.pow(2).sum(dim=1)
    prod = e @ e.t()
    res = (e_square.unsqueeze(1) + e_square.unsqueeze(0) - 2 * prod).clamp(min=eps).sqrt()
    res = res.clone()
    res[range(len(e)), range(len(e))] = 0
    return res


class HuberDist(nn.Module):
    @staticmethod
    def forward(student, teacher):
        N, C = student.shape

        with torch.no_grad():
            t_d = phi(teacher)
            mean_td = t_d[t_d > 0].mean()
            t_d = t_d / mean_td

        d = phi(student)
        mean_d = d[d > 0].mean()
        d = d / mean_d

        loss = F.smooth_l1_loss(d, t_d, reduction="sum") / N
        return loss

class ID(nn.Module):  # instance-aware label-wise embedding distillation
    def __init__(self):
        super().__init__()
        self.le_distill_criterion = HuberDist()

    def forward(self, le_student, le_teacher, targets):
        N, C, le_length = le_student.shape
        le_mask = targets.unsqueeze(2).repeat(1, 1, le_length)
        le_student_pos = le_student * le_mask
        le_teacher_pos = le_teacher * le_mask
        n_pos_per_instance = targets.sum(dim=1)
        loss = 0.0

        for i in range(N):
            if n_pos_per_instance[i] > 1:
                le_s_i = le_student_pos[i, :, :]
                le_t_i = le_teacher_pos[i, :, :]
                le_s_pos_i = le_s_i[~(le_s_i == 0).all(1)]
                le_t_pos_i = le_t_i[~(le_t_i == 0).all(1)]
                delta_loss = self.le_distill_criterion(le_s_pos_i, le_t_pos_i)
                loss += delta_loss

        return loss

class CD(nn.Module):  # class-aware label-wise embedding distillation
    def __init__(self):
        super().__init__()
        self.le_distill_criterion = HuberDist()

    def forward(self, le_student:torch.Tensor, le_teacher:torch.Tensor, targets:torch.Tensor):
        N, C, le_length = le_student.shape
        le_mask = targets.unsqueeze(2).repeat(1, 1, le_length)
        le_student_pos = le_student * le_mask
        le_teacher_pos = le_teacher * le_mask
        n_pos_per_label = targets.sum(dim=0)
        loss = 0.0

        for c in range(C):
            if n_pos_per_label[c] > 1:
                le_s_c = le_student_pos[:, c, :]
                le_t_c = le_teacher_pos[:, c, :]
                le_s_pos_c = le_s_c[~(le_s_c == 0).all(1)]
                le_t_pos_c = le_t_c[~(le_t_c == 0).all(1)]
                delta_loss = self.le_distill_criterion(le_s_pos_c, le_t_pos_c)
                loss += delta_loss

        return loss

class LEDLoss(nn.Module):
    def __init__(self, lambda_cd=100.0, lambda_id=1000.0):
        super().__init__()
        self.cd_distiller = CD()
        self.id_distiller = ID()
        self.lambda_cd = lambda_cd
        self.lambda_id = lambda_id

    def forward(self, le_student, le_teacher, targets):
        loss_cd = self.cd_distiller(le_student, le_teacher, targets)
        loss_id = self.id_distiller(le_student, le_teacher, targets)
        loss = self.lambda_cd * loss_cd + self.lambda_id * loss_id
        return loss

class MultiLossWeighting(nn.Module):
    def __init__(self,smoothing=0.,alpha=2) -> None:
        super().__init__()
        self.smoothing = smoothing
        self.alpha = alpha

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
        loss1 = -torch.mean(targets*torch.log(input1)*((1-input1)**self.alpha) + (1-targets)*torch.log(1-input1)*(input1**self.alpha),dim=1)
        loss2 = -torch.mean(targets*torch.log(input2)*((1-input2)**self.alpha) + (1-targets)*torch.log(1-input2)*(input2**self.alpha),dim=1)
        loss1 = torch.mean(weights*loss1)
        loss2 = torch.mean(weights*loss2)
        return loss1*0.5 + loss2 * 0.5

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

class RKdAngle(nn.Module):
    def forward(self, student, teacher):
        # N x C
        # N x N x C

        with torch.no_grad():
            td = (teacher.unsqueeze(0) - teacher.unsqueeze(1))
            norm_td = F.normalize(td, p=2, dim=2)
            t_angle = torch.bmm(norm_td, norm_td.transpose(1, 2)).view(-1)

        sd = (student.unsqueeze(0) - student.unsqueeze(1))
        norm_sd = F.normalize(sd, p=2, dim=2)
        s_angle = torch.bmm(norm_sd, norm_sd.transpose(1, 2)).view(-1)

        loss = F.smooth_l1_loss(s_angle, t_angle, reduction='elementwise_mean')
        return loss


class RkdDistance(nn.Module):
    def forward(self, student, teacher):
        with torch.no_grad():
            t_d = pdist(teacher, squared=False)
            mean_td = t_d[t_d>0].mean()
            t_d = t_d / mean_td

        d = pdist(student, squared=False)
        mean_d = d[d>0].mean()
        d = d / mean_d

        loss = F.smooth_l1_loss(d, t_d, reduction='elementwise_mean')
        return loss

if __name__ == '__main__':
    # loss = BCELoss()
    # data = {
    #     'clipwise_output':torch.randn((4,8)).cuda(),
    #     'maxframewise_output':torch.randn((4,8)).cuda(),
    # }
    # label = torch.ones((4,8)).cuda()
    # label[3,4] = 0.5
    # res = loss.forward(data,label,weights=1)
    # print(1)
    pred1 = torch.sigmoid(torch.randn((4,1024)))
    pred2 = torch.sigmoid(torch.randn((4,1024)))
    target = torch.ones_like(pred1)
    loss = RKdAngle()
    out = loss.forward(pred1,pred2)