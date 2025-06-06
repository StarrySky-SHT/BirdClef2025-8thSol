import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

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

class FocalLossBCE_logit(torch.nn.Module):
    def __init__(
            self,
            alpha: float = 0.25,
            gamma: float = 2,
            reduction: str = "mean",
            bce_weight: float = 1.0,
            focal_weight: float = 1.0,
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce = torch.nn.BCEWithLogitsLoss(reduction=reduction)
        self.bce_weight = bce_weight
        self.focal_weight = focal_weight

    def forward(self, logits, targets, weights=None):
        focall_loss = torchvision.ops.focal_loss.sigmoid_focal_loss(
            inputs=logits,
            targets=targets,
            alpha=self.alpha,
            gamma=self.gamma,
            reduction=self.reduction,
        )
        bce_loss = self.bce(logits, targets)
        return self.bce_weight * bce_loss + self.focal_weight * focall_loss

    
class BCELoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.loss = torch.nn.BCEWithLogitsLoss()
        # self.neg_weight = torch.from_numpy(np.load('/root/projects/BirdClef2025/BirdCLEF2023-30th-place-solution-master/usefulFunc/neg_weight.npy')).to(torch.float32).cuda().unsqueeze(0)
        # self.pos_weight = torch.from_numpy(np.load('/root/projects/BirdClef2025/BirdCLEF2023-30th-place-solution-master/usefulFunc/pos_weight.npy')).to(torch.float32).cuda().unsqueeze(0)
        self.neg_weight = 0.05
        self.pos_weight = 0.95

    def forward(self,input,targets,weights=None):
        # targets bs,numclasses
        input1 = torch.clamp(input['clipwise_output'],1e-3,0.999)
        input2 = torch.clamp(input['maxframewise_output'],1e-3,0.999)

        loss1 = -torch.mean(targets*torch.log(input1)*self.pos_weight + 
                           (1-targets)*torch.log(1-input1)*self.neg_weight,dim=(0,1))

        loss2 = -torch.mean(targets*torch.log(input2)*self.pos_weight + 
                           (1-targets)*torch.log(1-input2)*self.neg_weight,dim=(0,1))

        loss = loss1 + loss2
        return loss

class BCEDSLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.loss = torch.nn.BCEWithLogitsLoss()
        # self.neg_weight = torch.from_numpy(np.load('/root/projects/BirdClef2025/BirdCLEF2023-30th-place-solution-master/usefulFunc/neg_weight.npy')).to(torch.float32).cuda().unsqueeze(0)
        # self.pos_weight = torch.from_numpy(np.load('/root/projects/BirdClef2025/BirdCLEF2023-30th-place-solution-master/usefulFunc/pos_weight.npy')).to(torch.float32).cuda().unsqueeze(0)
        self.neg_weight = 0.05
        self.pos_weight = 0.95
        self.weight_list = [0.25,0.3,0.45]

    def forward(self,input:dict,targets,weights=None):
        # targets bs,numclasses
        loss = 0
        for idx,(key,value) in enumerate(input.items()):
            input1 = torch.clamp(value['clipwise_output'],1e-3,0.999)
            input2 = torch.clamp(value['maxframewise_output'],1e-3,0.999)

            loss1 = -torch.mean(targets*torch.log(input1)*self.pos_weight + 
                            (1-targets)*torch.log(1-input1)*self.neg_weight,dim=(0,1))

            loss2 = -torch.mean(targets*torch.log(input2)*self.pos_weight + 
                            (1-targets)*torch.log(1-input2)*self.neg_weight,dim=(0,1))

            loss += (loss1 + loss2) * self.weight_list[idx]
        return loss

class BCEMaskLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.loss = torch.nn.BCEWithLogitsLoss()

    def forward(self,input,targets,weights=None):
        # targets bs,numclasses
        loss_mask = (targets>=0).to(torch.float32)
        input1 = torch.clamp(input['clipwise_output'],1e-3,0.999)
        input2 = torch.clamp(input['maxframewise_output'],1e-3,0.999)

        loss1 = -targets*torch.log(input1)*0.95 - (1-targets)*torch.log(1-input1)*0.05
        loss2 = -targets*torch.log(input2)*0.95 - (1-targets)*torch.log(1-input2)*0.05

        loss1 = (loss_mask*loss1).sum()/(loss_mask.sum())
        loss2 = (loss_mask*loss2).sum()/(loss_mask.sum())

        loss = loss1 + loss2
        return loss

class MCCLoss(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def forward(self,y_pred,y_true,weight=None):
        """
        多标签分类的交叉熵（PyTorch 版本）

        参数说明：
        y_pred: Tensor，shape为 [batch_size, num_classes]，原始 logits（不加 sigmoid）
        y_true: Tensor，shape为 [batch_size, num_classes]，标签值为 0 或 1

        返回：
        Tensor，shape 为 [batch_size] 的损失值
        """
        y_pred = torch.logit(torch.clamp(y_pred,1e-3,0.999))
        # (1 - 2*y_true) * y_pred
        y_pred = (1 - 2 * y_true) * y_pred  # 正类变负，负类不变

        # 将正类对应的logits置为极小值（用于计算负类的loss）
        y_pred_neg = y_pred - y_true * 1e12
        # 将负类对应的logits置为极小值（用于计算正类的loss）
        y_pred_pos = y_pred - (1 - y_true) * 1e12

        # 添加一个0以避免logsumexp为空，保持数值稳定
        zeros = torch.zeros_like(y_pred[:, :1])
        y_pred_neg = torch.cat([y_pred_neg, zeros], dim=1)
        y_pred_pos = torch.cat([y_pred_pos, zeros], dim=1)

        # 计算logsumexp
        neg_loss = torch.logsumexp(y_pred_neg, dim=1)
        pos_loss = torch.logsumexp(y_pred_pos, dim=1)

        return neg_loss.mean() + pos_loss.mean()  # 每个样本一个loss
    
class BCEFrameDSLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.loss = torch.nn.BCEWithLogitsLoss()
        # self.neg_weight = torch.from_numpy(np.load('/root/projects/BirdClef2025/BirdCLEF2023-30th-place-solution-master/usefulFunc/neg_weight.npy')).to(torch.float32).cuda().unsqueeze(0).unsqueeze(-1)
        # self.pos_weight = torch.from_numpy(np.load('/root/projects/BirdClef2025/BirdCLEF2023-30th-place-solution-master/usefulFunc/pos_weight.npy')).to(torch.float32).cuda().unsqueeze(0).unsqueeze(-1)
        self.neg_weight = 0.05
        self.pos_weight = 0.95

    def forward(self,input1,input2,reverse=True):
        # targets bs,numclasses
        if reverse:
            input = input1[2]
            input = torch.clamp(input['framewise_output'],1e-3,0.999)

            loss = -torch.mean(input2*torch.log(input)*self.pos_weight + 
                            (1-input2)*torch.log(1-input)*self.neg_weight,dim=(0,1,2))
            return loss
        else:
            input = input2[2]
            input = torch.clamp(input['framewise_output'],1e-3,0.999)

            loss = -torch.mean(input1*torch.log(input)*self.pos_weight + 
                            (1-input1)*torch.log(1-input)*self.neg_weight,dim=(0,1,2))
            return loss

class BCEFrameLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.loss = torch.nn.BCEWithLogitsLoss()
        # self.neg_weight = torch.from_numpy(np.load('/root/projects/BirdClef2025/BirdCLEF2023-30th-place-solution-master/usefulFunc/neg_weight.npy')).to(torch.float32).cuda().unsqueeze(0).unsqueeze(-1)
        # self.pos_weight = torch.from_numpy(np.load('/root/projects/BirdClef2025/BirdCLEF2023-30th-place-solution-master/usefulFunc/pos_weight.npy')).to(torch.float32).cuda().unsqueeze(0).unsqueeze(-1)
        self.neg_weight = 0.05
        self.pos_weight = 0.95

    def forward(self,input,targets,weights=None):
        # targets bs,numclasses
        input = torch.clamp(input,1e-3,0.999) # b,c,frame

        loss = -torch.mean(targets*torch.log(input)*self.pos_weight + 
                           (1-targets)*torch.log(1-input)*self.neg_weight,dim=(0,1,2))
        return loss

class BCEFrameMaskLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.loss = torch.nn.BCEWithLogitsLoss()

    def forward(self,input,targets,weights=None):
        # targets bs,numclasses
        input = torch.clamp(input,1e-3,0.999) # b,c,frame

        loss_mask = (targets>=0).to(torch.float32)
        loss = -targets*torch.log(input)*0.95 - (1-targets)*torch.log(1-input)*0.05
        loss = (loss*loss_mask).sum()/(loss_mask.sum())
        return loss

class FocalBCELoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self,input,targets,weights=None):
        # targets bs,numclasses
        input1 = torch.clamp(input['clipwise_output'],1e-3,0.999)
        input2 = torch.clamp(input['maxframewise_output'],1e-3,0.999)

        loss1 = -torch.mean(targets*torch.log(input1) + 
                           (1-targets)*torch.log(1-input1),dim=(0,1))

        loss2 = -torch.mean(targets*torch.log(input2) + 
                           (1-targets)*torch.log(1-input2),dim=(0,1))

        loss_bce = loss1 + loss2

        loss1 = -torch.mean(targets*torch.log(input1)*((1-input1)**2)*0.75 + 
                           (1-targets)*torch.log(1-input1)*(input1**2)*0.25,dim=(0,1))
        loss2 = -torch.mean(targets*torch.log(input2)*((1-input2)**2)*0.75 + 
                           (1-targets)*torch.log(1-input2)*(input2**2)*0.25,dim=(0,1))
        loss_focal = loss1 + loss2
        loss = (loss_bce+loss_focal)/2
        return loss

class BCE_CE_logit_Loss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.loss_bce = torch.nn.BCEWithLogitsLoss()
        self.loss_ce = torch.nn.CrossEntropyLoss(reduction='mean')

    def forward(self,input,targets,weights=None):
        # targets bs,numclasses
        # input1 = torch.clamp(input['clipwise_output'],1e-3,0.999)
        # input2 = torch.clamp(input['maxframewise_output'],1e-3,0.999)
        input1 = input['clipwise_output']
        input2 = input['maxframewise_output']

        # for targets == 1, use BCE loss
        bce_index = torch.where(targets.sum(dim=-1) != 1)
        ce_index = torch.where(targets.sum(dim=-1) == 1)
        # for targets != 1, use CE loss
        input1_bce = input1[bce_index]
        input2_bce = input2[bce_index]
        targets_bce = targets[bce_index]
        input1_ce = input1[ce_index]
        input2_ce = input2[ce_index]
        targets_ce = targets[ce_index]
        # loss1_bce = -torch.mean(targets_bce*input1_bce + 
        #                    (1-targets_bce)*1-input1_bce)

        # loss2_bce = -torch.mean(targets_bce*input2_bce + 
        #                    (1-targets_bce)*(1-input2_bce))
        loss1_bce = self.loss_bce(input1_bce, targets_bce)
        loss2_bce = self.loss_bce(input2_bce, targets_bce)
        targets_ce = targets_ce.argmax(dim=-1)
        loss1_ce = self.loss_ce(input1_ce, targets_ce)
        loss2_ce = self.loss_ce(input2_ce, targets_ce)

        loss = loss1_bce + loss2_bce + loss1_ce + loss2_ce
        return loss

class BCE_CE_Loss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.loss_bce = torch.nn.BCEWithLogitsLoss()
        self.loss_ce = torch.nn.CrossEntropyLoss(reduction='mean')

    def forward(self,input,targets,weights=None):
        # targets bs,numclasses
        input1 = torch.clamp(input['clipwise_output'],1e-3,0.999)
        input2 = torch.clamp(input['maxframewise_output'],1e-3,0.999)

        # for targets == 1, use BCE loss
        bce_index = torch.where(targets.sum(dim=-1) != 1)
        ce_index = torch.where(targets.sum(dim=-1) == 1)
        # for targets != 1, use CE loss
        input1_bce = input1[bce_index]
        input2_bce = input2[bce_index]
        targets_bce = targets[bce_index]
        input1_ce = input1[ce_index]
        input2_ce = input2[ce_index]
        targets_ce = targets[ce_index]
        loss1_bce = -torch.mean(targets_bce*torch.log(input1_bce) + 
                           (1-targets_bce)*torch.log(1-input1_bce))

        loss2_bce = -torch.mean(targets_bce*torch.log(input2_bce) + 
                           (1-targets_bce)*torch.log(1-input2_bce))
        targets_ce = targets_ce.argmax(dim=-1)
        loss1_ce = self.loss_ce(torch.log(input1_ce), targets_ce)
        loss2_ce = self.loss_ce(torch.log(input2_ce), targets_ce)

        loss = loss1_bce + loss2_bce + loss1_ce + loss2_ce
        return loss

class BCECNNLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pos_weight = 0.95
        self.neg_weight = 0.05
        self.bceloss = nn.BCEWithLogitsLoss(pos_weight=torch.ones(206)*9).cuda()

    def forward(self,input,targets,weights=None):
        return self.bceloss(input,targets.to(torch.float32))

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

class KDLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.T = 20

    def forward(self,target,probs):
        input_ = probs
        KD_loss = nn.KLDivLoss()(
            torch.log_softmax(input_ / self.T, dim=1),
            target
            ) * (self.T * self.T)

        return KD_loss

class MLDDSLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.kl_loss = nn.KLDivLoss(reduction='none')
        self.weight_list = [0.25,0.3,0.45]
    
    def forward(self,input1,input2,reverse=True):
        loss_total = 0
        if reverse:
            for idx,(key,value) in enumerate(input1.items()):
                this_input1 = torch.clamp(value['clipwise_output'],1e-3,0.999)
                this_input2 = torch.clamp(value['maxframewise_output'],1e-3,0.999)
                probs = (this_input1 + this_input2) / 2
                loss = self.kl_loss(torch.log(probs),input2) + self.kl_loss(torch.log(1-probs),1-input2)
                loss_total += loss.mean() * self.weight_list[idx]
            return loss_total
        else:
            for idx,(key,value) in enumerate(input2.items()):
                this_input1 = torch.clamp(value['clipwise_output'],1e-3,0.999)
                this_input2 = torch.clamp(value['maxframewise_output'],1e-3,0.999)
                probs = (this_input1 + this_input2) / 2
                loss = self.kl_loss(torch.log(probs),input1) + self.kl_loss(torch.log(1-probs),1-input1)
                loss_total += loss.mean() * self.weight_list[idx]
            return loss_total

class MLDFrameDSLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.kl_loss = nn.KLDivLoss(reduction='none')
    
    def forward(self,input1,input2,reverse=True):
        if reverse:
            input1 = input1[2]
            probs = torch.clamp(input1,1e-3,0.999)
            target = torch.clamp(input2,1e-3,0.999)
            loss = self.kl_loss(torch.log(probs),target) + self.kl_loss(torch.log(1-probs),1-target)
            return loss
        else:
            input2 = input2[2]
            probs = torch.clamp(input2,1e-3,0.999)
            target = torch.clamp(input1,1e-3,0.999)
            loss = self.kl_loss(torch.log(probs),target) + self.kl_loss(torch.log(1-probs),1-target)
            return loss

class MLDLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.kl_loss = nn.KLDivLoss(reduction='none')
    
    def forward(self,target,probs):
        probs = torch.clamp(probs,1e-3,0.999)
        target = torch.clamp(target,1e-3,0.999)
        loss = self.kl_loss(torch.log(probs),target) + self.kl_loss(torch.log(1-probs),1-target)
        return loss.mean()

class MLDLogitLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.kl_loss = nn.KLDivLoss(reduction='none')
    
    def forward(self,target,probs):
        loss = self.kl_loss(torch.log_softmax(probs),target) + self.kl_loss(torch.log_softmax(1-probs),1-target)
        return loss.mean()

import torch
import torch.nn.functional as F

def custom_info_nce_loss(features:torch.Tensor, labels:torch.Tensor, temperature=0.07):
    """
    features: Tensor of shape [B, D] - 特征向量
    labels: Tensor of shape [B, C] - 多标签 one-hot 编码，值为 0 或 1
    """
    B = features.size(0)
    device = features.device

    # 1. Normalize features for cosine similarity
    features = F.normalize(features, dim=1)  # [B, D]

    # 2. 计算 pairwise cosine similarity
    sim_matrix = torch.matmul(features, features.T)  # [B, B]
    sim_matrix = sim_matrix / temperature

    # 3. 构造 label 相似矩阵
    label_eq = (labels @ labels.T)  # [B, B] - 计算标签交集个数
    label_sum = (labels.sum(dim=1, keepdim=True) == labels.sum(dim=1))  # [B, B] - 标签总数是否相同
    same_labels = (label_eq == labels.sum(dim=1, keepdim=True)) & label_sum  # 完全一致
    disjoint = (label_eq == 0).to(torch.float32).to(device)  # 没有交集

    same_labels = same_labels * (1-torch.eye(B).to(device))

    pos_loss = torch.exp(same_labels * sim_matrix).sum() / (same_labels.sum() + 1e-6)
    neg_loss = torch.exp(disjoint * sim_matrix).sum() / (disjoint.sum() + 1e-6)
    return -torch.log(pos_loss / (neg_loss + pos_loss + 1e-8))

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
    # pred1 = (torch.randn((4,1024)))
    # pred2 = torch.randn((4,1024))
    # pred = dict(
    #     {'clipwise_output':pred1,
    #      'maxframewise_output':pred2}
    # )
    # target = torch.zeros_like(pred1)
    # target[0,0] = 1
    # target[1,0] = 1
    # target[1,10] = 1
    # target[2,100] = 1
    # target[2, 500] = 1
    # target[3,28] = 1
    # loss = MCCLoss()
    features = torch.randn((4,128))
    labels = torch.zeros((4,4))
    labels[0,0] = 1
    labels[0,2] = 1
    labels[1,3] = 1
    labels[2,0] = 1
    labels[2,1] = 1
    labels[3,3] = 1
    loss = custom_info_nce_loss(features,labels)
