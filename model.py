import timm
from config import CFG  
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlibrosa import SpecAugmentation
import numpy as np
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation
from torchaudio.transforms import MelSpectrogram,AmplitudeToDB
from torchvision import transforms
from utils import random_power

class NormalizeMelSpec(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, X):
        mean = X.mean((1, 2), keepdim=True)
        std = X.std((1, 2), keepdim=True)
        Xstd = (X - mean) / (std + self.eps)
        norm_min, norm_max = Xstd.min(-1)[0].min(-1)[0], Xstd.max(-1)[0].max(-1)[0]
        fix_ind = (norm_max - norm_min) > self.eps * torch.ones_like(
            (norm_max - norm_min)
        )
        V = torch.zeros_like(Xstd)
        if fix_ind.sum():
            V_fix = Xstd[fix_ind]
            norm_max_fix = norm_max[fix_ind, None, None]
            norm_min_fix = norm_min[fix_ind, None, None]
            V_fix = torch.max(
                torch.min(V_fix, norm_max_fix),
                norm_min_fix,
            )
            # print(V_fix.shape, norm_min_fix.shape, norm_max_fix.shape)
            V_fix = (V_fix - norm_min_fix) / (norm_max_fix - norm_min_fix)
            V[fix_ind] = V_fix
        return V

def gem_freq(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), 1)).pow(1.0 / p)

class GeMFreq(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super().__init__()
        self.p = torch.nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return gem_freq(x, p=self.p, eps=self.eps)

def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """
    nn.init.xavier_uniform_(layer.weight)
 
    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)
            
def init_bn(bn):
    """Initialize a Batchnorm layer. """
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)

def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)

class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6, p_trainable=False):
        super(GeM,self).__init__()
        if p_trainable:
            self.p = nn.Parameter(torch.ones(1)*p)
        else:
            self.p = p
        self.eps = eps

    def forward(self, x):
        ret = gem(x, p=self.p, eps=self.eps)   
        ret = torch.flatten(ret,start_dim=1)
        return ret
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'

class BirdClefModel(nn.Module):
    def __init__(self, model_name=CFG.model, num_classes = CFG.num_classes, pretrained = CFG.pretrained,p=0.5):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = timm.create_model(model_name, pretrained=pretrained)
        if 'effi' in CFG.model:
            self.backbone.global_pool = nn.Identity()
            self.backbone.classifier = nn.Identity()
        elif 'eca' in CFG.model:
            self.backbone.head.fc = nn.Identity()
        if CFG.use_fsr:
            self.backbone.conv_stem.stride = (1,1)
        self.fc_audioset = nn.Linear(self.backbone.num_features, num_classes, bias=True)
        self.pooling = GeM()
        self.SpecAug = SpecAugmentation(time_drop_width=64, time_stripes_num=2,freq_drop_width=8, freq_stripes_num=2)
        self.use_spec_aug = CFG.use_spec_aug
        self.bn0 = nn.BatchNorm2d(CFG.mel_bins)
        
        # Spectrogram extractor
        self.spectrogram_extractor = MelSpectrogram(
            sample_rate=CFG.sample_rate,
            n_fft=2048,
            win_length=CFG.window_size,
            hop_length=CFG.hop_size,
            f_min=CFG.fmin,
            f_max=CFG.fmax,
            pad=0,
            n_mels=CFG.mel_bins,
            power=2,
            normalized=False,
        )
        # Logmel feature extractor
        self.logmel_extractor = AmplitudeToDB(top_db=None)
        self.normlize = NormalizeMelSpec()
        self.infer_period = CFG.infer_duration
        self.train_period = CFG.duration
        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)

        
    def get_mel_gram(self,audios):
        """
        Input: (batch_size, data_length)"""
        x = self.spectrogram_extractor(audios) # (batch_size,freq_bins time_steps)
        x = self.logmel_extractor(x) 
        x = self.normlize(x)
        x = x.permute(0,2,1)# (batch_size,time_steps, mel_bins)
        x = random_power(x,power=3,c=0.5)
        return x
    
    def forward(self,images):
        # b c f t
        if CFG.use_spec_aug and self.training:
            if np.random.uniform(0,1)>CFG.p_spec_aug:
                images = self.SpecAug(images)
        x = self.backbone.forward_features(images) #  4*bs,1,t,f

        x = self.pooling(x)
        x = self.fc_audioset(x)
        return x

class BirdClefSEDModel(nn.Module):
    def __init__(self, model_name=CFG.model, num_classes = CFG.num_classes, pretrained = CFG.pretrained,p=0.5):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = timm.create_model(model_name, pretrained=pretrained)
        if 'effi' in CFG.model:
            self.backbone.global_pool = nn.Identity()
            self.backbone.classifier = nn.Identity()
        elif 'eca' in CFG.model:
            self.backbone.head.fc = nn.Identity()
        if CFG.use_fsr:
            self.backbone.conv_stem.stride = (1,1)
        self.fc_audioset = nn.Linear(self.backbone.num_features, num_classes, bias=True)
        self.pooling = GeM()
        self.SpecAug = SpecAugmentation(time_drop_width=64, time_stripes_num=2,freq_drop_width=8, freq_stripes_num=2)
        self.use_spec_aug = CFG.use_spec_aug
        self.bn0 = nn.BatchNorm2d(CFG.mel_bins)
    
        # Spectrogram extractor
        self.spectrogram_extractor = MelSpectrogram(
            sample_rate=CFG.sample_rate,
            n_fft=2048,
            win_length=CFG.window_size,
            hop_length=CFG.hop_size,
            f_min=CFG.fmin,
            f_max=CFG.fmax,
            pad=0,
            n_mels=CFG.mel_bins,
            power=2,
            normalized=False,
        )
        # Logmel feature extractor
        self.logmel_extractor = AmplitudeToDB(top_db=None)
        self.normlize = NormalizeMelSpec()
        self.infer_period = CFG.infer_duration
        self.train_period = CFG.duration
        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)

        
    def get_mel_gram(self,audios):
        """
        Input: (batch_size, data_length)"""
        x = self.spectrogram_extractor(audios) # (batch_size,freq_bins time_steps)
        x = self.logmel_extractor(x) 
        x = self.normlize(x)
        x = x.permute(0,2,1)# (batch_size,time_steps, mel_bins)
        return x
    
    def forward(self,images):
        # b c f t
        if CFG.use_spec_aug and self.training:
            if np.random.uniform(0,1)>CFG.p_spec_aug:
                images = self.SpecAug(images)
        x = self.backbone.forward_features(images) #  bs,1,t,f
        x = torch.mean(x,dim=3) # pooling freq bs,c,t
        
        (x1,_) = torch.max(x,dim=2) # bs,c
        x2 = torch.mean(x,dim=2) # bs,c
        x = x1+x2
        x = F.dropout(x,p=0.5,training=self.training)
        x = self.fc_audioset(x)

        return x

def init_layer(layer):
    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, "bias"):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)


def init_bn(bn):
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.0)


def init_weights(model):
    classname = model.__class__.__name__
    if classname.find("Conv2d") != -1:
        nn.init.xavier_uniform_(model.weight, gain=np.sqrt(2))
        model.bias.data.fill_(0)
    elif classname.find("BatchNorm") != -1:
        model.weight.data.normal_(1.0, 0.02)
        model.bias.data.fill_(0)
    elif classname.find("GRU") != -1:
        for weight in model.parameters():
            if len(weight.size()) > 1:
                nn.init.orghogonal_(weight.data)
    elif classname.find("Linear") != -1:
        model.weight.data.normal_(0, 0.01)
        model.bias.data.zero_()


def interpolate(x: torch.Tensor, ratio: int):
    """Interpolate data in time domain. This is used to compensate the
    resolution reduction in downsampling of a CNN.
    Args:
      x: (batch_size, time_steps, classes_num)
      ratio: int, ratio to interpolate
    Returns:
      upsampled: (batch_size, time_steps * ratio, classes_num)
    """
    (batch_size, time_steps, classes_num) = x.shape
    upsampled = x[:, :, None, :].repeat(1, 1, ratio, 1)
    upsampled = upsampled.reshape(batch_size, time_steps * ratio, classes_num)
    return upsampled


def pad_framewise_output(framewise_output: torch.Tensor, frames_num: int):
    """Pad framewise_output to the same length as input frames. The pad value
    is the same as the value of the last frame.
    Args:
      framewise_output: (batch_size, frames_num, classes_num)
      frames_num: int, number of frames to pad
    Outputs:
      output: (batch_size, frames_num, classes_num)
    """
    output = F.interpolate(
        framewise_output.unsqueeze(1),
        size=(frames_num, framewise_output.size(2)),
        align_corners=True,
        mode="bilinear").squeeze(1)

    return output

class AttBlockV2(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 activation="linear"):
        super().__init__()

        self.activation = activation
        self.att = nn.Conv1d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True)
        self.cla = nn.Conv1d(
            in_channels=in_features,
            out_channels=out_features, 
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True)

        self.init_weights()

    def init_weights(self):
        init_layer(self.att)
        init_layer(self.cla)

    def forward(self, x):
        # x: (n_samples, n_in, n_time)
        norm_att = torch.softmax(torch.tanh(self.att(x)), dim=-1)
        cla = self.nonlinear_transform(self.cla(x))
        x = torch.sum(norm_att * cla, dim=2)
        return x, norm_att, cla

    def nonlinear_transform(self, x):
        if self.activation == 'linear':
            return x
        elif self.activation == 'sigmoid':
            return torch.sigmoid(x)


class BirdClefSEDAttModel(nn.Module):
    def __init__(self, model_name=CFG.model, num_classes = CFG.num_classes, pretrained = CFG.pretrained,p=0.5):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = timm.create_model(model_name, pretrained=pretrained)
        if 'effi' in CFG.model:
            self.backbone.global_pool = nn.Identity()
            self.backbone.classifier = nn.Identity()
        elif 'eca' in CFG.model:
            self.backbone.head.fc = nn.Identity()
        if CFG.use_fsr:
            self.backbone.conv_stem.stride = (1,1)
        self.pooling = GeM()
        self.SpecAug = SpecAugmentation(time_drop_width=64, time_stripes_num=2,freq_drop_width=8, freq_stripes_num=2)
        self.use_spec_aug = CFG.use_spec_aug
        self.bn0 = nn.BatchNorm2d(CFG.mel_bins)
    
        # Spectrogram extractor
        self.spectrogram_extractor = MelSpectrogram(
            sample_rate=CFG.sample_rate,
            n_fft=2048,
            win_length=CFG.window_size,
            hop_length=CFG.hop_size,
            f_min=CFG.fmin,
            f_max=CFG.fmax,
            pad=0,
            n_mels=CFG.mel_bins,
            power=2,
            normalized=False,
        )
        # Logmel feature extractor
        self.logmel_extractor = AmplitudeToDB(top_db=None)
        self.normlize = NormalizeMelSpec()

        self.fc1 = nn.Linear(self.backbone.num_features, self.backbone.num_features, bias=True)
        self.att_block = AttBlockV2(self.backbone.num_features, num_classes, activation="sigmoid")

        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc1)
    
    def get_mel_gram(self,audios):
        """
        Input: (batch_size, data_length)"""
        x = self.spectrogram_extractor(audios) # (batch_size,freq_bins time_steps)
        x = self.logmel_extractor(x) 
        x = self.normlize(x)
        x = x.permute(0,2,1)# (batch_size,time_steps, mel_bins)

        return x

    def forward(self, x):

        x = x.contiguous()
        # b c f t
        if CFG.use_spec_aug and self.training:
            if np.random.uniform(0,1)>CFG.p_spec_aug:
                x = self.SpecAug(x)

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        x = x.transpose(2, 3)

        x = self.backbone.forward_features(x)
        
        # Aggregate in frequency axis
        x = torch.mean(x, dim=2)

        x1 = F.max_pool1d(x, kernel_size=3, stride=1, padding=1)
        x2 = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        x = x1 + x2

        x = F.dropout(x, p=0.3, training=self.training)
        x = x.transpose(1, 2)
        x = F.relu(self.fc1(x),inplace=False)
        x = x.transpose(1, 2)
        x = F.dropout(x, p=0.3, training=self.training)

        (clipwise_output, norm_att, segmentwise_output) = self.att_block(x)

        maxframewise_output = nn.AdaptiveMaxPool1d(1)(segmentwise_output).squeeze(2)
        # maxframewise_output = nn.AdaptiveAvgPool1d(1)(segmentwise_output).squeeze(2)

        output_dict = {
            "clipwise_output": clipwise_output,
            "framewise_output":segmentwise_output,
            "maxframewise_output":maxframewise_output
        }

        return output_dict