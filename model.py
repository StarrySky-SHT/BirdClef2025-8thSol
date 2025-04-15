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
from transformers import EfficientNetForImageClassification
import torchaudio
import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd

from typing import Optional

from torch import Tensor
from torch.nn.modules.transformer import _get_activation_fn


class TransformerDecoderLayerOptimal(nn.Module):
    def __init__(self, d_model, nhead=8, dim_feedforward=2048, dropout=0.1, activation="relu",
                 layer_norm_eps=1e-5) -> None:
        super(TransformerDecoderLayerOptimal, self).__init__()
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = torch.nn.functional.relu
        super(TransformerDecoderLayerOptimal, self).__setstate__(state)

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None, tgt_is_causal: Optional[bool] = None,
                memory_is_causal: bool = False):
        tgt = tgt + self.dropout1(tgt)
        tgt = self.norm1(tgt)
        tgt2 = self.self_attn(tgt, memory, memory)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


@torch.jit.script
class GroupFC(object):
    def __init__(self, embed_len_decoder: int):
        self.embed_len_decoder = embed_len_decoder

    def __call__(self, h: torch.Tensor, duplicate_pooling: torch.Tensor, out_extrap: torch.Tensor):
        for i in range(h.shape[1]):
            h_i = h[:, i, :]
            if len(duplicate_pooling.shape) == 3:
                w_i = duplicate_pooling[i, :, :]
            else:
                w_i = duplicate_pooling
            out_extrap[:, i, :] = torch.matmul(h_i, w_i)


class MLDecoder(nn.Module):
    def __init__(self, num_classes, num_of_groups=-1, decoder_embedding=768,
                 initial_num_features=2048, zsl=0):
        super(MLDecoder, self).__init__()
        embed_len_decoder = 100 if num_of_groups < 0 else num_of_groups
        if embed_len_decoder > num_classes:
            embed_len_decoder = num_classes

        # switching to 768 initial embeddings
        decoder_embedding = 768 if decoder_embedding < 0 else decoder_embedding
        embed_standart = nn.Linear(initial_num_features, decoder_embedding)

        # non-learnable queries
        if not zsl:
            query_embed = nn.Embedding(embed_len_decoder, decoder_embedding)
            query_embed.requires_grad_(False)
        else:
            query_embed = None

        # decoder
        decoder_dropout = 0.1
        num_layers_decoder = 1
        dim_feedforward = 2048
        layer_decode = TransformerDecoderLayerOptimal(d_model=decoder_embedding,
                                                      dim_feedforward=dim_feedforward, dropout=decoder_dropout)
        self.decoder = nn.TransformerDecoder(layer_decode, num_layers=num_layers_decoder)
        self.decoder.embed_standart = embed_standart
        self.decoder.query_embed = query_embed
        self.zsl = zsl

        if self.zsl:
            if decoder_embedding != 300:
                self.wordvec_proj = nn.Linear(300, decoder_embedding)
            else:
                self.wordvec_proj = nn.Identity()
            self.decoder.duplicate_pooling = torch.nn.Parameter(torch.Tensor(decoder_embedding, 1))
            self.decoder.duplicate_pooling_bias = torch.nn.Parameter(torch.Tensor(1))
            self.decoder.duplicate_factor = 1
        else:
            # group fully-connected
            self.decoder.num_classes = num_classes
            self.decoder.duplicate_factor = int(num_classes / embed_len_decoder + 0.999)
            self.decoder.duplicate_pooling = torch.nn.Parameter(
                torch.Tensor(embed_len_decoder, decoder_embedding, self.decoder.duplicate_factor))
            self.decoder.duplicate_pooling_bias = torch.nn.Parameter(torch.Tensor(num_classes))
        torch.nn.init.xavier_normal_(self.decoder.duplicate_pooling)
        torch.nn.init.constant_(self.decoder.duplicate_pooling_bias, 0)
        self.decoder.group_fc = GroupFC(embed_len_decoder)
        self.train_wordvecs = None
        self.test_wordvecs = None

    def forward(self, x, le=False):  # label embedding
        if len(x.shape) == 4:  # [bs,2048,7,7]
            embedding_spatial = x.flatten(2).transpose(1, 2)
        else:  # [bs,2048,49]
            embedding_spatial = x.transpose(1, 2)
        embedding_spatial_786 = self.decoder.embed_standart(embedding_spatial)
        embedding_spatial_786 = torch.nn.functional.relu(embedding_spatial_786, inplace=True)

        bs = embedding_spatial_786.shape[0]
        if self.zsl:
            query_embed = torch.nn.functional.relu(self.wordvec_proj(self.decoder.query_embed))
        else:
            query_embed = self.decoder.query_embed.weight
        # tgt = query_embed.unsqueeze(1).repeat(1, bs, 1)
        tgt = query_embed.unsqueeze(1).expand(-1, bs, -1)  # no allocation of memory with expand
        h = self.decoder(tgt, embedding_spatial_786.transpose(0, 1))  # [embed_len_decoder, batch, 768]
        h = h.transpose(0, 1)

        out_extrap = torch.zeros(h.shape[0], h.shape[1], self.decoder.duplicate_factor, device=h.device, dtype=h.dtype)
        self.decoder.group_fc(h, self.decoder.duplicate_pooling, out_extrap)
        if not self.zsl:
            h_out = out_extrap.flatten(1)[:, :self.decoder.num_classes]
        else:
            h_out = out_extrap.flatten(1)
        h_out += self.decoder.duplicate_pooling_bias
        logits = h_out

        if not le:
            return logits
        else:
            return h, logits

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
        if 'effi' in model_name:
            self.backbone.global_pool = nn.Identity()
            self.backbone.classifier = nn.Identity()
        elif 'eca' in model_name:
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
            n_fft=CFG.n_fft,
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

    def forward(self, x,return_features = False):
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
        features = nn.AdaptiveAvgPool2d(1)(x).squeeze(3).squeeze(2) # B,C
        
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
        if return_features:
            output_dict = {
                "clipwise_output": clipwise_output,
                "framewise_output":segmentwise_output,
                "maxframewise_output":maxframewise_output,
                "features":features
            }
        else:
            output_dict = {
                "clipwise_output": clipwise_output,
                "framewise_output":segmentwise_output,
                "maxframewise_output":maxframewise_output
            }

        return output_dict
    

class BirdClefSEDAttModelSplit(BirdClefSEDAttModel):
    def __init__(self, split1,split2,model_name=CFG.model,num_classes=CFG.num_classes,  pretrained = CFG.pretrained,p=0.5,device='cuda'):
        super().__init__(model_name=model_name, num_classes = num_classes, pretrained = pretrained,p=0.5)
        self.att_block = AttBlockV2(self.backbone.num_features, num_classes, activation="sigmoid")
        self.split1 = torch.tensor(split1).to(device)
        self.split2 = torch.tensor(split2).to(device)

    def forward(self, x):
        output = super().forward(x)
        clipwise = output['clipwise_output']
        maxframewise = output['maxframewise_output']
        ouput_dict1 = {
            'clipwise_output': clipwise[:,self.split1],
            'maxframewise_output': maxframewise[:,self.split1],
        }
        ouput_dict2 = {
            'clipwise_output': clipwise[:,self.split2],
            'maxframewise_output': maxframewise[:,self.split2],
        }
        return ouput_dict1,ouput_dict2

class PowerToDB(torch.nn.Module):
    """
    A power spectrogram to decibel conversion layer. See birdset.datamodule.components.augmentations
    """

    def __init__(self, ref=1.0, amin=1e-10, top_db=80.0):
        super(PowerToDB, self).__init__()
        # Initialize parameters
        self.ref = ref
        self.amin = amin
        self.top_db = top_db

    def forward(self, S):
        # Convert S to a PyTorch tensor if it is not already
        S = torch.as_tensor(S, dtype=torch.float32)

        if self.amin <= 0:
            raise ValueError("amin must be strictly positive")

        if torch.is_complex(S):
            magnitude = S.abs()
        else:
            magnitude = S

        # Check if ref is a callable function or a scalar
        if callable(self.ref):
            ref_value = self.ref(magnitude)
        else:
            ref_value = torch.abs(torch.tensor(self.ref, dtype=S.dtype))

        # Compute the log spectrogram
        log_spec = 10.0 * torch.log10(
            torch.maximum(magnitude, torch.tensor(self.amin, device=magnitude.device))
        )
        log_spec -= 10.0 * torch.log10(
            torch.maximum(ref_value, torch.tensor(self.amin, device=magnitude.device))
        )

        # Apply top_db threshold if necessary
        if self.top_db is not None:
            if self.top_db < 0:
                raise ValueError("top_db must be non-negative")
            log_spec = torch.maximum(log_spec, log_spec.max() - self.top_db)

        return log_spec

class BirdSetSEDModel(nn.Module):
    def __init__(self, num_classes = CFG.num_classes):
        super().__init__()
        self.backbone = EfficientNetForImageClassification()
        self.powerToDB = PowerToDB()
        self.fc1 = nn.Linear(1280, 1280, bias=True)
        self.att_block = AttBlockV2(1280, num_classes, activation="sigmoid")
        self.bn0 = nn.BatchNorm2d(256)
        self.aud2mel = torchaudio.transforms.Spectrogram(
            n_fft=2048, hop_length=256, power=2.0
        )
        self.melscaled = torchaudio.transforms.MelScale(n_mels=256, n_stft=1025)
        self.normlizer = transforms.Normalize((-4.268,), (4.569,))
        self.SpecAug = SpecAugmentation(time_drop_width=64, time_stripes_num=2,freq_drop_width=8, freq_stripes_num=2)

        init_bn(self.bn0)
        # Resample to 32kHz

    def get_mel_gram(self,audios):
        spectrogram = self.aud2mel(audios)
        melspec = self.melscaled(spectrogram)
        dbscale = self.powerToDB(melspec)
        normalized_dbscale = self.normlizer(dbscale)
        return normalized_dbscale.permute(0,2,1)

    def forward(self,x):

        x = x.contiguous()
        # b c f t
        if CFG.use_spec_aug and self.training:
            if np.random.uniform(0,1)>CFG.p_spec_aug:
                x = self.SpecAug(x)

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        x = x.transpose(2, 3)

        x = self.backbone.efficientnet(x).last_hidden_state
        
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

class TeacherModel(nn.Module):
    def __init__(self,device='cuda'):
        super().__init__()
        self.model1 = BirdClefSEDAttModel(model_name='seresnext26d_32x4d',num_classes=CFG.num_classes,pretrained=True)
        self.model2 = BirdClefSEDAttModel(model_name='tf_efficientnetv2_b3',num_classes=CFG.num_classes,pretrained=True)
        self.model3 = BirdClefSEDAttModel(model_name='eca_nfnet_l0',num_classes=CFG.num_classes,pretrained=True)
        self.model1.load_state_dict(torch.load('/root/projects/BirdClef2025/BirdCLEF2023-30th-place-solution-master/logs/2025-03-30T03:24-seresnext/saved_model.pt'))
        self.model2.load_state_dict(torch.load('/root/projects/BirdClef2025/BirdCLEF2023-30th-place-solution-master/logs/2025-04-09T14:04-MLD-0.860/saved_model_lastepoch.pt'))
        self.model3.load_state_dict(torch.load('/root/projects/BirdClef2025/BirdCLEF2023-30th-place-solution-master/logs/2025-04-06T02:04-nfnetl0/saved_model.pt'))
        self.model1.eval()
        self.model2.eval()
        self.model3.eval()
        self.mode1 = self.model1.to(device)
        self.mode2 = self.model2.to(device)
        self.mode3 = self.model3.to(device)
        self.device = device


    @torch.no_grad
    def forward(self,x,return_features = False):

        pred1_dict = self.model1(x,return_features)
        pred2_dict = self.model2(x,return_features)
        pred3_dict = self.model3(x,return_features)
        pred1 = (pred1_dict['clipwise_output'] + pred1_dict['maxframewise_output'])/2 # B 206
        pred2 = (pred2_dict['clipwise_output'] + pred2_dict['maxframewise_output'])/2 # B 206
        pred3 = (pred3_dict['clipwise_output'] + pred3_dict['maxframewise_output'])/2 # B 206
        pred = (pred1+pred2+pred3) / 3
        if return_features:
            return pred,pred2_dict['features']
        else:
            return pred

if __name__ == '__main__':

    # bc_labels = pd.read_csv('/root/projects/BirdClef2025/bvc_model/assets/label.csv').iloc[:, 0].to_list()
    # bc_labels_indices = range(len(bc_labels))
    # ebird_name = pd.read_csv('/root/projects/BirdClef2025/data/train.csv')
    # scientific_name_common_name = list(ebird_name['scientific_name'] +'_'+ ebird_name['common_name'])
    # ebird_name_list = list(ebird_name['primary_label'])
    # ebird_birdclef2025 = list(pd.get_dummies(ebird_name['primary_label']).columns)

    # primary_labels_map = dict(zip(bc_labels, bc_labels_indices))
    # bvc_classes = [pl for pl in ebird_birdclef2025 if pl in bc_labels]

    # primary_labels_map = dict(zip(bc_labels, bc_labels_indices))
    # birdclassifier_last = len(bc_labels)
    # birdclassifier_indices = [primary_labels_map[i] for i in ebird_birdclef2025 if i in primary_labels_map]
    # birdclef2025_indices = [ebird_birdclef2025.index(i) for i in bvc_classes]

    # model = TeacherModel(bvc_index=birdclassifier_indices,birdclef_index=birdclef2025_indices,device='cuda')
    # model = model.cuda()
    # image = torch.randn(4, 3, 313, 192).cuda()
    # audio = torch.randn(4,10*32000).cuda()
    # pred = model(image,audio)
    # print(pred.shape)
    data = torch.randn((4,1536,7,15))
    model = MLDecoder(num_classes=206,initial_num_features=1536)
    out = model.forward(data,le=True)
    print(1)