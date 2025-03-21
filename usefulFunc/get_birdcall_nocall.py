import numpy as np
from glob import glob
import sys
sys.path.append('/home/lijw/BirdCLEF/BirdCLEF-Baselinev2/')

import torch
import torch.nn as nn
import timm
from config import CFG
import pandas as pd 
import librosa as lb
from torchlibrosa import SpecAugmentation
import numpy as np
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2
import soundfile as sf

data_transform = A.Compose([
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])

df_train = pd.read_csv(CFG.train_path)
df_valid = pd.read_csv(CFG.valid_path)

df_train = pd.concat([df_train, pd.get_dummies(df_train['primary_label'])], axis=1)
df_valid = pd.concat([df_valid, pd.get_dummies(df_valid['primary_label'])], axis=1)

df = pd.concat((df_train,df_valid))

class CustomResNext(nn.Module):
    def __init__(self, model_name='resnext50_32x4d', pretrained=True):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        n_features = self.model.fc.in_features
        self.model.fc = nn.Linear(n_features, 2)

    def forward(self, x):
        x = self.model(x)
        return x
    
# preprocess
def compute_melspec(y, sr, n_mels, fmin, fmax):
    """
    Computes a mel-spectrogram and puts it at decibel scale
    Arguments:
        y {np array} -- signal
        params {AudioParams} -- Parameters to use for the spectrogram. Expected to have the attributes sr, n_mels, f_min, f_max
    Returns:
        np array -- Mel-spectrogram
    """
    melspec = lb.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_mels, fmin=fmin, fmax=fmax,
        win_length=CFG.window_size,hop_length=CFG.hop_size,center=True,
        n_fft=CFG.window_size,pad_mode='reflect',window='hann'
    )

    melspec = lb.power_to_db(melspec,amin=1e-10,ref=1.0,top_db=None).astype(np.float32)
    return melspec

def mono_to_color(X, eps=1e-6, mean=None, std=None):
    mean = mean or X.mean()
    std = std or X.std()
    X = (X - mean) / (std + eps)
    
    _min, _max = X.min(), X.max()

    if (_max - _min) > eps:
        V = np.clip(X, _min, _max)
        V = 255 * (V - _min) / (_max - _min)
        V = V.astype(np.uint8)
    else:
        V = np.zeros_like(X, dtype=np.uint8)

    return V

def crop_or_pad(y, length, is_train=True, start=None):
    if len(y) < length:
        y = np.concatenate([y, np.zeros(length - len(y))])
        
        n_repeats = length // len(y)
        epsilon = length % len(y)
        
        y = np.concatenate([y]*n_repeats + [y[:epsilon]])
        
    elif len(y) > length:
        if not is_train:
            start = start or 0
        else:
            start = start or np.random.randint(len(y) - length)

        y = y[start:start + length]

    return y
# dataset
import torch
import numpy as np
from torch.utils.data import Dataset,DataLoader

class BirdDataset(torch.utils.data.Dataset):
    def __init__(self, data, sr=CFG.sample_rate, n_mels=128, fmin=0, fmax=None, duration=CFG.duration, step=None, res_type="kaiser_fast", resample=True):
        
        self.data = data
        
        self.sr = sr
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax or self.sr//2

        self.duration = duration
        self.audio_length = self.duration*self.sr
        self.step = step or self.audio_length
        
        self.res_type = res_type
        self.resample = resample

    def __len__(self):
        return len(self.data)
    
    
    def audio_to_image(self, audio):
        image = compute_melspec(audio, self.sr, self.n_mels, self.fmin, self.fmax) 
        image = mono_to_color(image)
        image = np.stack((image,image,image),axis=-1)
        image = data_transform(image=image)['image']
        return image

    def read_file(self, filepath):
        audio, orig_sr = sf.read(filepath, dtype="float32")

        if self.resample and orig_sr != self.sr:
            audio = lb.resample(audio, orig_sr, self.sr, res_type=self.res_type)
          
        audios = []
        for i in range(self.audio_length, len(audio) + self.step, self.step):
            start = max(0, i - self.audio_length)
            end = start + self.audio_length
            audios.append(audio[start:end])
            
        if len(audios[-1]) < self.audio_length:
            audios[-1] = crop_or_pad(audios[-1],length=self.audio_length)
            
        images = [self.audio_to_image(audio) for audio in audios]
        images = torch.stack(images)
        
        return images
    
    def __getitem__(self, idx):
        return self.read_file(self.data.loc[idx, "path"])

model = CustomResNext()
model.load_state_dict(torch.load('/home/lijw/BirdCLEF/BirdCLEF-Indentify-Call/logsresnext50_32x4d_best.pth')['model'])
ds_test = BirdDataset(df_train,sr = CFG.sample_rate,duration = CFG.duration)
# eval
model.to('cpu')
model.eval()    
predictions = []
for en in range(len(ds_test)):
    images = ds_test[en]
    with torch.no_grad():
        outputs = model(images).sigmoid().detach().cpu().numpy()
    print(outputs)
    predictions.append(outputs)