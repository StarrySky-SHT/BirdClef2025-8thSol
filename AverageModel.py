from email.mime import audio
import torch
import pandas as pd
import numpy as np
from glob import glob
import logging
import random
import os
import sys
sys.path.append('/home/lijw/BirdCLEF/BirdCLEF-Baselinev2')
from config import CFG  
import albumentations as A
from dataset import BirdDataset,fetch_scheduler,AudioAug
from torch.utils.data import Dataset,DataLoader
from torch import optim
from model import BirdClefModel,BirdClefSEDModel,BirdClefSEDAttModel
from torch.cuda import amp  
import time
from tensorboardX import SummaryWriter
from torch import nn
from metrics import padded_cmap,map_score
import sklearn
from torchtoolbox.tools import mixup
from torch.nn.functional import binary_cross_entropy_with_logits,binary_cross_entropy
from utils import *
from losses import SmoothBCEFocalLoss,MultiLoss
import audiomentations as AA
from torchtoolbox.tools import cutmix_data,MixingDataController
import torch.nn.functional as F
from utils import update_bn

def set_seed(seed = 42):
    '''Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY.'''
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    print('> SEEDING DONE')
set_seed(42)

df_train = pd.read_csv(CFG.train_path)
df_valid = pd.read_csv(CFG.valid_path)

df_train = pd.concat((df_train,df_valid))

df_train = pd.concat([df_train, pd.get_dummies(df_train['primary_label'])], axis=1)
df_valid = pd.concat([df_valid, pd.get_dummies(df_valid['primary_label'])], axis=1)

birds = list(df_train.primary_label.unique())
missing_birds = list(set(list(df_train.primary_label.unique())).difference(list(df_valid.primary_label.unique())))
non_missing_birds = list(set(list(df_train.primary_label.unique())).difference(missing_birds))
df_valid[missing_birds] = 0
df_valid = df_valid[df_train.columns] ## Fix order

df_train = upsample_data(df_train,thr=70)
df_train = downsample_data(df_train,thr=400)

wav_transform = Compose(
                [
                    OneOf(
                        [
                            NoiseInjection(p=1, max_noise_level=0.04),
                            GaussianNoise(p=1, min_snr=5, max_snr=20),
                            PinkNoise(p=1, min_snr=5, max_snr=20)
                        ],
                        p=0.5,
                    ),
                    AA.Shift(p=0.2),
                    RandomVolume(p=0.2),
                    # PitchShift(p=0.2),
                    AddBackgroundNoise('/home/data/lijw/dataset/BirdCLEF-2021/train_soundscapes_nocall',min_snr_in_db=0,max_snr_in_db=3,p=0.25),
                    AddBackgroundNoise('/home/data/lijw/dataset/f1010bird/ff1010bird_wav/wav_nocall',min_snr_in_db=0,max_snr_in_db=3,p=0.5),
                ]
            )
train_set = BirdDataset(df_train,sr = CFG.sample_rate,duration = CFG.duration,train = True,audio_augmentations=wav_transform)
val_set = BirdDataset(df_valid,sr = CFG.sample_rate,duration = CFG.infer_duration,train = False)
trainloader = DataLoader(train_set,batch_size=CFG.batch_size,shuffle=True,num_workers=16,pin_memory=True)
valloader = DataLoader(val_set,batch_size=CFG.batch_size,shuffle=False,num_workers=16,pin_memory=True)


toloadmodel = glob('/home/lijw/BirdCLEF/BirdCLEF-Baselinev2/logs/2023-05-20T00:23/*.pt')
toloadmodel = toloadmodel[0:2]
model = BirdClefSEDAttModel(CFG.model,num_classes=CFG.num_classes,pretrained=CFG.pretrained).to(torch.float64)
for name,para in model.named_parameters():
    para.data = torch.zeros_like(para.data)
models = []
for i in toloadmodel:
    models.append(torch.load(i,map_location=torch.device('cpu')))
model_dict = model.state_dict()

for key,value in model_dict.items():
    for trained_model in models:
        model_dict[key] += trained_model[key]

for key,value in model_dict.items():
    model_dict[key] = (value/len(models)).to(torch.float32)

model.load_state_dict(model_dict)
model.cuda()
update_bn(trainloader,model,model)
torch.save(model_dict,'/home/lijw/BirdCLEF/BirdCLEF-Baselinev2/logs/2023-05-20T00:23/avg_model.pt')
