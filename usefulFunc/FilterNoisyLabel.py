import os

import torch.utils
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import sys
sys.path.append('/root/projects/BirdClef2025/BirdCLEF2023-30th-place-solution-master/')
from email.mime import audio
import torch
import pandas as pd
import numpy as np
from glob import glob
import logging
import random
import os
import sys
from config import CFG  
import albumentations as A
from dataset import BirdDataset,fetch_scheduler,AudioAug,BirdDatasetTwoLabel,BirdDatasetSplitTrain
from torch.utils.data import Dataset,DataLoader,WeightedRandomSampler
from torch import optim
from model import BirdClefSEDAttModel
from torch.cuda import amp  
import time
from tensorboardX import SummaryWriter
from torch import nn
from metrics import padded_cmap,map_score
import sklearn
from torchtoolbox.tools import mixup
from torch.nn.functional import binary_cross_entropy_with_logits,binary_cross_entropy
from utils import *
from losses import SmoothBCEFocalLoss,MultiLoss,MultiLossWeighting,BCELoss
import  audiomentations as AA
from torchtoolbox.tools import cutmix_data,MixingDataController
import torch.nn.functional as F
import torchvision.transforms as A

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

df_valid = pd.read_csv(CFG.train_path)
df_valid['path'] = CFG.data_root+df_valid['filename']
birds = list(pd.get_dummies(df_valid['primary_label']).columns)
df_valid = pd.concat([df_valid, pd.get_dummies(df_valid['primary_label'])], axis=1)
val_set = BirdDatasetTwoLabel(df_valid,bird_cols=birds,sr = CFG.sample_rate,duration = 10,train = False)
valloader = DataLoader(val_set,batch_size=CFG.batch_size,shuffle=False,num_workers=16,pin_memory=True)

# model
model = BirdClefSEDAttModel('tf_efficientnetv2_b3',num_classes=CFG.num_classes,pretrained=CFG.pretrained).cuda()
model.load_state_dict(torch.load('/root/projects/BirdClef2025/BirdCLEF2023-30th-place-solution-master/logs/2025-04-21T09:17-efv2b3-0.858/saved_model_lastepoch.pt'))

model.eval()
predall_tensor = torch.zeros(0).to(CFG.device)
gtall_tensor = torch.zeros(0).to(CFG.device)
for idx,(audios,labels,weights) in enumerate(valloader):
    audios = audios.to(torch.float32).to(CFG.device)
    labels = labels.to(torch.long).to(CFG.device)

    with torch.no_grad():
        images = model.get_mel_gram(audios)
        images = torch.repeat_interleave(images.unsqueeze(1),repeats=3,dim=1)
        pred = model.forward(images)
        pred = (pred['clipwise_output'] + pred['maxframewise_output'])/2
    predall_tensor = torch.cat((predall_tensor,pred),dim=0)
    gtall_tensor = torch.cat((gtall_tensor,labels),dim=0)

predall_numpy = predall_tensor.cpu().detach().numpy()
gtall_numpy = gtall_tensor.cpu().detach().numpy()

pred_df = pd.DataFrame(predall_numpy,columns=birds)
pred_df['path'] = df_valid['path']
gt_df = pd.DataFrame(gtall_numpy,columns=birds)
gt_df['path'] = df_valid['path']

pred_df.to_csv('/root/projects/BirdClef2025/BirdCLEF2023-30th-place-solution-master/usefulFunc/pred_df.csv')
gt_df.to_csv('/root/projects/BirdClef2025/BirdCLEF2023-30th-place-solution-master/usefulFunc/gt_df.csv')