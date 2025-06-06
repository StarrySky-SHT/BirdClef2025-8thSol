import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ['CUDA_VISIBLE_DEVICES'] = "2"
import sys
sys.path.append('/root/projects/BirdClef2025/BirdCLEF2023-30th-place-solution-master/')

from email.mime import audio
import torch
import pandas as pd
import numpy as np
from glob import glob
import logging
import random
from config import CFG  
import albumentations as A
from dataset import BirdDataset,fetch_scheduler,AudioAug
from torch.utils.data import Dataset,DataLoader
from torch import optim
from model import BirdClefCNNModel,BirdClefSEDAttModel,BirdClefCNNFCModel,BirdClefCNNFCModelV2
from torch.cuda import amp  
import time
from tensorboardX import SummaryWriter
from torch import nn
from metrics import padded_cmap,map_score
import sklearn
from torchtoolbox.tools import mixup_criterion,mixup_data
from torch.nn.functional import binary_cross_entropy_with_logits
from losses import SmoothBCEFocalLoss,MultiLoss,BCECNNLoss
from utils import *
import  audiomentations as AA
from torchtoolbox.tools import cutmix_data,MixingDataController

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s]- %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])
 
    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
 
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
 
    return logger

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
df_valid = pd.read_csv(CFG.train_path)

bird2021_df = pd.read_csv('/root/projects/BirdClef2025/externaldata/birdclef-2021/train_metadata.csv')
bird2022_df = pd.read_csv('/root/projects/BirdClef2025/externaldata/birdclef-2022/train_metadata.csv')
bird2023_df = pd.read_csv('/root/projects/BirdClef2025/externaldata/birdclef-2023/train_metadata.csv')
bird2024_df = pd.read_csv('/root/projects/BirdClef2025/externaldata/birdclef-2024/train_metadata.csv')

bird2021_df['filename'] = bird2021_df['primary_label']+'/'+bird2021_df['filename']
bird2021_df['path'] = '/root/projects/BirdClef2025/externaldata/birdclef-2021/train_short_audio/'+bird2021_df['filename']
bird2022_df['path'] = '/root/projects/BirdClef2025/externaldata/birdclef-2022/train_audio/'+bird2022_df['filename']
bird2023_df['path'] = '/root/projects/BirdClef2025/externaldata/birdclef-2023/train_audio/'+bird2023_df['filename']
bird2024_df['path'] = '/root/projects/BirdClef2025/externaldata/birdclef-2024/train_audio/'+bird2024_df['filename']

df_train['path'] = '/root/projects/BirdClef2025/data/train_audio/'+df_train['filename']
df_valid['path'] = '/root/projects/BirdClef2025/data/train_audio/'+df_valid['filename']

allbird_df = pd.concat((df_train[df_train.columns],
                        bird2021_df[['filename','primary_label','path']],
                        bird2022_df[['filename','primary_label','path']],
                        bird2023_df[['filename','primary_label','path']],
                        bird2024_df[['filename','primary_label','path']],
                        ))

# bird2020_df = pd.read_csv('/root/projects/BirdClef2025/externaldata/external_data_ogg/cornell-birdsong-recognition-2020/train.csv')
# external1_df = pd.read_csv('/root/projects/BirdClef2025/externaldata/external_data_ogg/xeno-canto-bird-recordings-extended/train_extended.csv')

# bird2020_df['filename'] = bird2020_df['ebird_code']+'/'+bird2020_df['filename']
# bird2020_df['path'] = '/root/projects/BirdClef2025/externaldata/external_data_ogg/cornell-birdsong-recognition-2020/train_audio/' + bird2020_df['filename']

# external1_df['filename'] = external1_df['ebird_code'] +'/'+ external1_df['filename']
# external1_df['path'] = '/root/projects/BirdClef2025/externaldata/external_data_ogg/xeno-canto-bird-recordings-extended/train_audio/' + external1_df['filename']
# # external_df = external1_df
# external_df = pd.concat((bird2020_df,external1_df))
# del external_df['primary_label']
# external_df = external_df.rename(columns={"ebird_code":"primary_label"})

df_train = pd.concat([allbird_df, pd.get_dummies(allbird_df['primary_label'])], axis=1)
df_valid = pd.concat([df_valid, pd.get_dummies(df_valid['primary_label'])], axis=1)

birds = list(df_train.primary_label.unique())
missing_birds = list(set(list(df_train.primary_label.unique())).difference(list(df_valid.primary_label.unique())))
non_missing_birds = list(set(list(df_train.primary_label.unique())).difference(missing_birds))
df_valid[missing_birds] = 0

df_valid = df_valid[df_train.columns] ## Fix order
df_train = df_train[~df_train.filename.isin(df_valid.filename)]

# df_train = upsample_data(df_train,thr=50)
CFG.lr = 1e-4
CFG.num_classes = len(birds)
CFG.duration = 7

all_bgnoise = glob('/root/projects/BirdClef2025/externaldata/backgroundnoise/*/*')
all_bgnoise.sort()

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
                    AA.AddBackgroundNoise(all_bgnoise,min_snr_db=3,max_snr_db=30,p=0.5),
                    AA.Gain(min_gain_db=-12, max_gain_db=12, p=0.2),
                ]
            )
train_set = BirdDataset(df_train,sr = CFG.sample_rate,duration = CFG.duration,train = True,audio_augmentations=wav_transform)
val_set = BirdDataset(df_valid,sr = CFG.sample_rate,duration = CFG.infer_duration,train = False)
trainloader = DataLoader(train_set,batch_size=CFG.batch_size,shuffle=True,num_workers=24,pin_memory=True)
valloader = DataLoader(val_set,batch_size=CFG.batch_size,shuffle=False,num_workers=24,pin_memory=True)

# loss
# criterion = torch.nn.BCEWithLogitsLoss()
# loss
# criterion = torch.nn.BCEWithLogitsLoss()
loss_ = BCECNNLoss()
# loss_ = MultiLossWeighting(smoothing=CFG.smoothing_factor)

# model
model = BirdClefCNNFCModel(CFG.model,num_classes=CFG.num_classes,pretrained=CFG.pretrained).cuda()
# model_dict = model.state_dict()
# pretrianed_dict = torch.load('/home/lijw/BirdCLEF/BirdCLEF-Baselinev2/logs/2023-04-23T10:18-efv2b1-mel224-pretrained/saved_model_lastepoch.pt')
# pretrianed_dict = {k:v for k,v in pretrianed_dict.items() if 'att_block' not in k}
# model_dict.update(pretrianed_dict)
# model.load_state_dict(pretrianed_dict,strict=False)

# optimzer
# backbone_params = list(filter(lambda x: 'att_block' not in x[0],model.named_parameters())) # vit
# for ind, param in enumerate(backbone_params):
#     backbone_params[ind] = param[1]
# fc_params         = list(filter(lambda x: 'att_block' in x[0]*10,model.named_parameters())) # vit
# for ind, param in enumerate(fc_params):
#     fc_params[ind] = param[1]
# to_optim          = [{'params':backbone_params,'lr':CFG.lr},
#                         {'params':fc_params,'lr':CFG.lr*10}]

optimizer = optim.Adam(model.parameters(),lr=CFG.lr)
scheduler = fetch_scheduler(optimizer,len(trainloader))

cutmix_aug = MixingDataController(mixup=CFG.use_mixup,cutmix=CFG.use_cutmix,mixup_prob=CFG.mixup_pro,cutmix_prob=CFG.cutmix_pro)
mixup_clef = Mixup(mix_beta=1)

best_cmAP = -np.inf
current_time = time.strftime("%Y-%m-%dT%H:%M", time.localtime())
my_writer = SummaryWriter(log_dir=CFG.log_dir+current_time+'-pretrain/')
logger = get_logger(CFG.log_dir+current_time+'-pretrain/'+'/training.log')
logger.info('the hyperparameters of the network is:')
for k, v in dict(vars(CFG)).items():
    if '__' not in k:
        logger.info(k+f':{v}')
for i in range(CFG.epochs):
    model.train()
    scaler = amp.GradScaler()
    for idx,(audios,labels) in enumerate(trainloader):
        audios = audios.to(torch.float32).to(CFG.device)
        labels = labels.to(CFG.device)
        images = model.get_mel_gram(audios)
        images = torch.repeat_interleave(images.unsqueeze(1),repeats=3,dim=1)

        data_label_list = mixup_data(images,labels)
        with amp.autocast(enabled=True):
            if data_label_list[-1]:
                images = data_label_list[0]
                pred = model(images)
                label_a,label_b = data_label_list[1],data_label_list[2]
                loss = mixup_criterion(loss_,pred,label_a,label_b,data_label_list[3])
            else:
                images = data_label_list[0]
                pred = model(images)
                loss = loss_(pred,data_label_list[1])
            loss = loss/CFG.n_accumulate
        scaler.scale(loss).backward()
        
        if (idx + 1) % CFG.n_accumulate == 0:
            scaler.step(optimizer)
            scaler.update()

            # zero the parameter gradients
            optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()
        lr = optimizer.param_groups[0]['lr']
        logger.info('epoch:[{}/{}]  step:[{}/{}]  lr={:.7f}  loss={:.5f}'.format(i,CFG.epochs,idx , len(trainloader), lr, loss))

    torch.save(model.state_dict(),CFG.log_dir+current_time+'-pretrain/'+'/saved_model.pt')
torch.save(model.state_dict(),CFG.log_dir+current_time+'-pretrain/'+'/saved_model_lastepoch.pt')

        

        
        
