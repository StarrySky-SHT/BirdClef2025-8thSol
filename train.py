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
df_valid = pd.read_csv(CFG.valid_path)

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

# loss
# criterion = torch.nn.BCEWithLogitsLoss()
criterion = nn.BCELoss()    
loss_ = SmoothBCEFocalLoss(smoothing=CFG.smoothing_factor)

# model
model = BirdClefModel(CFG.model,num_classes=CFG.num_classes,pretrained=CFG.pretrained).cuda()
# model_dict = model.state_dict()
# pretrianed_dict = torch.load('/home/lijw/BirdCLEF/BirdCLEF-Baselinev2/logs/2023-05-17T01:03-efv2b3-pretrained-new/saved_model.pt')
# pretrianed_dict = {k:v for k,v in pretrianed_dict.items() if 'att_block' not in k}
# pretrianed_dict = {k:v for k,v in pretrianed_dict.items() if 'bn0' not in k}
# pretrianed_dict = {k:v for k,v in pretrianed_dict.items() if 'spectrogram_extractor' not in k}
# model_dict.update(pretrianed_dict)
# model.load_state_dict(pretrianed_dict,strict=False)

# optimzer
backbone_params = list(filter(lambda x: 'att_block' not in x[0],model.named_parameters())) # vit
for ind, param in enumerate(backbone_params):
    backbone_params[ind] = param[1]
fc_params         = list(filter(lambda x: 'att_block' in x[0]*10,model.named_parameters())) # vit
for ind, param in enumerate(fc_params):
    fc_params[ind] = param[1]
to_optim          = [{'params':backbone_params,'lr':CFG.lr},
                        {'params':fc_params,'lr':CFG.lr*10}]

optimizer = optim.Adam(model.parameters(),lr=CFG.lr)
scheduler = fetch_scheduler(optimizer,len(trainloader))

cutmix_aug = MixingDataController(mixup=CFG.use_mixup,cutmix=CFG.use_cutmix,mixup_prob=CFG.mixup_pro,cutmix_prob=CFG.cutmix_pro)
mixup_clef = Mixup(mix_beta=1)

best_cmAP = -np.inf
current_time = time.strftime("%Y-%m-%dT%H:%M", time.localtime())
my_writer = SummaryWriter(log_dir='/home/lijw/BirdCLEF/BirdCLEF-Baselinev2/logs/'+current_time)
logger = get_logger('/home/lijw/BirdCLEF/BirdCLEF-Baselinev2/logs/'+current_time+'/training.log')
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

        data_label_list = cutmix_aug.get_data(images,labels)
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

    model.eval()
    logger.info('start valid...')
    predall_tensor = torch.zeros(0).to(CFG.device)
    gtall_tensor = torch.zeros(0).to(CFG.device)
    for idx,(audios,labels) in enumerate(valloader):
        audios = audios.to(torch.float32).to(CFG.device)
        labels = labels.to(torch.long).to(CFG.device)

        with torch.no_grad():
            images = model.get_mel_gram(audios)
            images = torch.repeat_interleave(images.unsqueeze(1),repeats=3,dim=1)
            pred = model.forward(images)
        predall_tensor = torch.cat((predall_tensor,pred),dim=0)
        gtall_tensor = torch.cat((gtall_tensor,labels),dim=0)
    
    predall_numpy = predall_tensor.cpu().detach().numpy()
    gtall_numpy = gtall_tensor.cpu().detach().numpy()

    pred_df = pd.DataFrame(predall_numpy,columns=birds)
    gt_df = pd.DataFrame(gtall_numpy,columns=birds)
    # cmAP_pad3_score = padded_cmap(gt_df, pred_df, padding_factor = 3)
    cmAP_pad5_score,cmAP_pad5_score_ = padded_cmap(gt_df, pred_df, padding_factor = 5)
    AP_score = sklearn.metrics.label_ranking_average_precision_score(np.int16(gtall_numpy),predall_numpy)

    logging.info('epoch:[{}/{}]  cmAP:{:.4f} AP:{:.4f}'.format(i,CFG.epochs,cmAP_pad5_score,AP_score))

    if cmAP_pad5_score> best_cmAP:
        last_best_cmAP = best_cmAP
        best_cmAP = cmAP_pad5_score.copy()
        logger.info(f'epoch:{i} val_cmAP improved from {last_best_cmAP} to {best_cmAP}')
        torch.save(model.state_dict(),'/home/lijw/BirdCLEF/BirdCLEF-Baselinev2/logs/'+current_time+'/saved_model.pt')
    else:
        logger.info(f'epoch:{i} val_cmAP did not improved from {best_cmAP}, now val_cmAP is {cmAP_pad5_score}...')
torch.save(model.state_dict(),'/home/lijw/BirdCLEF/BirdCLEF-Baselinev2/logs/'+current_time+'/saved_model_lastepoch.pt')

        
