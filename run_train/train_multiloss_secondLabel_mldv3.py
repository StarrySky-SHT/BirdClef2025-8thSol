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
from dataset import BirdDataset,fetch_scheduler,AudioAug,BirdDatasetTwoLabel,BirdDatasetSplitTrain,BirdDatasetWithPseudoLabel
from torch.utils.data import Dataset,DataLoader,WeightedRandomSampler
from torch import optim
from model import BirdClefSEDAttModel,TeacherModel
from torch.cuda import amp  
import time
from tensorboardX import SummaryWriter
from torch import nn
from metrics import padded_cmap,map_score
import sklearn
from torchtoolbox.tools import mixup
from torch.nn.functional import binary_cross_entropy_with_logits,binary_cross_entropy
from utils import *
from losses import FocalLossBCE,BCELoss,LEDLoss,RKdAngle,RkdDistance,MLDLoss
import  audiomentations as AA
from torchtoolbox.tools import cutmix_data,MixingDataController
import torch.nn.functional as F
import tensorflow_hub as hub
import tensorflow as tf


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
df_valid = df_valid.sample(frac=0.1,random_state=42)

df_train['path'] = CFG.data_root+df_train['filename']
df_valid['path'] = CFG.data_root+df_valid['filename']

# external_train = pd.read_csv('/root/projects/BirdClef2025/data/external_train.csv')
# df_train = pd.concat((df_train,external_train))
pesudo_df = pd.read_csv('/root/projects/BirdClef2025/BirdCLEF2023-30th-place-solution-master/usefulFunc/pesudo_labelv12_ensemble.csv')
# pesudo_df['primary_label'] = pesudo_df[pesudo_df.columns[1:]].idxmax(axis=1)
# pesudo_df['rating'] = 5

# external_train = pd.read_csv('/root/projects/BirdClef2025/data/external_train.csv')
# df_train = pd.concat((df_train,external_train))

df_train = pd.concat([df_train, pd.get_dummies(df_train['primary_label']).astype(np.int32)], axis=1)
df_valid = pd.concat([df_valid, pd.get_dummies(df_valid['primary_label']).astype(np.int32)], axis=1)

birds = list(pd.get_dummies(df_train['primary_label']).columns)
missing_birds = list(set(list(df_train.primary_label.unique())).difference(list(df_valid.primary_label.unique())))
non_missing_birds = list(set(list(df_train.primary_label.unique())).difference(missing_birds))
df_valid[missing_birds] = 0
df_valid = df_valid[df_train.columns] ## Fix order

df_train = upsample_data(df_train,thr=50)

# sample_weights = (
#     df_train['primary_label'].value_counts() / 
#     df_train['primary_label'].value_counts().sum()
# )  ** (-0.5)
# sample_weights_dict = dict(sample_weights)
# sample_weights = df_train['primary_label'].map(lambda x:sample_weights_dict[x])
# sampler = WeightedRandomSampler(sample_weights,num_samples=len(df_train))

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
                        p=0.1,
                    ),
                    AA.Shift(p=0.1),
                    RandomVolume(p=0.1),
                    # PitchShift(p=0.2),
                    AA.AddBackgroundNoise(all_bgnoise,min_snr_db=3,max_snr_db=30,p=0.1),
                    AA.Gain(min_gain_db=-12, max_gain_db=12, p=0.1),
                ]
            )

train_set = BirdDatasetWithPseudoLabel(df_train,pesudo_df,bird_cols=birds,sr=CFG.sample_rate, duration=CFG.duration, audio_augmentations=wav_transform, train=True)
val_set = BirdDatasetTwoLabel(df_valid,bird_cols=birds,sr = CFG.sample_rate,duration = CFG.infer_duration,train = False)
trainloader = DataLoader(train_set,batch_size=CFG.batch_size,num_workers=16,pin_memory=True,shuffle=True)
valloader = DataLoader(val_set,batch_size=CFG.batch_size,shuffle=False,num_workers=16,pin_memory=True)

# loss
# criterion = torch.nn.BCEWithLogitsLoss()
loss_ = BCELoss()
loss_mld = MLDLoss()
# dist_criterion = RkdDistance()
# angle_criterion = RKdAngle()
# loss_ = MultiLossWeighting(smoothing=CFG.smoothing_factor)

# model
model_teacher = TeacherModel()
model = BirdClefSEDAttModel(CFG.model,num_classes=CFG.num_classes,pretrained=CFG.pretrained).cuda()

# optimzer
if CFG.finetune_weight == True:
    pretrianed_dict = torch.load('/root/projects/BirdClef2025/BirdCLEF2023-30th-place-solution-master/logs/2025-03-19T16:33-pretrain-efv2b3Pretrain/saved_model_lastepoch.pt')
    to_load_dict = dict()
    for k,v in pretrianed_dict.items():
        if 'backbone' in k or 'fc1' in k:
            to_load_dict[k] = v
    model.load_state_dict(to_load_dict,strict=False)

    backbone_params = list(filter(lambda x: 'att_block' not in x[0],model.named_parameters())) # vit
    for ind, param in enumerate(backbone_params):
        backbone_params[ind] = param[1]
    fc_params         = list(filter(lambda x: 'att_block' in x[0],model.named_parameters())) # vit
    for ind, param in enumerate(fc_params):
        fc_params[ind] = param[1]
    to_optim          = [{'params':backbone_params,'lr':CFG.lr},
                            {'params':fc_params,'lr':CFG.lr*10}]
else:
    to_optim = model.parameters()


optimizer = optim.Adam(to_optim,lr=CFG.lr)
scheduler = fetch_scheduler(optimizer,len(trainloader))

best_cmAP = -np.inf
current_time = time.strftime("%Y-%m-%dT%H:%M", time.localtime())
my_writer = SummaryWriter(log_dir=CFG.log_dir+current_time)
logger = get_logger(CFG.log_dir+current_time+'/training.log')
logger.info('the hyperparameters of the network is:')
for k, v in dict(vars(CFG)).items():
    if '__' not in k:
        logger.info(k+f':{v}')
for i in range(CFG.epochs):
    model.train()
    scaler = amp.GradScaler()
    for idx,(audios,labels,weights) in enumerate(trainloader):
        audios = audios.to(torch.float32).to(CFG.device)
        labels = labels.to(CFG.device)
        weights = weights.to(CFG.device)
        if np.random.uniform(0,1) < 0.5:
            summix_res = sumix(audios,labels)
            audios = summix_res['waves']
            labels = summix_res['labels']

        images = model.get_mel_gram(audios)
        images = torch.repeat_interleave(images.unsqueeze(1),repeats=3,dim=1)

        data_label_list = mixup_data(images,labels,use_mixup=CFG.use_mixup)
        with amp.autocast(enabled=True):
            if data_label_list[-1]:
                images = data_label_list[0]
                pred = model.forward(images)
                pred_student = (pred['clipwise_output'] + pred['maxframewise_output'])/2
                label_a,label_b = data_label_list[1],data_label_list[2]
                loss = mixup_criterion(loss_,pred,label_a,label_b,data_label_list[3],classes_weight=weights)
                with torch.no_grad():
                    pred_teacher = model_teacher.forward(images)
                mld_loss = loss_mld(pred_teacher,pred_student)
                loss = 0.1*loss+0.9*mld_loss
            else:
                images = data_label_list[0]
                pred = model.forward(images)
                pred_student = (pred['clipwise_output'] + pred['maxframewise_output'])/2
                loss = loss_(pred,data_label_list[1],weights=weights)
                with torch.no_grad():
                    pred_teacher = model_teacher.forward(images)
                
                mld_loss = loss_mld(pred_teacher,pred_student)
                loss = 0.1*loss+0.9*mld_loss
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
    gt_df = pd.DataFrame(gtall_numpy,columns=birds)
    # cmAP_pad3_score = padded_cmap(gt_df, pred_df, padding_factor = 3)
    cmAP_pad5_score,cmAP_pad5_score_ = padded_cmap(gt_df, pred_df, padding_factor = 5)
    AP_score = sklearn.metrics.label_ranking_average_precision_score(np.int16(gtall_numpy),predall_numpy)
    scored_gt = gtall_numpy[:,np.where(gtall_numpy.sum(axis=0)!=0)[0]]
    scored_pred = predall_numpy[:,np.where(gtall_numpy.sum(axis=0)!=0)[0]]
    macro_auc = sklearn.metrics.roc_auc_score(np.int16(scored_gt),scored_pred,)

    logging.info('epoch:[{}/{}]  cmAP:{:.4f} AP:{:.4f} AUC:{:.4f}'.format(i,CFG.epochs,cmAP_pad5_score,AP_score,macro_auc))

    if cmAP_pad5_score> best_cmAP:
        last_best_cmAP = best_cmAP
        best_cmAP = cmAP_pad5_score.copy()
        logger.info(f'epoch:{i} val_cmAP improved from {last_best_cmAP} to {best_cmAP}')
        torch.save(model.state_dict(),CFG.log_dir+current_time+'/saved_model.pt')
    else:
        logger.info(f'epoch:{i} val_cmAP did not improved from {best_cmAP}, now val_cmAP is {cmAP_pad5_score}...')
torch.save(model.state_dict(),CFG.log_dir+current_time+'/saved_model_lastepoch.pt')

        