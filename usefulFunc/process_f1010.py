import numpy as np
from glob import glob
import pandas as pd
import sys
sys.path.append('/home/lijw/BirdCLEF/BirdCLEF-Baselinev2/')
from config import CFG

df_train = pd.read_csv(CFG.train_path)
df_valid = pd.read_csv(CFG.valid_path)

df_train = pd.concat([df_train, pd.get_dummies(df_train['primary_label'])], axis=1)
df_valid = pd.concat([df_valid, pd.get_dummies(df_valid['primary_label'])], axis=1)

birds = list(df_train.primary_label.unique())
missing_birds = list(set(list(df_train.primary_label.unique())).difference(list(df_valid.primary_label.unique())))
non_missing_birds = list(set(list(df_train.primary_label.unique())).difference(missing_birds))
df_valid[missing_birds] = 0
df_valid = df_valid[df_train.columns] ## Fix order

filelist = glob('/home/data/lijw/dataset/f1010bird/ff1010bird_wav/wav/*')
df_f1010 = pd.read_csv('/home/lijw/BirdCLEF/BirdCLEF-Indentify-Call/ff1010bird-duration7/rich_metadata.csv')
df_birdclef2023 = df_train
df_f1010_nocall = df_f1010[df_f1010['hasbird']==0]
df_f1010_nocall = df_f1010_nocall.sample(1000)
df_f1010_nocall = pd.DataFrame({'filename':list(df_f1010_nocall['filename'])})
df_f1010_nocall['filename'] = '/home/data/lijw/dataset/f1010bird/ff1010bird_wav/wav/'+ df_f1010_nocall['filename'] 
pd_concat = pd.concat((df_f1010_nocall,df_train),axis='index')
pd_concat.iloc[:1000,1:] = 0
print(1)