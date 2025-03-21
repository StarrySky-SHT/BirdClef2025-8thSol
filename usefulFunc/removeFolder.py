import pandas as pd
from glob import glob
import numpy as np
import os

df = pd.read_csv('/home/data/lijw/dataset/BirdCLEF/train_metadata.csv')
primary_label = list(df['primary_label'].unique())
scitific_name = list(df['scientific_name'].unique())
name_dict = dict(zip(scitific_name,primary_label))

targetpath = '/home/data/lijw/dataset/downloadAudios/'
filelist = glob(targetpath+'*')
filelist = [i.replace('-',' ') for i in filelist]
for i in filelist:
    os.rename(targetpath+i.split('/')[-1].replace(' ','-'),targetpath+name_dict[i.split('/')[-1]])