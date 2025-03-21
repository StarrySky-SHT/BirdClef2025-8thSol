import librosa as lb
from joblib import Parallel,delayed
from glob import glob
import pandas as pd
import shutil
import soundfile as sf
from tqdm import tqdm
import numpy as np
targetPath = '/home/data/lijw/dataset/BirdCLEF-2021/train_soundscapes_nocall/'
srcPath = '/home/data/lijw/dataset/BirdCLEF-2021/train_soundscapes/'
df = pd.read_csv('/home/data/lijw/dataset/BirdCLEF-2021/train_soundscape_labels.csv')
# def process(row):
#     if row['hasbird'] == 0:
#         audio,sr = lb.load(srcPath+row['filename'],sr=32000)
#         np.save(srcPath.replace('/wav','/wav_nocall')+row['filename'].replace('wav','npy'),audio)

# Parallel(8)(delayed(process)(row) for idx,row in tqdm(df.iterrows(),total=df.shape[0]))

filelist = glob(srcPath+'*.ogg')
for i in range(len(filelist)):
    audio,sr = sf.read(filelist[i],dtype='float32')
    for j,row in df.iterrows():
        if row['row_id'].split('_')[0] in filelist[i] and row['birds']=='nocall':
            save_audio = audio[((int(row['seconds'])-5)*sr):(int(row['seconds'])*sr)]
            np.save(targetPath+row['row_id']+'.npy',save_audio)