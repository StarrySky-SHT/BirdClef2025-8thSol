from joblib import Parallel, delayed
import torchaudio
import numpy as np
import pandas as pd
import librosa as lb
import soundfile as sf
import os
import tqdm
from pydub import AudioSegment
AudioSegment.converter = '/usr/bin/ffmpeg'
import faulthandler
# 在import之后直接添加以下启用代码即可
faulthandler.enable()

targetpath = '/root/projects/BirdClef2025/externaldata/extentedbird/train_audio_ogg/'

df = pd.read_csv('/root/projects/BirdClef2025/externaldata/extentedbird/train_extended.csv')
df['filename'] = df['ebird_code'] +'/'+ df['filename']
df['path'] = '/root/projects/BirdClef2025/externaldata/extentedbird/train_audio/' + df['filename']

import torch
def process1(row):
    # audio,sr = torchaudio.load(row['path'])
    # audio_numpy = torch.mean(audio,dim=0).numpy()
    audio = AudioSegment.from_file(row['path'])
    sr = audio.frame_rate
    channels = audio.channels
    if channels == 2:
        left_channel = audio.split_to_mono()[0].get_array_of_samples() # 左声道
        right_channel = audio.split_to_mono()[1].get_array_of_samples()  # 右声道
        audio_numpy = np.array(left_channel)/2+np.array(right_channel)/2
    else:
        audio_numpy = audio.get_array_of_samples()
        audio_numpy = np.array(audio_numpy)
    audio_numpy = audio_numpy.astype(np.float32)
    audio_numpy = audio_numpy/32768
    #分离声道
    if sr != 32000:
        audio_numpy = lb.resample(audio_numpy, orig_sr=sr, target_sr=32000, res_type="kaiser_fast")
    if not os.path.exists(targetpath+'/'+row['path'].split('/')[-2]):
        os.mkdir(targetpath+'/'+row['path'].split('/')[-2])
    sf.write(targetpath+'/'+row['path'].split('/')[-2]+'/'+row['path'].split('/')[-1],audio_numpy,samplerate=32000,format='OGG',subtype='VORBIS')
errorfile = []
for idx,row in tqdm.tqdm(df.iterrows(),total=df.shape[0]):
    try:
        process1(row)
    except:
        errorname = row['filename']
        print(f'error:{errorname}')
        errorfile.append(row['filename'])
print(1)
# Parallel(n_jobs=16)(delayed(process1)(row) for idx,row in tqdm(df.iterrows(), total=df.shape[0]))