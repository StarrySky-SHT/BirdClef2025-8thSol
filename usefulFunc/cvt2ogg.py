import numpy as np
from glob import glob
from pydub import AudioSegment
import soundfile as sf
import pandas as pd
import os

fileslist = glob('/root/projects/BirdClef2025/externaldata/download_xc_data/*/*')
train_path = '/root/projects/BirdClef2025/data/train_audio'
filtered_filelist = []
for file in fileslist:
    ebird = file.split('/')[-2]
    existed_filenames = glob(train_path+f'/{ebird}/*')
    existed_filenames = [i.split('/')[-1].replace('XC','').replace('.ogg','') for i in existed_filenames]
    download_filename = file.split('XC')[-1].split('.')[0]
    if download_filename not in existed_filenames:
        filtered_filelist.append(file)

from tqdm import tqdm
import librosa as lb
error_list = []
for idx,file in tqdm(enumerate(filtered_filelist),total=len(filtered_filelist)):
    audio = lb.load(file,sr=32000)
    signal = audio[0]
    targetpath = '/'.join(file.replace('download_xc_data','download_xc_data_resample').split('/')[:-1])
    if not os.path.exists(targetpath):
        os.mkdir(targetpath)
    sf.write(file.replace('download_xc_data','download_xc_data_resample').replace('.ogg','.wav'),signal,samplerate=audio[1])
    print(f'error file:{file}')
    error_list.append(file)

filtered_filelist = glob('/root/projects/BirdClef2025/externaldata/download_xc_data_resample/*/*.wav')
for file in tqdm(filtered_filelist,total=len(filtered_filelist)):
    try:
        if '.wav' in file:
            mode = 'wav'
        elif '.mp3' in file:
            mode = 'mp3'
        elif '.m4a' in file:
            mode = 'm4a'
        elif '.mpga' in file:
            mode = 'mp3'
        else:
            raise TypeError
        audio = AudioSegment.from_file(file,format=mode)
        audio = audio.set_frame_rate(32000)  # 修改采样率为32kHz
        if not os.path.exists('/'.join(file.replace('download_data','download_data_ogg').split('/')[:-1])):
            os.mkdir('/'.join(file.replace('download_data','download_data_ogg').split('/')[:-1]))
        audio.export(file.replace('download_data','download_data_ogg').replace('.wav','.ogg').replace('.mp3','.ogg').replace('.mpga','.ogg').replace('.m4a','.ogg'),
                    format='ogg')
    except:
        print(f'error file:{file}')