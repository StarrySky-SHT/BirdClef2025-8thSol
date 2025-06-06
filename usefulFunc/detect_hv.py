from silero_vad import load_silero_vad, read_audio, get_speech_timestamps
import numpy as np
import torch
from joblib import Parallel, delayed
from multiprocessing.dummy import Pool as ThreadPool
from glob import glob
from tqdm import tqdm
import pickle

model = load_silero_vad()
filelist = glob('/root/projects/BirdClef2025/data/train_audio/*/*.ogg')
filelist.sort()

fileresult = dict()
for file in tqdm(filelist,total=len(filelist)):
    audio = read_audio(file)
    speech_timestamps = get_speech_timestamps(
    audio,
    threshold=0.3,
    min_silence_duration_ms=1000,  # Threshold for speech detection
    model=model,
    return_seconds=True,  # Return speech timestamps in seconds (default is samples)
    )
    filename = file.split('/')[-1]
    if len(speech_timestamps) > 0:
        fileresult[filename] = speech_timestamps

with open('/root/projects/BirdClef2025/BirdCLEF2023-30th-place-solution-master/usefulFunc/speech_timestamps.pkl', 'wb') as f:
    pickle.dump(fileresult, f)