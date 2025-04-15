import pandas as pd
import numpy as np
import soundfile as sf
from glob import glob
from tqdm import tqdm
from joblib import Parallel, delayed

df = pd.read_csv('/root/projects/BirdClef2025/BirdCLEF2023-30th-place-solution-master/usefulFunc/train.csv')
df['path'] = '/root/projects/BirdClef2025/data/train_audio/' + df['filename']

def get_duration(path):
    waveform,sr = sf.read(path)
    duration = len(waveform) / sr
    return path,duration
# multiprocess
path_durations = Parallel(n_jobs=16)(delayed(get_duration)(path) for path in tqdm(df['path']))
# add to original df
df['duration'] = [x[1] for x in path_durations]
# save to csv
df.to_csv('/root/projects/BirdClef2025/BirdCLEF2023-30th-place-solution-master/usefulFunc/train_duration.csv', index=False)