import librosa as lb
import soundfile as sf
import numpy as np
from pydub import AudioSegment
import pandas as pd

df = pd.read_csv('/home/data/lijw/dataset/BirdCLEF/specs/train.csv')

for idx,row in df.iterrows():
    if row['duration'] > 20:
        audio = sf.read(row['path'])
        
