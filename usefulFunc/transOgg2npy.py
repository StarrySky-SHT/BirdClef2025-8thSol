import numpy as np
import os
from joblib import Parallel,delayed
from glob import glob

from pickle import TRUE
import numpy as np
from  soundfile import SoundFile
import soundfile as sf
import pandas as pd
from pathlib import Path
import librosa as lb
import tqdm
import joblib, json, re
from joblib import Parallel,delayed
from tqdm import trange
from tqdm import tqdm
from  sklearn.model_selection  import StratifiedKFold
import os
from torchlibrosa.stft import Spectrogram

df = pd.read_csv('/home/data/lijw/dataset/BirdCLEF/train_metadata.csv')
df['secondary_labels'] = df['secondary_labels'].apply(lambda x: re.findall(r"'(\w+)'", x))
df['len_sec_labels'] = df['secondary_labels'].map(len)

from sklearn.model_selection import train_test_split
import pandas as pd

def birds_stratified_split(df, target_col, test_size=0.2):
    class_counts = df[target_col].value_counts()
    low_count_classes = class_counts[class_counts < 2].index.tolist() ### Birds with single counts

    df['train'] = df[target_col].isin(low_count_classes)

    train_df, val_df = train_test_split(df[~df['train']], test_size=test_size, stratify=df[~df['train']][target_col], random_state=42)

    train_df = pd.concat([train_df, df[df['train']]], axis=0).reset_index(drop=True)

    # Remove the 'valid' column
    train_df.drop('train', axis=1, inplace=True)
    val_df.drop('train', axis=1, inplace=True)

    return train_df, val_df

train_df, valid_df = birds_stratified_split(df, 'primary_label', 0.2)

class Config:
    sampling_rate = 32000
    duration = 30
    fmin = 0
    fmax = None
    audios_path = Path("/home/data/lijw/dataset/BirdCLEF/train_audio")
    out_dir_train = Path("/home/data/lijw/dataset/BirdCLEF/train_audio_npy/") 
    

Config.out_dir_train.mkdir(exist_ok=True, parents=True)

def get_audio_info(filepath):
    """Get some properties from  an audio file"""
    with SoundFile(filepath) as f:
        sr = f.samplerate
        frames = f.frames
        duration = float(frames)/sr
    return {"frames": frames, "sr": sr, "duration": duration}

def add_path_df(df):
    
    df["path"] = [str(Config.audios_path/filename) for filename in df.filename]
    df = df.reset_index(drop=True)
    tasks = Parallel(n_jobs=2)(delayed(get_audio_info)(filepath) for filepath in df.path)
    df2 = pd.DataFrame(tasks).reset_index(drop=True)
    df = pd.concat([df,df2], axis=1).reset_index(drop=True)

    return df

train_df = add_path_df(train_df)
valid_df = add_path_df(valid_df)

class AudioToImage:
    def __init__(self, sr=Config.sampling_rate, n_mels=128, fmin=Config.fmin, fmax=Config.fmax, duration=Config.duration, step=None, res_type="kaiser_fast", resample=True, train = True):

        self.sr = sr
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax or self.sr//2

        self.duration = duration
        self.audio_length = self.duration*self.sr
        self.step = step or self.audio_length
        
        self.res_type = res_type
        self.resample = resample

        self.train = train

    def __call__(self, row, save=True):

      audio, orig_sr = sf.read(row.path, dtype="float32")

      if self.resample and orig_sr != self.sr:
        audio = lb.resample(audio, orig_sr, self.sr, res_type=self.res_type)
 
      if save:
        path = Config.out_dir_train/f"{row.filename}.npy"
        path.parent.mkdir(exist_ok=True, parents=True)
        np.save(str(path), audio)
      else:
        return  row.filename, audio
      
# converter = AudioToImage(step=int(Config.duration*0.666*Config.sampling_rate),train=True)
# for row in tqdm(train_df.itertuples(False),total=train_df.shape[0]):
#     converter(row)

def get_audios_as_numpy(df, train = True):
    converter = AudioToImage(step=int(Config.duration*Config.sampling_rate),train=train)
    Parallel(8)(delayed(converter)(row) for row in tqdm(df.itertuples(False),total=df.shape[0]))

all_df = pd.concat((train_df,valid_df))
get_audios_as_numpy(all_df, train = True)