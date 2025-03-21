import numpy as np
import pandas as pd
from glob import glob
from joblib import Parallel,delayed
from sklearn.model_selection import train_test_split

bird2021_df = pd.read_csv('/home/data/lijw/dataset/BirdCLEF-2021/train_metadata.csv')
bird2022_df = pd.read_csv('/home/data/lijw/dataset/BirdCLEF-2022/train_metadata.csv')

bird2021_df['filepath'] = '/home/data/lijw/dataset/BirdCLEF-2021/train_audio/'+bird2021_df['filename']
bird2022_df['filepath'] = '/home/data/lijw/dataset/BirdCLEF-2022/train_audio/'+bird2022_df['filename']

allbird_df = pd.concat((bird2021_df,bird2022_df))