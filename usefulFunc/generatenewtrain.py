import pandas as pd
import numpy as np
from glob import glob
import soundfile as sf
import warnings
warnings.filterwarnings('error')   

filelist = glob('/home/data/lijw/dataset/downloadAudiosRemoveDup/*/*')
df = pd.read_csv("/home/data/lijw/dataset/BirdCLEF/specs/train.csv")

primary_list = [i.split('/')[-2] for i in filelist]
path_list = ['/home/data/lijw/dataset/BirdCLEF/train_audio_add_new/'+i.split('/')[-2]+'/'+i.split('/')[-1] for i in filelist]
df_new = pd.DataFrame({'primary_label':primary_list,'path':path_list})
df_ = pd.concat((df,df_new))
df_ = df_.drop('Unnamed: 0',axis=1)
df_.to_csv("/home/data/lijw/dataset/BirdCLEF/specs/new_train_v2.csv")
# print(1)

# warningfile = []
# for idx,row in df_new.iterrows():
#     try:
#         print(row['path'])
#         sf.read(row['path'])
#     except:
#         warningfile.append(row['path'])
# print(1)
