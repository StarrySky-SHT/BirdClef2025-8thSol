import numpy as np
from glob import glob
import librosa as lb
import soundfile as sf
from tqdm import trange
import os
from pydub import AudioSegment

filelist = glob('/home/data/lijw/dataset/downloadAudios4/*/*')
removed_file = []
for i in trange(len(filelist)):
    audio = AudioSegment.from_file(filelist[i])
    audio.set_frame_rate(32000).export(filelist[i],format="ogg")

# for i in trange(len(filelist)):
#     data,sr = sf.read(filelist[i])

print(1)
