from joblib import Parallel, delayed
import soundfile as sf
from tqdm import tqdm
from glob import glob

src_path = '/root/projects/BirdClef2025/data/train_soundscapes/'
def cut_audio(filename):
    src_path = '/root/projects/BirdClef2025/data/train_soundscapes/'
    audio,sr = sf.read(src_path+filename)
    for time in range(0,60,10):
        audio_seg = audio[int(sr*time):int(sr*(time+10))]
        # write to file
        sf.write('/root/projects/BirdClef2025/data/train_soundscapes_10s/'+filename.replace('.ogg','')+'_'+str(time)+'s'+'.ogg',audio_seg,sr)

filelist = glob('/root/projects/BirdClef2025/data/train_soundscapes/*.ogg')
filelist.sort()
filelist = [i.split('/')[-1] for i in filelist]
Parallel(n_jobs=32)(delayed(cut_audio)(filename) for filename in tqdm(filelist,total=len(filelist)))