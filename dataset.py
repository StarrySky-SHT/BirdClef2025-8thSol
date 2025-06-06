import torch
from config import CFG
import numpy as np
from torch.optim import lr_scheduler
import librosa as lb
import soundfile as sf
import audiomentations as AA
import torchvision.transforms as T
from utils import upsample_data,downsample_data,sumix
from model import NormalizeMelSpec
import pandas as pd
import pickle
from torchaudio.transforms import MelSpectrogram,AmplitudeToDB

# Generates random integer
def random_int(minval=0, maxval=1):
    return np.random.uniform(low=minval,high=maxval)

# Generats random float
def random_float(minval=0.0, maxval=1.0):
    rnd = np.random.uniform(low=minval,high=maxval)
    return rnd

# Randomly shift audio -> any sound at <t> time may get shifted to <t+shift> time
def TimeShift(audio, prob=0.5):
    # Randomly apply time shift with probability `prob`
    if random_float() < prob:
        # Calculate random shift value
        shift = random_int(minval=0, maxval=len(audio))
        # Randomly set the shift to be negative with 50% probability
        if random_float() < 0.5:
            shift = -shift
        # Roll the audio signal by the shift value
        audio = np.roll(audio, np.int16(shift), axis=0)
    return audio

# Apply random noise to audio data
def GaussianNoise(audio, std=[0.0025, 0.025], prob=0.5):
    # Select a random value of standard deviation for Gaussian noise within the given range
    std = random_float(std[0], std[1])
    # Randomly apply Gaussian noise with probability `prob`
    if random_float() < prob:
        # Add random Gaussian noise to the audio signal
        GN = np.random.normal(0,std,size=(len(audio),))
        audio = audio+GN # training=False don't apply noise to data
    return audio

# Applies augmentation to Audio Signal
def AudioAug(audio):
    # Apply time shift and Gaussian noise to the audio signal
    audio = TimeShift(audio, prob=CFG.time_shift_prob)
    audio = GaussianNoise(audio, prob=CFG.gn_prob)
    return audio

class BirdDataset(torch.utils.data.Dataset):

    def __init__(self,df, sr = CFG.sample_rate, duration = CFG.duration, audio_augmentations = None, train = True):
        self.df = df
        self.sr = sr 
        self.train = train
        self.duration = duration
        self.audio_length = self.duration * sr
        self.use_mixup = CFG.use_mixup
        self.audio_augmentations = audio_augmentations

    def __len__(self):
        return len(self.df)

    @staticmethod
    def normalize(image):
        image = image / 255.0
        #image = torch.stack([image, image, image])
        return image

    def normalize(self,x,type='z-score'):
        if type=='z-score':
            mean = np.mean(x)
            std = np.std(x)
            x = (x - mean)/(std+1e-8)
        elif type=='min-max':
            x = (x-x.min())/(x.max()-x.min())
        return x

    def crop_or_pad(self,y, length, is_train=True, start=None):
        len_y = len(y)
        effective_length = self.duration * self.sr
        if len_y < effective_length:
            new_y = np.ones(effective_length, dtype=y.dtype)*1e-9
            start = np.random.randint(effective_length - len_y)
            new_y[start:start + len_y] = y
            y = new_y.astype(np.float32)
        elif len_y > effective_length:
            start = np.random.randint(len_y - effective_length)
            y = y[start:start + effective_length].astype(np.float32)
        else:
            y = y.astype(np.float32)

        return y
    
    def get_first_duration(self,y):
        len_y = len(y)
        effective_length = self.duration * self.sr
        if len_y < effective_length:
            new_y = np.ones(effective_length, dtype=y.dtype)*1e-9
            start = np.random.randint(effective_length - len_y)
            new_y[start:start + len_y] = y
            y = new_y.astype(np.float32)
        elif len_y > effective_length:
            start = 0
            y = y[start:start + effective_length].astype(np.float32)
        else:
            y = y.astype(np.float32)

        return y

    def __getitem__(self, idx):
        row = self.df.iloc[idx].copy()
        audio,orig_sr = sf.read(row['path'].replace('.mp3','.ogg'))
        if len(audio.shape)==2:
            audio = np.mean(audio,axis=1)
        if orig_sr!=self.sr:
            audio = lb.resample(audio, orig_sr=orig_sr, target_sr=self.sr, res_type="kaiser_fast")
    
        if self.train:
            audio = self.crop_or_pad(audio,self.audio_length,self.train) # constant length (l,) array
            
            if self.audio_augmentations:
                audio = self.audio_augmentations(audio)
        else:
            audio = self.get_first_duration(audio)
            if self.audio_augmentations:
                audio = self.audio_augmentations(audio)
        return torch.from_numpy(audio),torch.tensor(row[14:]).float()

class BirdDatasetTwoLabel(torch.utils.data.Dataset):

    def __init__(self,df, bird_cols = None,sr = CFG.sample_rate, duration = CFG.duration, audio_augmentations = None, train = True):
        self.df = df
        self.sr = sr 
        self.train = train
        self.duration = duration
        self.audio_length = self.duration * sr
        self.use_mixup = CFG.use_mixup
        self.audio_augmentations = audio_augmentations
        self.bird_cols = bird_cols
        with open('/root/projects/BirdClef2025/data/train_voice_data_numpy.pkl','rb') as f:
            self.hv_record = pickle.load(f)

    def __len__(self):
        return len(self.df)

    @staticmethod
    def normalize(image):
        image = image / 255.0
        #image = torch.stack([image, image, image])
        return image

    def normalize(self,x,type='z-score'):
        if type=='z-score':
            mean = np.mean(x)
            std = np.std(x)
            x = (x - mean)/(std+1e-8)
        elif type=='min-max':
            x = (x-x.min())/(x.max()-x.min())
        return x

    def pick_rms(self, waves, duration_samples):
        # duration_samples = duration_samples * 32000
        if len(waves) <= duration_samples:
            # Circular padding
            repeat_count = int(np.ceil(duration_samples / len(waves)))
            waves = np.tile(waves, repeat_count)[:duration_samples]
            return waves

        stride = 32000
        max_rms = 0
        max_rms_start = 0

        for start in range(0, len(waves) - duration_samples + 1, stride):
            window = waves[start:start + duration_samples]
            rms = np.sqrt(np.mean(window ** 2))
            if rms > max_rms:
                max_rms = rms
                max_rms_start = start

        wave = waves[max_rms_start:max_rms_start + duration_samples]
        return wave

    def crop_or_pad(self,y, length, is_train=True, start=None):
        len_y = len(y)
        effective_length = self.duration * self.sr
        if len_y < effective_length:
            new_y = y.copy()
            while len(new_y) < effective_length:
                new_y = np.concatenate((new_y,y))
            y = new_y[0:effective_length]
        elif len_y > effective_length:
            start = np.random.randint(len_y - effective_length)
            y = y[start:start + effective_length].astype(np.float32)
        else:
            y = y.astype(np.float32)

        return y
    
    def crop_or_pad_rms(self,y, length, is_train=True, start=None,rms_prob=None):
        len_y = len(y)
        effective_length = self.duration * self.sr
        if len_y < effective_length:
            new_y = y.copy()
            while len(new_y) < effective_length:
                new_y = np.concatenate((new_y,y))
            y = new_y[0:effective_length]
        elif len_y > effective_length:
            choice_idx = np.random.choice(np.arange(0,len(rms_prob)),p=rms_prob)
            start = choice_idx * self.sr//2
            if start+effective_length//2 >= len_y:
                y = y[len_y-effective_length:len_y].astype(np.float32)
            elif start-effective_length//2 <= 0:
                y = y[0:effective_length].astype(np.float32)
            else:
                y = y[start - effective_length//2 : start + effective_length//2]
            # start = np.random.randint(len_y - effective_length)
            # y = y[start:start + effective_length].astype(np.float32)
        else:
            y = y.astype(np.float32)

        return y
    
    def get_first_duration(self,y):
        len_y = len(y)
        effective_length = self.duration * self.sr
        if len_y < effective_length:
            new_y = y.copy()
            while len(new_y) < effective_length:
                new_y = np.concatenate((new_y,y))
            y = new_y[0:effective_length]
        elif len_y > effective_length:
            start = 0
            y = y[start:start + effective_length].astype(np.float32)
        else:
            y = y.astype(np.float32)

        return y

    def __getitem__(self, idx):
        row = self.df.iloc[idx].copy()
        audio,orig_sr = sf.read(row['path'].replace('.mp3','.ogg'))
        ogg_name = row['path'].split('/')[-1]
        if ogg_name in self.hv_record:
            delete_index = self.hv_record[ogg_name]
            delete_index = delete_index[np.where(delete_index<len(audio))]
            audio = np.delete(audio,delete_index)
            if len(audio) <= 32000:
                audio,orig_sr = sf.read(row['path'].replace('.mp3','.ogg'))
        if len(audio.shape)==2:
            audio = audio[:,0]
        if orig_sr!=self.sr:
            audio = lb.resample(audio, orig_sr=orig_sr, target_sr=self.sr, res_type="kaiser_fast")

        if self.train:
            if row['secondary_labels'] != "['']" and row['secondary_labels'] != '[]' and type(row['secondary_labels']) != float:
                labellist = row['secondary_labels'].replace('[','').replace(']','').replace('\'','').split(', ')
                for j in labellist:
                    row[j] = 1.0
            audio = self.crop_or_pad(audio,self.audio_length,self.train) # constant length (l,) array
            # audio = self.pick_rms(audio,duration_samples=CFG.duration*CFG.sample_rate)
            if self.audio_augmentations:
                audio = self.audio_augmentations(audio)
            if row['rating']!=0:
                if np.isnan(row['rating']):
                    weight = 1
                elif type(row['rating'])==np.dtype(np.float64):
                    weight = row['rating']/5
                else:
                    weight = 1
            else:
                weight = 1
            # weight = None
        else:
            if row['secondary_labels'] != "['']" and row['secondary_labels'] != '[]' and type(row['secondary_labels']) != float:
                labellist = row['secondary_labels'].replace('[','').replace(']','').replace('\'','').split(', ')
                for j in labellist:
                    row[j] = 1.0

            audio = self.get_first_duration(audio)
            if row['rating']!=0:
                if np.isnan(row['rating']):
                    weight = 1
                elif type(row['rating'])==np.dtype(np.float64):
                    weight = row['rating']/5
                else:
                    weight = 1
            else:
                weight = 1
        return torch.from_numpy(audio),torch.tensor(row[self.bird_cols]).float(),torch.tensor(weight).float()

class BirdDatasetSpec(BirdDatasetTwoLabel):
    def __init__(self, df, bird_cols=None, sr=CFG.sample_rate, duration=CFG.duration, audio_augmentations=None, train=True,image_augmentations=None):
        super().__init__(df, bird_cols, sr, duration, audio_augmentations, train)
        self.spectrogram_extractor = MelSpectrogram(
            sample_rate=CFG.sample_rate,
            n_fft=CFG.n_fft,
            win_length=CFG.window_size,
            hop_length=CFG.hop_size,
            f_min=CFG.fmin,
            f_max=CFG.fmax,
            pad=0,
            n_mels=CFG.mel_bins,
            power=2,
            normalized=False,
        )
        # Logmel feature extractor
        self.logmel_extractor = AmplitudeToDB(top_db=None)
        self.normlize = NormalizeMelSpec()
        self.image_augmentations = image_augmentations
    def __getitem__(self, idx):
        row = self.df.iloc[idx].copy()
        audio,orig_sr = sf.read(row['path'].replace('.mp3','.ogg'))
        ogg_name = row['path'].split('/')[-1]
        if ogg_name in self.hv_record:
            delete_index = self.hv_record[ogg_name]
            delete_index = delete_index[np.where(delete_index<len(audio))]
            audio = np.delete(audio,delete_index)
        if len(audio.shape)==2:
            audio = audio[:,0]
        if orig_sr!=self.sr:
            audio = lb.resample(audio, orig_sr=orig_sr, target_sr=self.sr, res_type="kaiser_fast")

        if self.train:
            if row['secondary_labels'] != "['']" and row['secondary_labels'] != '[]' and type(row['secondary_labels']) != float:
                labellist = row['secondary_labels'].replace('[','').replace(']','').replace('\'','').split(', ')
                for j in labellist:
                    row[j] = 1.0
            audio = self.crop_or_pad(audio,self.audio_length,self.train) # constant length (l,) array
            if self.audio_augmentations:
                audio = self.audio_augmentations(audio)
            if row['rating']!=0:
                if np.isnan(row['rating']):
                    weight = 1
                elif type(row['rating'])==np.dtype(np.float64):
                    weight = row['rating']/5
                else:
                    weight = 1
            else:
                weight = 1
            # weight = None
        else:
            if row['secondary_labels'] != "['']" and row['secondary_labels'] != '[]' and type(row['secondary_labels']) != float:
                labellist = row['secondary_labels'].replace('[','').replace(']','').replace('\'','').split(', ')
                for j in labellist:
                    row[j] = 1.0

            audio = self.get_first_duration(audio)
            if row['rating']!=0:
                if np.isnan(row['rating']):
                    weight = 1
                elif type(row['rating'])==np.dtype(np.float64):
                    weight = row['rating']/5
                else:
                    weight = 1
            else:
                weight = 1
        audios,label,weights = torch.from_numpy(audio),torch.tensor(row[self.bird_cols]).float(),torch.tensor(weight).float()
        audios = audios.to(torch.float32)
        images = self.spectrogram_extractor(audios.unsqueeze(0)) # (batch_size,freq_bins time_steps)
        images = self.logmel_extractor(images) 
        images = self.normlize(images)
        images = images.permute(0,2,1)# (batch_size,time_steps, mel_bins)
        images = images.unsqueeze(1)
        images = torch.repeat_interleave(images,repeats=3,dim=1)

        if self.image_augmentations:
            images = self.image_augmentations(images)
        return images.squeeze(0),label,weights


class BirdDatasetSplitTrain(BirdDatasetTwoLabel):
    def __init__(self, df, sr=CFG.sample_rate, duration=CFG.duration, audio_augmentations=None, train=True):
        super().__init__(df, sr, duration, audio_augmentations, train)

    def __getitem__(self, idx):
        row = self.df.iloc[idx].copy()
        audio,orig_sr = sf.read(row['path'].replace('.mp3','.ogg'))
        if len(audio.shape)==2:
            audio = audio[:,0]
        if orig_sr!=self.sr:
            audio = lb.resample(audio, orig_sr=orig_sr, target_sr=self.sr, res_type="kaiser_fast")

        if self.train:
            audio = self.crop_or_pad(audio,self.audio_length,self.train) # constant length (l,) array
            if self.audio_augmentations:
                audio = self.audio_augmentations(audio)
            if row['rating']!=0:
                if np.isnan(row['rating']):
                    weight = 1
                elif type(row['rating'])==np.dtype(np.float64):
                    weight = row['rating']/5
                else:
                    weight = 1
            else:
                weight = 1
            # weight = None
        else:
            audio = self.get_first_duration(audio)
            if row['rating']!=0:
                if np.isnan(row['rating']):
                    weight = 1
                elif type(row['rating'])==np.dtype(np.float64):
                    weight = row['rating']/5
                else:
                    weight = 1
            else:
                weight = 1
        return torch.from_numpy(audio),torch.tensor(row[14:]).float(),torch.tensor(weight).float()


class BirdDatasetWithPseudoLabel(BirdDatasetTwoLabel):
    def __init__(self, df, pesudo_df = None, bird_cols=None, sr=CFG.sample_rate, duration=CFG.duration, audio_augmentations=None, train=True):
        super().__init__(df, bird_cols, sr, duration, audio_augmentations, train)
        self.pseudo_df = pesudo_df
    
    def __add_noise(self, tgt_wav, add_wav, db):
        tgt_rms = np.sqrt(np.mean(np.square(tgt_wav), axis=-1))
        add_rms = np.sqrt(np.mean(np.square(add_wav), axis=-1))

        noise_rms = tgt_rms / (10**(float(db) / 20)) 
        new_wav = tgt_wav + add_wav * (noise_rms / (add_rms + 1e-6))
        new_wav = np.clip(new_wav, np.min(new_wav)*2, np.max(new_wav)*2)
        return new_wav

    def __getitem__(self, idx):
        row = self.df.iloc[idx].copy()
        audio,orig_sr = sf.read(row['path'].replace('.mp3','.ogg'))
        ogg_name = row['path'].split('/')[-1]
        if ogg_name in self.hv_record:
            delete_index = self.hv_record[ogg_name]
            delete_index = delete_index[np.where(delete_index<len(audio))]
            audio = np.delete(audio,delete_index)
        if len(audio.shape)==2:
            audio = audio[:,0]
        if orig_sr!=self.sr:
            audio = lb.resample(audio, orig_sr=orig_sr, target_sr=self.sr, res_type="kaiser_fast")

        if self.train:
            if row['secondary_labels'] != "['']" and row['secondary_labels'] != '[]' and type(row['secondary_labels']) != float:
                labellist = row['secondary_labels'].replace('[','').replace(']','').replace('\'','').split(', ')
                for j in labellist:
                    row[j] = 1.0
            audio = self.crop_or_pad(audio,self.audio_length,self.train) # constant length (l,) array
            if self.audio_augmentations:
                audio = self.audio_augmentations(audio)
            if row['rating']!=0:
                if np.isnan(row['rating']):
                    weight = 1
                elif type(row['rating'])==np.dtype(np.float64):
                    weight = row['rating']/5
                else:
                    weight = 1
            else:
                weight = 1
            # weight = None
        else:
            audio = self.get_first_duration(audio)
            if row['rating']!=0:
                if np.isnan(row['rating']):
                    weight = 1
                elif type(row['rating'])==np.dtype(np.float64):
                    weight = row['rating']/5
                else:
                    weight = 1
            else:
                weight = 1
        
        if np.random.uniform(0,1) < 0.5:
            random_idx = np.random.randint(0,len(self.pseudo_df))
            pesudo_file = self.pseudo_df.iloc[random_idx]['path']
            pesudo_label = self.pseudo_df.iloc[random_idx][self.bird_cols]
            pesudo_audio,_ = sf.read(pesudo_file)
            db = np.random.uniform(20,40)
            audio = self.__add_noise(pesudo_audio,audio,db)
            label = np.array(row[self.bird_cols]) + np.array(pesudo_label)
            label = label.astype(np.float32)
            label = np.clip(label,0,1)
            audio,label,weights = torch.from_numpy(audio),torch.from_numpy(label).float(),torch.tensor(weight).float()
        else:
            label = np.array(row[self.bird_cols]).astype(np.float32)
            audio,label,weights = torch.from_numpy(audio),torch.from_numpy(label).float(),torch.tensor(weight).float()
        return audio,label,weights

def fetch_scheduler(optimizer,steps_per_epoch,epochs=CFG.epochs):
    if CFG.scheduler == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer,T_max=epochs*steps_per_epoch/CFG.n_accumulate, 
                                                   eta_min=CFG.min_lr)
    elif CFG.scheduler == 'CosineAnnealingWarmRestarts':
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=CFG.T_0, 
                                                             eta_min=CFG.min_lr)
    elif CFG.scheduler == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,
                                                   mode='min',
                                                   factor=0.1,
                                                   patience=7,
                                                   threshold=0.0001,
                                                   min_lr=CFG.min_lr,)
    elif CFG.scheduer == 'ExponentialLR':
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.75)
    elif CFG.scheduler == None:
        return None
        
    return scheduler

if __name__ == '__main__':
    import pandas as pd
    df = pd.read_csv('/root/projects/BirdClef2025/data/train.csv')
    df['path'] = CFG.data_root+df['filename']
    df = pd.concat([df, pd.get_dummies(df['primary_label'])], axis=1)
    
    df = upsample_data(df,thr=50)
    df = downsample_data(df,thr=400)

    pesudo_df = pd.read_csv('/root/projects/BirdClef2025/BirdCLEF2023-30th-place-solution-master/usefulFunc/pesudo_labelv12_ensemble.csv')
    birds_cols = list(pd.get_dummies(df['primary_label']).columns)

    image_augmentations = T.Compose(
        [
            T.Resize(CFG.img_size),
            T.RandomVerticalFlip(p=0.5)
        ]
    )

    dataset = BirdDatasetSpec(df, birds_cols, sr=CFG.sample_rate, duration=CFG.duration, train=True,image_augmentations=image_augmentations)
    data1 = dataset.__getitem__(10)
    print(1)