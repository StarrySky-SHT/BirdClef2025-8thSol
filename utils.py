import numpy as np
import pandas as pd 
from glob import glob
from config import  CFG 
import random
import colorednoise as cn
import librosa
import functools
import random
import warnings
from pathlib import Path
from typing import Optional, List, Callable, Union
import math
import numpy as np
import os
import torch
from torch.distributions import Beta
import torch.nn as nn
from audiomentations.core.transforms_interface import BaseWaveformTransform
from audiomentations.core.utils import (
    calculate_desired_noise_rms,
    calculate_rms,
    convert_decibels_to_amplitude_ratio,
    find_audio_files_in_paths,
)
import torch.nn.functional as F
from tqdm import tqdm

def sumix(waves: torch.Tensor, labels: torch.Tensor, max_percent: float = 1.0, min_percent: float = 0.3):
    batch_size = len(labels)
    perm = torch.randperm(batch_size)
    coeffs_1 = torch.rand(batch_size, device=waves.device).view(-1, 1) * (
        max_percent  - min_percent
    ) + min_percent
    coeffs_2 = torch.rand(batch_size, device=waves.device).view(-1, 1) * (
        max_percent  - min_percent
    ) + min_percent
    label_coeffs_1 = torch.where(coeffs_1 >= 0.5, 1, 1 - 2 * (0.5 - coeffs_1))
    label_coeffs_2 = torch.where(coeffs_2 >= 0.5, 1, 1 - 2 * (0.5 - coeffs_2))
    labels = label_coeffs_1 * labels + label_coeffs_2 * labels[perm]

    waves = coeffs_1 * waves + coeffs_2 * waves[perm]
    return {
        "waves": waves,
        "labels": torch.clip(labels, 0, 1)
    }

def random_power(images, power = 1.5, c= 0.7):
    images = images - images.min()
    images = images/(images.max()+0.0000001)
    images = images**(random.random()*power + c)
    return images

class Mixup(nn.Module):
    def __init__(self, mix_beta):

        super(Mixup, self).__init__()
        self.beta_distribution = Beta(mix_beta, mix_beta)

    def forward(self, X, Y, weight=None):

        bs = X.shape[0]
        n_dims = len(X.shape)
        perm = torch.randperm(bs)
        coeffs = self.beta_distribution.rsample(torch.Size((bs,))).to(X.device)

        if n_dims == 2:
            X = coeffs.view(-1, 1) * X + (1 - coeffs.view(-1, 1)) * X[perm]
        elif n_dims == 3:
            X = coeffs.view(-1, 1, 1) * X + (1 - coeffs.view(-1, 1, 1)) * X[perm]
        else:
            X = coeffs.view(-1, 1, 1, 1) * X + (1 - coeffs.view(-1, 1, 1, 1)) * X[perm]

        Y = coeffs.view(-1, 1) * Y + (1 - coeffs.view(-1, 1)) * Y[perm]

        if weight is None:
            return X, Y
        else:
            weight = coeffs.view(-1) * weight + (1 - coeffs.view(-1)) * weight[perm]
            return X, Y, weight

class Compose:
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, y: np.ndarray, sr=CFG.sample_rate):
        for trns in self.transforms:
            y = trns(y, sr)
        return y


class AudioTransform:
    def __init__(self, always_apply=False, p=0.5):
        self.always_apply = always_apply
        self.p = p

    def __call__(self, y: np.ndarray, sr):
        if self.always_apply:
            return self.apply(y, sr=sr)
        else:
            if np.random.rand() < self.p:
                return self.apply(y, sr=sr)
            else:
                return y

    def apply(self, y: np.ndarray, **params):
        raise NotImplementedError

class Random_Power:
    def __init__(self,power_value=3,p=0.5):
        self.power_value = power_value
        self.p = p

    def apply(self,y: np.ndarray):
        power_value = np.random.uniform(0,self.power_value)
        y = np.power(y,power_value)
        return y

class OneOf(Compose):
    # https://github.com/albumentations-team/albumentations/blob/master/albumentations/core/composition.py
    def __init__(self, transforms, p=0.5):
        super().__init__(transforms)
        self.p = p
        transforms_ps = [t.p for t in transforms]
        s = sum(transforms_ps)
        self.transforms_ps = [t / s for t in transforms_ps]

    def __call__(self, y: np.ndarray, sr):
        data = y
        if self.transforms_ps and (random.random() < self.p):
            random_state = np.random.RandomState(random.randint(0, 2 ** 32 - 1))
            t = random_state.choice(self.transforms, p=self.transforms_ps)
            data = t(y, sr)
        return data


class Normalize(AudioTransform):
    def __init__(self, always_apply=False, p=1):
        super().__init__(always_apply, p)

    def apply(self, y: np.ndarray, **params):
        max_vol = np.abs(y).max()
        y_vol = y * 1 / (max_vol+1e-9)
        return np.asfortranarray(y_vol)


class NewNormalize(AudioTransform):
    def __init__(self, always_apply=False, p=1):
        super().__init__(always_apply, p)

    def apply(self, y: np.ndarray, **params):
        y_mm = y - y.mean()
        return y_mm / y_mm.abs().max()


class NoiseInjection(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, max_noise_level=0.5):
        super().__init__(always_apply, p)

        self.noise_level = (0.0, max_noise_level)

    def apply(self, y: np.ndarray, **params):
        noise_level = np.random.uniform(*self.noise_level)
        noise = np.random.randn(len(y))
        augmented = (y + noise * noise_level).astype(y.dtype)
        return augmented


class GaussianNoise(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, min_snr=5, max_snr=20):
        super().__init__(always_apply, p)

        self.min_snr = min_snr
        self.max_snr = max_snr

    def apply(self, y: np.ndarray, **params):
        snr = np.random.uniform(self.min_snr, self.max_snr)
        a_signal = np.sqrt(y ** 2).max()
        a_noise = a_signal / (10 ** (snr / 20))

        white_noise = np.random.randn(len(y))
        a_white = np.sqrt(white_noise ** 2).max()
        augmented = (y + white_noise * 1 / a_white * a_noise).astype(y.dtype)
        return augmented


class PinkNoise(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, min_snr=5, max_snr=20):
        super().__init__(always_apply, p)

        self.min_snr = min_snr
        self.max_snr = max_snr

    def apply(self, y: np.ndarray, **params):
        snr = np.random.uniform(self.min_snr, self.max_snr)
        a_signal = np.sqrt(y ** 2).max()
        a_noise = a_signal / (10 ** (snr / 20))

        pink_noise = cn.powerlaw_psd_gaussian(1, len(y))
        a_pink = np.sqrt(pink_noise ** 2).max()
        augmented = (y + pink_noise * 1 / a_pink * a_noise).astype(y.dtype)
        return augmented


class PitchShift(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, max_range=5):
        super().__init__(always_apply, p)
        self.max_range = max_range

    def apply(self, y: np.ndarray, sr, **params):
        n_steps = np.random.randint(-self.max_range, self.max_range)
        augmented = librosa.effects.pitch_shift(y, sr, n_steps)
        return augmented


class TimeStretch(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, max_rate=1):
        super().__init__(always_apply, p)
        self.max_rate = max_rate

    def apply(self, y: np.ndarray, **params):
        rate = np.random.uniform(0, self.max_rate)
        augmented = librosa.effects.time_stretch(y, rate)
        return augmented


def _db2float(db: float, amplitude=True):
    if amplitude:
        return 10 ** (db / 20)
    else:
        return 10 ** (db / 10)


def volume_down(y: np.ndarray, db: float):
    """
    Low level API for decreasing the volume
    Parameters
    ----------
    y: numpy.ndarray
        stereo / monaural input audio
    db: float
        how much decibel to decrease
    Returns
    -------
    applied: numpy.ndarray
        audio with decreased volume
    """
    applied = y * _db2float(-db)
    return applied


def volume_up(y: np.ndarray, db: float):
    """
    Low level API for increasing the volume
    Parameters
    ----------
    y: numpy.ndarray
        stereo / monaural input audio
    db: float
        how much decibel to increase
    Returns
    -------
    applied: numpy.ndarray
        audio with increased volume
    """
    applied = y * _db2float(db)
    return applied


class RandomVolume(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, limit=10):
        super().__init__(always_apply, p)
        self.limit = limit

    def apply(self, y: np.ndarray, **params):
        db = np.random.uniform(-self.limit, self.limit)
        if db >= 0:
            return volume_up(y, db)
        else:
            return volume_down(y, db)


class CosineVolume(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, limit=10):
        super().__init__(always_apply, p)
        self.limit = limit

    def apply(self, y: np.ndarray, **params):
        db = np.random.uniform(-self.limit, self.limit)
        cosine = np.cos(np.arange(len(y)) / len(y) * np.pi * 2)
        dbs = _db2float(cosine * db)
        return y * dbs

def filter_data(df, thr=5):
    # Count the number of samples for each class
    counts = df.primary_label.value_counts()

    # Condition that selects classes with less than `thr` samples
    cond = df.primary_label.isin(counts[counts<thr].index.tolist())

    # Add a new column to select samples for cross validation
    df['cv'] = True

    # Set cv = False for those class where there is samples less than thr
    df.loc[cond, 'cv'] = False

    # Return the filtered dataframe
    return df
    
def upsample_data(df, thr=20):
    # get the class distribution
    class_dist = df['primary_label'].value_counts()

    # identify the classes that have less than the threshold number of samples
    down_classes = class_dist[class_dist < thr].index.tolist()

    # create an empty list to store the upsampled dataframes
    up_dfs = []

    # loop through the undersampled classes and upsample them
    for c in down_classes:
        # get the dataframe for the current class
        class_df = df.query("primary_label==@c")
        # find number of samples to add
        num_up = thr - class_df.shape[0]
        # upsample the dataframe
        class_df = class_df.sample(n=num_up, replace=True, random_state=CFG.seed)
        # append the upsampled dataframe to the list
        up_dfs.append(class_df)

    # concatenate the upsampled dataframes and the original dataframe
    up_df = pd.concat([df] + up_dfs, axis=0, ignore_index=True)
    
    return up_df

def downsample_data(df, thr=500):
    # get the class distribution
    class_dist = df['primary_label'].value_counts()
    
    # identify the classes that have less than the threshold number of samples
    up_classes = class_dist[class_dist > thr].index.tolist()

    # create an empty list to store the upsampled dataframes
    down_dfs = []

    # loop through the undersampled classes and upsample them
    for c in up_classes:
        # get the dataframe for the current class
        class_df = df.query("primary_label==@c")
        # Remove that class data
        df = df.query("primary_label!=@c")
        # upsample the dataframe
        class_df = class_df.sample(n=thr, replace=False, random_state=CFG.seed)
        # append the upsampled dataframe to the list
        down_dfs.append(class_df)

    # concatenate the upsampled dataframes and the original dataframe
    down_df = pd.concat([df] + down_dfs, axis=0, ignore_index=True)
    
    return down_df

def instance_mixup(x,alpha=0.2,use_instance_mixup=False,instance_mixup_pro=0.5):
    if use_instance_mixup and (np.random.uniform(0,1)>instance_mixup_pro):
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        x = torch.stack(torch.chunk(x,chunks=4,dim=2))
        rand_index = torch.randperm(x.size(0))
        mixed_x = lam * x + (1 - lam) * x[rand_index,...]
        mixed_x = torch.cat(torch.chunk(mixed_x,chunks=4,dim=0),dim=3)[0,...]
    else:
        mixed_x = torch.cat(torch.chunk(x,chunks=4,dim=2),dim=2)
    return mixed_x

def mixup_data(x, y, alpha=0.5,use_mixup=True):
    """Returns mixed inputs, pairs of targets, and lambda
    """
    if not use_mixup:
        return x,y,0
    if np.random.uniform(0,1)<=1.0:
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
    else:
        return x,y,0

    mixed_x = lam * x + (1 - lam) * x.flip(dims=(0, ))
    y_a, y_b = y, y.flip(dims=(0, ))
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam, classes_weight=None):
    if classes_weight is not None:
        return lam * criterion(pred, y_a,classes_weight) + (1 - lam) * criterion(pred, y_b,classes_weight)
    else:
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def mixup_criterion_clef(criterion, pred, y_a, y_b, lam, classes_weight=None):
    if classes_weight is not None:
        return lam * criterion(pred, y_a,classes_weight) + (1 - lam) * criterion(pred, y_b,classes_weight)
    else:
        loss1 = (1 - lam) * criterion(pred, y_b[:y_b.shape[0]//4])
        loss2 = (1 - lam) * criterion(pred, y_b[y_b.shape[0]//4:2*y_b.shape[0]//4])
        loss3 = (1 - lam) * criterion(pred, y_b[2*y_b.shape[0]//4:3*y_b.shape[0]//4])
        loss4 = (1 - lam) * criterion(pred, y_b[3*y_b.shape[0]//4:])
        loss = (loss1+loss2+loss3+loss4)/4
        return lam * criterion(pred, y_a) + loss

SUPPORTED_EXTENSIONS = (
".npy"
)

def sample_one_per_group(df: pd.DataFrame, group_column: str) -> tuple:
    """
    从指定列的每个类别中随机抽样1个元素，并合并成新的 DataFrame
    如果某个类别只有一个元素，则不进行抽样
    返回一个元组，包含：
    1. 抽样结果的 DataFrame
    2. 从原始 DataFrame 中删除了抽样结果后的新 DataFrame
    :param df: 原始 DataFrame
    :param group_column: 按此列进行分组的列名
    :return: tuple (抽样结果的 DataFrame, 删除抽样结果后的原始 DataFrame)
    """
    # 按指定列进行分组
    grouped = df.groupby(group_column)
  
    # 对每个组进行抽样，如果组内元素数量大于1，则抽样1个；否则不抽样
    sampled_dfs = []
    sampled_indices = []  # 用于记录抽样的索引
  
    for name, group in grouped:
        if len(group) > 1:
            sample = group.sample(n=1)
            sampled_dfs.append(sample)
            sampled_indices.extend(sample.index.tolist())  # 记录抽样的行索引
  
    # 合并所有抽样结果
    if sampled_dfs:
        result_df = pd.concat(sampled_dfs, ignore_index=True)
    else:
        result_df = pd.DataFrame()  # 如果没有符合条件的抽样结果，返回空 DataFrame
  
    # 从原始 DataFrame 中删除抽样结果
    if sampled_indices:
        remaining_df = df.drop(sampled_indices)
    else:
        remaining_df = df.copy()  # 如果没有抽样结果，返回原始 DataFrame 的副本
  
    return result_df, remaining_df

def find_audio_files(
    root_path,
    filename_endings=SUPPORTED_EXTENSIONS,
    traverse_subdirectories=True,
    follow_symlinks=True,
):
    """Return a list of paths to all audio files with the given extension(s) in a directory.
    Also traverses subdirectories by default.
    """
    file_paths = []

    for root, dirs, filenames in os.walk(root_path, followlinks=follow_symlinks):
        filenames = sorted(filenames)
        for filename in filenames:
            input_path = os.path.abspath(root)
            file_path = os.path.join(input_path, filename)

            if filename.lower().endswith(filename_endings):
                file_paths.append(Path(file_path))
        if not traverse_subdirectories:
            # prevent descending into subfolders
            break

    return file_paths


def find_audio_files_in_paths(
    paths: Union[List[Path], List[str], Path, str],
    filename_endings=SUPPORTED_EXTENSIONS,
    traverse_subdirectories=True,
    follow_symlinks=True,
):
    """Return a list of paths to all audio files with the given extension(s) contained in the list or in its directories.
    Also traverses subdirectories by default.
    """

    file_paths = []

    if isinstance(paths, (list, tuple, set)):
        paths = list(paths)
    else:
        paths = [paths]

    for p in paths:
        if str(p).lower().endswith(SUPPORTED_EXTENSIONS):
            file_path = Path(os.path.abspath(p))
            file_paths.append(file_path)
        elif os.path.isdir(p):
            file_paths += find_audio_files(
                p,
                filename_endings=filename_endings,
                traverse_subdirectories=traverse_subdirectories,
                follow_symlinks=follow_symlinks,
            )
    return file_paths

class AddBackgroundNoise(BaseWaveformTransform):
    """Mix in another sound, e.g. a background noise. Useful if your original sound is clean and
    you want to simulate an environment where background noise is present.
    Can also be used for mixup, as in https://arxiv.org/pdf/1710.09412.pdf
    A folder of (background noise) sounds to be mixed in must be specified. These sounds should
    ideally be at least as long as the input sounds to be transformed. Otherwise, the background
    sound will be repeated, which may sound unnatural.
    Note that the gain of the added noise is relative to the amount of signal in the input if the parameter noise_rms
    is set to "relative" (default option). This implies that if the input is completely silent, no noise will be added.
    Here are some examples of datasets that can be downloaded and used as background noise:
    * https://github.com/karolpiczak/ESC-50#download
    * https://github.com/microsoft/DNS-Challenge/
    """

    def __init__(
        self,
        sounds_path: Union[List[Path], List[str], Path, str],
        min_snr_in_db: float = 3.0,
        max_snr_in_db: float = 30.0,
        noise_rms: str = "relative",
        min_absolute_rms_in_db: float = -45.0,
        max_absolute_rms_in_db: float = -15.0,
        noise_transform: Optional[Callable[[np.ndarray, int], np.ndarray]] = None,
        p: float = 0.5,
        lru_cache_size: int = 2,
    ):
        """
        :param sounds_path: A path or list of paths to audio file(s) and/or folder(s) with
            audio files. Can be str or Path instance(s). The audio files given here are
            supposed to be background noises.
        :param min_snr_in_db: Minimum signal-to-noise ratio in dB. Is only used if noise_rms is set to "relative"
        :param max_snr_in_db: Maximum signal-to-noise ratio in dB. Is only used if noise_rms is set to "relative"
        :param noise_rms: Defines how the background noise will be added to the audio input. If the chosen
            option is "relative", the RMS of the added noise will be proportional to the RMS of
            the input sound. If the chosen option is "absolute", the background noise will have
            a RMS independent of the RMS of the input audio file. The default option is "relative".
        :param min_absolute_rms_in_db: Is only used if noise_rms is set to "absolute". It is
            the minimum RMS value in dB that the added noise can take. The lower the RMS is,
            the lower the added sound will be.
        :param max_absolute_rms_in_db: Is only used if noise_rms is set to "absolute". It is
            the maximum RMS value in dB that the added noise can take. Note that this value
            can not exceed 0.
        :param noise_transform: A callable waveform transform (or composition of transforms) that
            gets applied to the noise before it gets mixed in. The callable is expected
            to input audio waveform (numpy array) and sample rate (int).
        :param p: The probability of applying this transform
        :param lru_cache_size: Maximum size of the LRU cache for storing noise files in memory
        """
        super().__init__(p)
        self.sound_file_paths = find_audio_files_in_paths(sounds_path)
        self.sound_file_paths = [str(p) for p in self.sound_file_paths]

        assert min_absolute_rms_in_db <= max_absolute_rms_in_db <= 0
        assert min_snr_in_db <= max_snr_in_db
        assert len(self.sound_file_paths) > 0

        self.noise_rms = noise_rms
        self.min_snr_in_db = min_snr_in_db
        self.min_absolute_rms_in_db = min_absolute_rms_in_db
        self.max_absolute_rms_in_db = max_absolute_rms_in_db
        self.max_snr_in_db = max_snr_in_db
        self._load_sound = functools.lru_cache(maxsize=lru_cache_size)(
            AddBackgroundNoise._load_sound
        )
        self.noise_transform = noise_transform

    def randomize_parameters(self, samples: np.ndarray, sample_rate: int):
        super().randomize_parameters(samples, sample_rate)
        if self.parameters["should_apply"]:
            self.parameters["snr_in_db"] = random.uniform(
                self.min_snr_in_db, self.max_snr_in_db
            )
            self.parameters["rms_in_db"] = random.uniform(
                self.min_absolute_rms_in_db, self.max_absolute_rms_in_db
            )
            self.parameters["noise_file_path"] = random.choice(self.sound_file_paths)

            num_samples = len(samples)
            noise_sound = self._load_sound(
                self.parameters["noise_file_path"], sample_rate
            )

            num_noise_samples = len(noise_sound)
            min_noise_offset = 0
            max_noise_offset = max(0, num_noise_samples - num_samples - 1)
            self.parameters["noise_start_index"] = random.randint(
                min_noise_offset, max_noise_offset
            )
            self.parameters["noise_end_index"] = (
                self.parameters["noise_start_index"] + num_samples
            )

    @staticmethod
    def _load_sound(file_path, sample_rate):
        return np.load(file_path)
    
    def apply(self, samples: np.ndarray, sample_rate: int):
        noise_sound = self._load_sound(
            self.parameters["noise_file_path"], sample_rate
        )
        noise_sound = noise_sound[
            self.parameters["noise_start_index"] : self.parameters["noise_end_index"]
        ]

        if self.noise_transform:
            noise_sound = self.noise_transform(noise_sound, sample_rate)

        noise_rms = calculate_rms(noise_sound)
        if noise_rms < 1e-9:
            warnings.warn(
                "The file {} is too silent to be added as noise. Returning the input"
                " unchanged.".format(self.parameters["noise_file_path"])
            )
            return samples

        clean_rms = calculate_rms(samples)

        if self.noise_rms == "relative":
            desired_noise_rms = calculate_desired_noise_rms(
                clean_rms, self.parameters["snr_in_db"]
            )

            # Adjust the noise to match the desired noise RMS
            noise_sound = noise_sound * (desired_noise_rms / noise_rms)

        if self.noise_rms == "absolute":
            desired_noise_rms_db = self.parameters["rms_in_db"]
            desired_noise_rms_amp = convert_decibels_to_amplitude_ratio(
                desired_noise_rms_db
            )
            gain = desired_noise_rms_amp / noise_rms
            noise_sound = noise_sound * gain

        # Repeat the sound if it shorter than the input sound
        num_samples = len(samples)
        while len(noise_sound) < num_samples:
            noise_sound = np.concatenate((noise_sound, noise_sound))

        if len(noise_sound) > num_samples:
            noise_sound = noise_sound[0:num_samples]

        # Return a mix of the input sound and the background noise sound
        return samples + noise_sound

    def __getstate__(self):
        state = self.__dict__.copy()
        warnings.warn(
            "Warning: the LRU cache of AddBackgroundNoise gets discarded when pickling it."
            " E.g. this means the cache will not be used when using AddBackgroundNoise together"
            " with multiprocessing on Windows"
        )
        del state["_load_sound"]
        return state

@torch.no_grad()
def update_bn(loader, model, origin_model, device=None):
    r"""Updates BatchNorm running_mean, running_var buffers in the model.

    It performs one pass over data in `loader` to estimate the activation
    statistics for BatchNorm layers in the model.
    Args:
        loader (torch.utils.data.DataLoader): dataset loader to compute the
            activation statistics on. Each data batch should be either a
            tensor, or a list/tuple whose first element is a tensor
            containing data.
        model (torch.nn.Module): model for which we seek to update BatchNorm
            statistics.
        device (torch.device, optional): If set, data will be transferred to
            :attr:`device` before being passed into :attr:`model`.

    Example:
        >>> loader, model = ...
        >>> torch.optim.swa_utils.update_bn(loader, model)

    .. note::
        The `update_bn` utility assumes that each data batch in :attr:`loader`
        is either a tensor or a list or tuple of tensors; in the latter case it
        is assumed that :meth:`model.forward()` should be called on the first
        element of the list or tuple corresponding to the data batch.
    """
    momenta = {}
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.running_mean = torch.zeros_like(module.running_mean)
            module.running_var = torch.ones_like(module.running_var)
            momenta[module] = module.momentum

    if not momenta:
        return

    was_training = model.training
    model.train()
    for module in momenta.keys():
        module.momentum = None
        module.num_batches_tracked *= 0

    for idx,input in enumerate(tqdm(loader)):
        if isinstance(input, (list, tuple)):
            input = input[0]
            input = input.to(torch.float32).to(CFG.device)
            input = origin_model.get_mel_gram(input)
            input = torch.repeat_interleave(input.unsqueeze(1),repeats=3,dim=1)
        if device is not None:
            input = input.to(device)

        model(input)

    for bn_module in momenta.keys():
        bn_module.momentum = momenta[bn_module]
    model.train(was_training)