import pandas as pd
import numpy as np

df_train = pd.read_csv("/home/data/lijw/dataset/BirdCLEF/specs/train.csv")
df_valid = pd.read_csv("/home/data/lijw/dataset/BirdCLEF/specs/valid.csv")

Taxonomydf = pd.read_csv('/home/data/lijw/dataset/BirdCLEF/eBird_Taxonomy_v2021.csv')

name_list = list(df_train['primary_label'].value_counts()._stat_axis)
value_list = list(df_train['primary_label'].value_counts())

LowNumName = []
for i in range(len(value_list)):
    if value_list[i]<10:
        LowNumName.append(name_list[i])

eBirdName = []
for i in range(len(LowNumName)):
    eBirdName.append(df_train[df_train['primary_label'] == LowNumName[i]]['scientific_name'].unique().item())


