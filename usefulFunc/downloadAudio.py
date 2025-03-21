import numpy as np
import requests
from bs4 import BeautifulSoup
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import re
from glob import glob

def urldownload(url,filename=None):
    """
    下载文件到指定目录
    :param url: 文件下载的url
    :param filename: 要存放的目录及文件名，例如：./test.xls
    :return:
    """
    resp = requests.get(url, stream=True,timeout=5)
    # 拿到文件的长度，并把total初始化为0
    total = int(resp.headers.get('content-length', 0))
    with open(filename, 'wb') as file, tqdm(
        desc=filename,
        total=total,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)

df = pd.read_csv("/home/data/lijw/dataset/BirdCLEF/train_metadata.csv")

name_list = list(df['primary_label'].value_counts()._stat_axis)
value_list = list(df['primary_label'].value_counts())

LowNumName = []
for i in range(len(value_list)):
    if value_list[i]<20:
        LowNumName.append(name_list[i])

eBirdName = []
for i in range(len(LowNumName)):
    nameTemp = df[df['primary_label'] == LowNumName[i]]['scientific_name'].unique().item()
    nameTemp = nameTemp.replace(' ','-')
    eBirdName.append(nameTemp)

plist = []
res = requests.get('https://xeno-canto.org/collection/species/all')
content = BeautifulSoup(res.text,"html.parser")
datas = content.find_all(href=re.compile("species"))
for data in datas:
    http_temp = data.get('href')
    plist.append(http_temp)

filteredPlist = []
for i in eBirdName:
    for j in plist:
        if i in j:
            filteredPlist.append(j)

download_dict = {}
for i in filteredPlist:
    name = i.split('/')[-1]
    resTemp = requests.get(i)
    contentTemp = BeautifulSoup(resTemp.text,"html.parser")
    datasTemp = contentTemp.find_all(href=re.compile("/download"))
    http_list = []
    for data in datasTemp:
        http_list.append(data.get('href'))
    download_dict[name] = http_list.copy()

targetPath = '/home/data/lijw/dataset/downloadAudios/'
for key,value in download_dict.items():
    for url in value:
        nameTemp = url.split('/')[-2]
        if not os.path.exists(targetPath+key):
            os.mkdir(targetPath+key)
        urldownload(url,targetPath+key+'/XC'+nameTemp+'.ogg')