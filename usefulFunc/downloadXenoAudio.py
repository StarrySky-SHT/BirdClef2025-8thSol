import numpy as np
import requests
from bs4 import BeautifulSoup
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import re
from glob import glob
from tqdm import tqdm

def urldownload(url,filename=None):
    """
    下载文件到指定目录
    :param url: 文件下载的url
    :param filename: 要存放的目录及文件名，例如：./test.xls
    :return:
    """
    resp = requests.get(url, stream=True,timeout=10)
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

df = pd.read_csv("/root/projects/BirdClef2025/data/train.csv")

name_list = list(df['primary_label'].value_counts().index)
value_list = list(df['primary_label'].value_counts())

LowNumName = []
for i in range(len(value_list)):
    if value_list[i]<50:
        LowNumName.append(name_list[i])

eBirdNames = []
for i in range(len(LowNumName)):
    nameTemp = df[df['primary_label'] == LowNumName[i]]['scientific_name'].unique().item()
    nameTemp = nameTemp.replace(' ','-')
    eBirdNames.append(nameTemp)

df['single_filename'] = df['filename'].map(lambda x:x.split('/')[1].replace('.ogg',''))
download_dict = {}
all_url_list = []
for idx,eBirdName in tqdm(enumerate(eBirdNames),total=len(eBirdNames)):
    name = LowNumName[idx]
    res = requests.get(f'https://xeno-canto.org/species/{eBirdName}')
    content = BeautifulSoup(res.text,"html.parser")
    datasTemp = content.find_all(href=re.compile("/download"))
    http_list = []
    for data in datasTemp:
        thisfilename = datasTemp[0].contents[0].attrs['title'].split('Download file')[1].split(' - ')[0].replace(' \'','')
        # if thisfilename not in list(df['single_filename']):
        if data.get('href').replace('/download','') not in list(df['url']):
            http_list.append(data.get('href'))
            all_url_list.append(data)
    download_dict[name] = http_list.copy()

error_list = []
targetPath = '/root/projects/BirdClef2025/externaldata/download_xc_data/'
for key,value in tqdm(download_dict.items(),total=len(download_dict)):
    for url in value:
        nameTemp = url.split('/')[-2]
        if not os.path.exists(targetPath+key):
            os.mkdir(targetPath+key)
        try:
            urldownload(url,targetPath+key+'/XC'+nameTemp+'.ogg')
        except:
            error_list.append(url)
print(error_list)
