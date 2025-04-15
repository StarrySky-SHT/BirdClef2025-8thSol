from pyinaturalist import get_observations
import requests
import os
from tqdm import tqdm
from urllib.parse import urlparse
import time
from pyinaturalist import get_taxa, get_observations
import pandas as pd

def download_species_audio(
    taxon_id: int,               # 物种Taxon ID
    output_dir: str = "audio",   # 输出目录
    quality_grade: str = None,   # 质量等级 (research/casual)
    max_files: int = 100,        # 最大下载文件数
    request_delay: float = 1.0,  # API请求间隔（防封禁）
):
    """下载指定物种的iNaturalist音频观测记录
    
    Args:
        taxon_id: 通过get_taxa获取的物种ID
        output_dir: 音频文件存储目录
        quality_grade: 数据质量等级过滤
        max_files: 最大下载数量
        request_delay: 请求间隔秒数（遵守API速率限制）
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取包含音频的观测记录
    observations = []
    page = 1
    with tqdm(desc="扫描观测记录", unit="page") as pbar:
        while len(observations) < max_files:
            response = get_observations(
                taxon_id=taxon_id,
                page=page,
                per_page=30,  # 每页数量
                quality_grade=quality_grade,
                sounds=True,
                # media_type="sound",  # 关键参数：只获取音频记录
            )
            
            if not response.get("results"):
                break
                
            observations.extend(response["results"])
            pbar.update(1)
            page += 1
            time.sleep(request_delay)  # 遵守API速率限制
            
            if len(observations) >= max_files:
                observations = observations[:max_files]
                break

    # 下载音频文件
    downloaded_files = []
    for obs in tqdm(observations, desc="下载音频", unit="file"):
        try:
            # 提取音频URL（优先原始质量）
            sounds = obs.get("sounds", [])
            if not sounds:
                continue
                
            # 选择最佳音质（存在多个录音时）
            best_audio = max(sounds, key=lambda x: x.get("file_size", 0))
            audio_url = best_audio["file_url"]
            
            # 生成文件名
            parsed_url = urlparse(audio_url)
            filename = f"{obs['id']}_{parsed_url.path.split('/')[-1]}"
            filepath = os.path.join(output_dir, filename)
            
            # 下载文件
            response = requests.get(audio_url, stream=True, timeout=10)
            response.raise_for_status()
            
            with open(filepath, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    
            downloaded_files.append(filepath)
            time.sleep(request_delay)  # 控制下载频率
            
        except Exception as e:
            print(f"下载失败 (观测ID {obs['id']}): {str(e)}")
    
    print(f"\n成功下载 {len(downloaded_files)} 个音频文件到 {output_dir}/")
    return downloaded_files

# 使用示例：
if __name__ == "__main__":
    # 先通过物种名称获取taxon_id（参考之前的物种搜索代码）

    df = pd.read_csv('/root/projects/BirdClef2025/data/train.csv')
    value_counts = df['primary_label'].value_counts()
    small_size_species = list(value_counts[value_counts<50].index)

    for ebird in small_size_species:
        species_name = df[df["primary_label"]==ebird].iloc[0]['scientific_name']
        taxon_response = get_taxa(q=species_name, rank='species')
        if not taxon_response['results']:
            print(f"未找到物种: {species_name}")
            continue
        
        taxon_id = taxon_response['results'][0]['id']
        print(f"找到物种ID: {taxon_id}")
        
        # 执行下载
        audio_files = download_species_audio(
            taxon_id=taxon_id,
            output_dir=f"/root/projects/BirdClef2025/externaldata/download_data/{ebird}",
            quality_grade=None,  # 仅下载研究级数据
            max_files=100,
            request_delay=0.5,         # 保守的请求间隔
        )