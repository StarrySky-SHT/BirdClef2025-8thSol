import torch

class CFG:
    seed = 42
    num_classes = 206
    batch_size = 64
    epochs = 30
    PRECISION = 16    
    PATIENCE = 8    
    img_size = [224,224]
    model = "tf_efficientnetv2_b0"
    pretrained = True            
    weight_decay = 1e-4
    use_mixup = True
    mixup_pro = 0.5
    mixup_alpha = 0.3  

    use_cutmix = True
    cutmix_pro = 0.5
    cutmix_alpha = 0.3 
    use_spec_aug = True
    p_spec_aug = 0.5
    time_shift_prob = 0.5
    gn_prob = 0.5
    secondary_labels_weight = 0.5
    smoothing_factor = 0.005
    use_fsr = False

    use_instance_mixup = True
    instance_mixup_pro = 0.5
    instance_mixup_alpha = 0.3 

    finetune_weight = False

    fmin = 40
    fmax = 14000
    mel_bins = 192
    n_fft = 2048
    window_size = 1024
    hop_size = 512

    device = torch.device('cuda')  

    data_root = "/root/projects/BirdClef2025/data/train_audio/"
    train_path = "/root/projects/BirdClef2025/data/train.csv"
    valid_path = "/root/projects/BirdClef2025/data/external_valid.csv"
    log_dir = "/root/projects/BirdClef2025/BirdCLEF2023-30th-place-solution-master/logs/"
    sample_rate = 32000
    duration = 10
    infer_duration = 5
    max_read_samples = 10
    lr = 5e-5

    scheduler     = 'CosineAnnealingLR'
    min_lr        = 1e-6
    T_max         = int(30000/batch_size*epochs)+50
    T_0           = 25

    n_accumulate = 4
