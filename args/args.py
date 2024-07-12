class Args:
    arch = 'vit_small'
    patch_size = 4
    window_size = 7
    out_dim = 128
    patch_out_dim = 128
    norm_last_layer = True
    momentum_teacher = 0.996
    use_masked_im_modeling = True
    pred_ratio = [0.3]
    pred_ratio_var = [0]
    pred_shape = 'block'
    pred_start_epoch = 0
    lambda1 = 1.0
    lambda2 = 1.0
    warmup_teacher_temp = 0.04
    teacher_temp = 0.04
    warmup_teacher_patch_temp = 0.04
    teacher_patch_temp = 0.07
    warmup_teacher_temp_epochs = 30
    use_fp16 = True
    weight_decay = 0.04
    weight_decay_end = 0.4
    clip_grad = 3.0
    batch_size_per_gpu = 128
    epochs = 100
    freeze_last_layer = 1
    lr = 0.0005
    warmup_epochs = 10
    min_lr = 1e-6
    optimizer = 'adamw'
    load_from = None
    drop_path = 0.1
    global_crops_number = 2
    global_crops_scale = (0.14, 1.0)
    pad_to_32 = True
    local_crops_number = 0
    local_crops_scale = (0.05, 0.4)
    output_dir = "trained_models/"
    saveckp_freq = 40
    seed = 0
    num_workers = 1
    dist_url = "env://"
    local_rank = 0

def get_args():
    args = Args()
    return args