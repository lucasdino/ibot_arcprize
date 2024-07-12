class Args:
    arch = 'vit_small'
    patch_size = 2
    window_size = 7
    out_dim = 128
    patch_out_dim = 128
    norm_last_layer = True
    momentum_teacher = 0.996
    use_masked_im_modeling = True
    pred_ratio_mean = 0.25
    pred_ratio_var = 0.05
    pred_ratio_pad_divisor = 10
    pred_start_epoch = 10
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


# # Model parameters
# parser.add_argument('--arch', default='vit_small', type=str,
#     choices=['vit_tiny', 'vit_small', 'vit_base', 'vit_large', 'deit_tiny', 'deit_small',
#                 'swin_tiny','swin_small', 'swin_base', 'swin_large'],
#     help="""Name of architecture to train. For quick experiments with ViTs,
#     we recommend using vit_tiny or vit_small.""")
# parser.add_argument('--patch_size', default=4, type=int, help="""Size in pixels
#     of input square patches - default 4 (for 4x4 patches). Using smaller
#     values leads to better performance but requires more memory. Applies only
#     for ViTs (vit_tiny, vit_small and vit_base). If <16, we recommend disabling
#     mixed precision training (--use_fp16 false) to avoid unstabilities.""")
# parser.add_argument('--window_size', default=7, type=int, help="""Size of window - default 7.
#     This config is only valid for Swin Transofmer and is ignoired for vanilla ViT architectures.""")
# parser.add_argument('--out_dim', default=128, type=int, help="""Dimensionality of
#     output for [CLS] token.""")
# parser.add_argument('--patch_out_dim', default=128, type=int, help="""Dimensionality of
#     output for patch tokens.""")
# parser.add_argument('--norm_last_layer', default=True, type=utils.bool_flag,
#     help="""Whether or not to weight normalize the last layer of the head.
#     Not normalizing leads to better performance but can make the training unstable.
#     In our experiments, we typically set this paramater to False with vit_small and True with vit_base.""")
# parser.add_argument('--momentum_teacher', default=0.996, type=float, help="""Base EMA
#     parameter for teacher update. The value is increased to 1 during training with cosine schedule.
#     We recommend setting a higher value with small batches: for example use 0.9995 with batch size of 256.""")
# parser.add_argument('--use_masked_im_modeling', default=True, type=utils.bool_flag,
#     help="Whether to use masked image modeling (mim) in backbone (Default: True)")
# parser.add_argument('--pred_ratio_mean', default=0.3, type=float, nargs='+', help="""Ratio of partial prediction.
#     If a list of ratio is specified, one of them will be randomly choosed for each patch.""")
# parser.add_argument('--pred_ratio_var', default=0, type=float, nargs='+', help="""Variance of partial prediction
#     ratio. Length should be indentical to the length of pred_ratio. 0 for disabling. """)
# parser.add_argument('--pred_ratio_pad_divisor', default=5, type=float, nargs='+', help="""Amount to divide our pred
#     ratio by to determine the pred ratio for our padding. """)
# parser.add_argument('--pred_start_epoch', default=0, type=int, help="""Start epoch to perform masked
#     image prediction. We typically set this to 50 for swin transformer. (Default: 0)""")
# parser.add_argument('--lambda1', default=1.0, type=float, help="""loss weight for dino
#     loss over [CLS] tokens (Default: 1.0)""")
# parser.add_argument('--lambda2', default=1.0, type=float, help="""loss weight for beit 
#     loss over masked patch tokens (Default: 1.0)""")
    
# # Temperature teacher parameters
# parser.add_argument('--warmup_teacher_temp', default=0.04, type=float,
#     help="""Initial value for the teacher temperature: 0.04 works well in most cases.
#     Try decreasing it if the training loss does not decrease.""")
# parser.add_argument('--teacher_temp', default=0.04, type=float, help="""Final value (after linear warmup)
#     of the teacher temperature. For most experiments, anything above 0.07 is unstable. We recommend
#     starting with the default value of 0.04 and increase this slightly if needed.""")
# parser.add_argument('--warmup_teacher_patch_temp', default=0.04, type=float, help="""See 
#     `--warmup_teacher_temp`""")
# parser.add_argument('--teacher_patch_temp', default=0.07, type=float, help=""""See 
#     `--teacher_temp`""")
# parser.add_argument('--warmup_teacher_temp_epochs', default=30, type=int,
#     help='Number of warmup epochs for the teacher temperature (Default: 30).')

# # Training/Optimization parameters
# parser.add_argument('--use_fp16', type=utils.bool_flag, default=True, help="""Whether or not
#     to use half precision for training. Improves training time and memory requirements,
#     but can provoke instability and slight decay of performance. We recommend disabling
#     mixed precision if the loss is unstable, if reducing the patch size or if training with bigger ViTs.""")
# parser.add_argument('--weight_decay', type=float, default=0.04, help="""Initial value of the
#     weight decay. With ViT, a smaller value at the beginning of training works well.""")
# parser.add_argument('--weight_decay_end', type=float, default=0.4, help="""Final value of the
#     weight decay. We use a cosine schedule for WD and using a larger decay by
#     the end of training improves performance for ViTs.""")
# parser.add_argument('--clip_grad', type=float, default=3.0, help="""Maximal parameter
#     gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can
#     help optimization for larger ViT architectures. 0 for disabling.""")
# parser.add_argument('--batch_size_per_gpu', default=128, type=int,
#     help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
# parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of training.')
# parser.add_argument('--freeze_last_layer', default=1, type=int, help="""Number of epochs
#     during which we keep the output layer fixed. Typically doing so during
#     the first epoch helps training. Try increasing this value if the loss does not decrease.""")
# parser.add_argument("--lr", default=0.0005, type=float, help="""Learning rate at the end of
#     linear warmup (highest LR used during training). The learning rate is linearly scaled
#     with the batch size, and specified here for a reference batch size of 256.""")
# parser.add_argument("--warmup_epochs", default=10, type=int,
#     help="Number of epochs for the linear learning-rate warm up.")
# parser.add_argument('--min_lr', type=float, default=1e-6, help="""Target LR at the
#     end of optimization. We use a cosine LR schedule with linear warmup.""")
# parser.add_argument('--optimizer', default='adamw', type=str,
#     choices=['adamw', 'sgd', 'lars'], help="""Type of optimizer. We recommend using adamw with ViTs.""")
# parser.add_argument('--load_from', default=None, help="""Path to load checkpoints to resume training.""")
# parser.add_argument('--drop_path', type=float, default=0.1, help="""Drop path rate for student network.""")

# # Multi-crop parameters
# parser.add_argument('--global_crops_number', type=int, default=2, help="""Number of global
#     views to generate. Default is to use two global crops. """)
# parser.add_argument('--global_crops_scale', type=float, nargs='+', default=(0.14, 1.),
#     help="""Scale range of the cropped image before resizing, relatively to the origin image.
#     Used for large global view cropping. When disabling multi-crop (--local_crops_number 0), we
#     recommand using a wider range of scale ("--global_crops_scale 0.14 1." for example)""")
# parser.add_argument('--pad_to_32', type=bool, default=True, help="""Tell model if you want to pad your image
#     to 32x32 regardless of image size (if True) -- if False, will pad to nearest mult of 4. """)
# parser.add_argument('--local_crops_number', type=int, default=0, help="""Number of small
#     local views to generate. Set this parameter to 0 to disable multi-crop training.
#     When disabling multi-crop we recommend to use "--global_crops_scale 0.14 1." """)
# parser.add_argument('--local_crops_scale', type=float, nargs='+', default=(0.05, 0.4),
#     help="""Scale range of the cropped image before resizing, relatively to the origin image.
#     Used for small local view cropping of multi-crop.""")

# # Misc
# parser.add_argument('--output_dir', default="trained_models/", type=str, help='Path to save logs and checkpoints.')
# parser.add_argument('--saveckp_freq', default=40, type=int, help='Save checkpoint every x epochs.')
# parser.add_argument('--seed', default=0, type=int, help='Random seed.')
# parser.add_argument('--num_workers', default=1, type=int, help='Number of data loading workers per GPU.')
# parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")