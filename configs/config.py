import torch


class Config:

    #####################################################
    # Dataset
    #####################################################

    DATASET = "BCICIV2A"

    DATA_PATH = "/kaggle/input/datasets/trishithapenmatsa/bcic2a-data"

    NUM_CLASSES = 4

    NUM_CHANNELS = 3

    SAMPLING_RATE = 250

    TIME_POINTS = 1000

    #####################################################
    # Training
    #####################################################

    EPOCHS = 500

    BATCH_SIZE = 32

    LR = 3e-4

    WEIGHT_DECAY = 5e-4

    LABEL_SMOOTHING = 0.05

    DROPOUT = 0.2

    ATTENTION_DROPOUT = 0.1

    CLIP_GRAD = 0.5

    WARMUP_EPOCHS = 20

    MIN_LR = 1e-6

    NUM_WORKERS = 4

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    SEED = 42

    #####################################################
    # Model
    #####################################################

    EMBED_DIM = 128

    NUM_LAYERS = 6

    NUM_QUERY_HEADS = 8

    NUM_KV_HEADS = 2

    HEAD_DIM = 16

    MLP_RATIO = 4


    ATTENTION_DROPOUT = 0.1

    STOCHASTIC_DEPTH = 0.1

    #####################################################
    # Tokenizer
    #####################################################

    TEMPORAL_KERNELS = [32, 64, 128]

    POOL_SIZE = 8

    TOKEN_DIM = EMBED_DIM

    #####################################################
    # RoPE
    #####################################################

    USE_ROPE = True

    ROPE_BASE = 10000

    #####################################################
    # Relative Position Bias
    #####################################################

    USE_RPB = True

    MAX_POSITION = 128

    #####################################################
    # Grouped Query Attention
    #####################################################

    USE_GQA = True

    #####################################################
    # Classification
    #####################################################


    #####################################################
    # Scheduler
    #####################################################



    MIN_LR = 1e-6

    #####################################################
    # Optimizer
    #####################################################

    BETAS = (0.9, 0.999)

    EPS = 1e-8

    #####################################################
    # Checkpoints
    #####################################################

    SAVE_DIR = "./checkpoints"

    LOG_DIR = "./outputs"

    #####################################################
    # Mixed Precision
    #####################################################

    USE_AMP = True

    #####################################################
    # Gradient Clipping
    #####################################################



cfg = Config()

import os

if os.path.exists("/kaggle/input/datasets/trishithapenmatsa/bcic2a-data"):
    cfg.DATA_PATH = "/kaggle/input/datasets/trishithapenmatsa/bcic2a-data"
