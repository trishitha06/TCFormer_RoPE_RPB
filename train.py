import os

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW

from configs.config import Config

from datasets.bcic2a import BCIC2aDataset

from models.tcformer import TCFormer

from trainer.engine import Trainer
from trainer.losses import build_loss

from utils.scheduler import get_scheduler
from utils.seed import set_seed

def main():

    cfg = Config()

    set_seed(cfg.SEED)

    device = torch.device(cfg.DEVICE)

    print(f"Using device: {device}")

    train_subjects = [1, 2, 3, 4, 5, 6, 7]
    val_subjects = [8, 9]

    train_dataset = BCIC2aDataset(
        root=cfg.DATA_PATH,
        subjects=train_subjects,
        train=True
    )

    val_dataset = BCIC2aDataset(
        root=cfg.DATA_PATH,
        subjects=val_subjects,
        train=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True
    )

    model = TCFormer(

        num_channels=cfg.NUM_CHANNELS,

        embed_dim=cfg.EMBED_DIM,

        depth=cfg.NUM_LAYERS,

        q_heads=cfg.NUM_QUERY_HEADS,

        kv_heads=cfg.NUM_KV_HEADS,

        head_dim=cfg.HEAD_DIM,

        num_classes=cfg.NUM_CLASSES,

        dropout=cfg.DROPOUT

    ).to(device)

    optimizer = AdamW(

        model.parameters(),

        lr=cfg.LR,

        weight_decay=cfg.WEIGHT_DECAY,

        betas=cfg.BETAS,

        eps=cfg.EPS

    )

    scheduler = get_scheduler(

        optimizer,

        epochs=cfg.EPOCHS,

        eta_min=cfg.MIN_LR

    )

    criterion = build_loss(

        label_smoothing=cfg.LABEL_SMOOTHING

    )

    trainer = Trainer(

        model=model,

        optimizer=optimizer,

        scheduler=scheduler,

        criterion=criterion,

        device=device

    )

    save_path = os.path.join(

        cfg.SAVE_DIR,

        "tcformer_best.pth"

    )

    trainer.fit(

        train_loader=train_loader,

        val_loader=val_loader,

        epochs=cfg.EPOCHS,

        save_path=save_path

    )

if __name__ == "__main__":

    main()
    