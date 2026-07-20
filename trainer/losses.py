import torch.nn as nn


def build_loss(label_smoothing=0.1):

    criterion = nn.CrossEntropyLoss(

        label_smoothing=label_smoothing

    )

    return criterion