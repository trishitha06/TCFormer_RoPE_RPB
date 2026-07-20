import torch


def get_scheduler(
    optimizer,
    epochs,
    eta_min=1e-6
):
    """
    Cosine Annealing Learning Rate Scheduler.
    """

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=eta_min
    )

    return scheduler