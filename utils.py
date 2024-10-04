import wandb


def safe_log_to_wandb(data: dict[str, any]):
    """
    Logs data to wandb.
    If wandb is not initialized, it will not log anything.
    """
    if wandb.run is not None:
        wandb.log(data)
