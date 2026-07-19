import wandb

def init_run(cfg):
    run = wandb.init(
        project=cfg["wandb"]["project"],
        name=cfg["run_name"],
        tags=cfg["wandb"].get("tags", []),
        config=cfg,
    )
    return run


def log(metrics, step=None):
    if step is not None:
        wandb.log(metrics, step=step)
    else:
        wandb.log(metrics)


def finish():
    wandb.finish()