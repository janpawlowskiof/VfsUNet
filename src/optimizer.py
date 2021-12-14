import torch


def get_optimizer(parameters, optimizer_config) -> torch.optim.Optimizer:
    if optimizer_config["name"] == "Adam":
        return torch.optim.Adam(parameters, lr=optimizer_config["lr"])
    if optimizer_config["name"] == "AdamW":
        return torch.optim.AdamW(parameters, lr=optimizer_config["lr"])
    raise RuntimeError(f"Unknown optimizer config {optimizer_config}")
