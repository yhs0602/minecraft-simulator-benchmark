import torch


def get_device(dev_num: int = 0) -> torch.device:
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{dev_num}")
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device
