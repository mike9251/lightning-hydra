import torch


def cosine_dist_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    dist = torch.sum(x * y, dim=1) / (torch.linalg.norm(x, dim=1) * torch.linalg.norm(y, dim=1))
    return (1.0 - dist).mean()
