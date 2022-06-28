from typing import Tuple
import torch
import torch.nn.functional as F


def adv_hinge_gen_loss(fake_logits: torch.Tensor) -> torch.Tensor:
    return -fake_logits.mean()


def adv_hinge_disc_loss(fake_logits: torch.Tensor, real_logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    fake_loss = (F.relu(torch.ones_like(fake_logits) + fake_logits)).mean()
    real_loss = (F.relu(torch.ones_like(real_logits) - real_logits)).mean()
    return fake_loss, real_loss
