import numpy as np
import torch
from models.losses import focal_loss, smooth_l1_loss

def test_focal():
    focal = focal_loss()
    y_pred = torch.rand(8, 3, 1)
    y_true = torch.rand(8, 3, 2)
    y_true[..., -1] = torch.tensor([1, -1, 0])
    assert (y_true[..., -1] == torch.tensor([1, -1, 0], dtype=torch.float32)).all()
    assert focal(y_true, y_pred)


def test_smooth_l1():
    smooth_l1 = smooth_l1_loss()
    y_pred = torch.rand(8, 3, 4)
    y_true = torch.rand(8, 3, 5)
    y_true[..., -1] = torch.tensor([1, -1, 0])
    assert (y_true[..., -1] == torch.tensor([1, -1, 0], dtype=torch.float32)).all()
    assert smooth_l1(y_true, y_pred)
