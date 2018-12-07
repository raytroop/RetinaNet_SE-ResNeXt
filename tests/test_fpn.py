import math
import numpy as np
import torch
import torch.nn as nn
from models.fpn import top_down, classification_subnet, regression_subnet

def test_top_down():
    model = top_down()
    C3 = torch.randn(4, 512, 28, 28)
    C4 = torch.randn(4, 1024, 14, 14)
    C5 = torch.randn(4, 2048, 7, 7)

    P3, P4, P5, P6, P7 = model((C3, C4, C5))
    assert P3.shape == (4, 256, 28, 28)
    assert P4.shape == (4, 256, 14, 14)
    assert P5.shape == (4, 256, 7, 7)
    assert P6.shape == (4, 256, 4, 4)
    assert P7.shape == (4, 256, 2, 2)

    # ----------------------------------
    C3 = torch.randn(4, 512, 64, 64)
    C4 = torch.randn(4, 1024, 32, 32)
    C5 = torch.randn(4, 2048, 16, 16)

    P3, P4, P5, P6, P7 = model((C3, C4, C5))
    assert P3.shape == (4, 256, 64, 64)
    assert P4.shape == (4, 256, 32, 32)
    assert P5.shape == (4, 256, 16, 16)
    assert P6.shape == (4, 256, 8, 8)
    assert P7.shape == (4, 256, 4, 4)


def test_classification_subnet():
    model = classification_subnet()
    P3 = torch.randn(4, 256, 28, 28)
    feat3 = model(P3)
    P4 = torch.randn(4, 256, 14, 14)
    feat4 = model(P4)
    P5 = torch.randn(4, 256, 7, 7)
    feat5 = model(P5)
    P6 = torch.randn(4, 256, 4, 4)
    feat6 = model(P6)
    P7 = torch.randn(4, 256, 2, 2)
    feat7 = model(P7)
    assert feat3.shape == (4, 9*28*28, 1)
    assert feat4.shape == (4, 9*14*14, 1)
    assert feat5.shape == (4, 9*7*7, 1)
    assert feat6.shape == (4, 9*4*4, 1)
    assert feat7.shape == (4, 9*2*2, 1)

    assert len(list(model.children())) == 2

    assert isinstance(model.head, nn.Conv2d)
    assert model.head.weight.shape == (9, 256, 3, 3)
    np.testing.assert_almost_equal(model.head.weight.mean().item(), 0, decimal=2)
    np.testing.assert_almost_equal(model.head.weight.std().item(), 0.01, decimal=2)
    prior_probability=0.01
    np.testing.assert_almost_equal(model.head.bias.data.numpy(), -math.log((1 - prior_probability) / prior_probability))


def test_regression_subnet():
    model = regression_subnet()
    P3 = torch.randn(4, 256, 28, 28)
    feat3 = model(P3)
    P4 = torch.randn(4, 256, 14, 14)
    feat4 = model(P4)
    P5 = torch.randn(4, 256, 7, 7)
    feat5 = model(P5)
    P6 = torch.randn(4, 256, 4, 4)
    feat6 = model(P6)
    P7 = torch.randn(4, 256, 2, 2)
    feat7 = model(P7)
    assert feat3.shape == (4, 9*28*28, 4)
    assert feat4.shape == (4, 9*14*14, 4)
    assert feat5.shape == (4, 9*7*7, 4)
    assert feat6.shape == (4, 9*4*4, 4)
    assert feat7.shape == (4, 9*2*2, 4)
