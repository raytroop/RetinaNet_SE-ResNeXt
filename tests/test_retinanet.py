import os
import json
import numpy as np
import torch
import torch.nn as nn
from models.fpn import retinanet
from dataGen.data_loader import load_dicom, fetch_trn_loader, fetch_val_loader
from models.losses import focal_loss, smooth_l1_loss


config_path = os.path.join(os.path.dirname(__file__), '..', 'config.json')
with open(config_path, 'r') as f:
    config = json.load(f)
num_anchors = len(config['anchor_ratios_default']) * len(config['anchor_scales_default'])
length = (28*28+14*14+7*7+4*4+2*2)*num_anchors

def test_retinanet():
    model = retinanet()
    images = torch.randn(8, 3, 224, 224)
    classification, regression = model(images)
    assert classification.shape == (8, length, 1)
    assert regression.shape == (8, length, 4)

    results = model.predict(images)
    assert len(results) == 8
    assert len(results[0]) == 3
    for res in results:
        assert res[0].shape[0] == res[1].shape[0] == res[2].shape[0]
        assert res[0].shape[1] == 4
        assert len(res[1].shape) == 1
        assert len(res[2].shape) == 1
        assert res[0].dtype == np.float32
        assert res[1].dtype == np.float32
        assert res[2].dtype == np.int64


def test_merged():
    model = retinanet()
    model = model.cuda()
    for k in range(1, 6):
        dl = iter(fetch_trn_loader(k))
        for i in range(100):
            img_batch, labels_batch, regression_batch = next(dl)
            img_batch = img_batch.cuda()
            labels_batch = labels_batch.cuda()
            regression_batch = regression_batch.cuda()

            classification, regression = model(img_batch)
            assert classification.shape == (config['batch_size'], length, config['num_classes'])
            assert labels_batch.shape == (config['batch_size'], length, config['num_classes']+1)
            assert regression.shape == (config['batch_size'], length, 4)
            assert regression_batch.shape == (config['batch_size'], length, 4+1)

            focal = focal_loss()
            smooth_l1 = smooth_l1_loss()
            assert focal(labels_batch, classification).shape == torch.Size([])
            assert smooth_l1(regression_batch, regression).shape == torch.Size([])

            results = model.predict(img_batch)

    for k in range(1, 6):
        dl = iter(fetch_val_loader(k))
        for i in range(100):
            img_batch, labels_batch, regression_batch = next(dl)
            img_batch = img_batch.cuda()
            labels_batch = labels_batch.cuda()
            regression_batch = regression_batch.cuda()

            classification, regression = model(img_batch)
            assert classification.shape == (config['batch_size'], length, config['num_classes'])
            assert labels_batch.shape == (config['batch_size'], length, config['num_classes']+1)
            assert regression.shape == (config['batch_size'], length, 4)
            assert regression_batch.shape == (config['batch_size'], length, 4+1)

            focal = focal_loss()
            smooth_l1 = smooth_l1_loss()
            assert focal(labels_batch, classification).shape == torch.Size([])
            assert smooth_l1(regression_batch, regression).shape == torch.Size([])

            results = model.predict(img_batch)
