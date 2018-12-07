import os
import random
import json
import torch
import numpy as np
from dataGen.data_loader import load_dicom, fetch_trn_loader, fetch_val_loader

config_path = os.path.join(os.path.dirname(__file__), '..', 'config.json')
with open(config_path, 'r') as f:
    config = json.load(f)
num_anchors = len(config['anchor_ratios_default']) * len(config['anchor_scales_default'])
length = (28*28+14*14+7*7+4*4+2*2)*num_anchors

def test_loaddicm():
    ids = os.listdir(os.path.join(os.path.dirname(__file__), '..', config['dicom_train']))
    assert len(ids) > 0
    imgid = random.sample(ids, 1)[0]
    img = load_dicom(os.path.splitext(imgid)[0])
    assert img.shape == (1024, 1024, 3)


def test_dataloader():
    for i in range(1, 6):
        dl = iter(fetch_trn_loader(i))
        img_batch, labels_batch, regression_batch = next(dl)
        assert img_batch.shape == (config['batch_size'], 3, *config['image_shape'])
        assert labels_batch.shape == (config['batch_size'], length, config['num_classes']+1)
        assert regression_batch.shape == (config['batch_size'], length, 4+1)

    for i in range(1, 6):
        dl = iter(fetch_val_loader(i))
        img_batch, labels_batch, regression_batch = next(dl)
        assert img_batch.shape == (config['batch_size'], 3, *config['image_shape'])
        assert labels_batch.shape == (config['batch_size'], length, config['num_classes']+1)
        assert regression_batch.shape == (config['batch_size'], length, 4+1)
