import os
import json
import pickle
import pydicom
import torch
import numpy as np
from albumentations import Resize, Compose
from dataGen.data_loader import RsnaDataset, fetch_val_loader
from models.misc import build_anchors, bbox_transform_inv

config_path = os.path.join(os.path.dirname(__file__), '..', 'config.json')
with open(config_path, 'r') as f:
    config = json.load(f)

def test_boxtfsinv():
    def load_dicom(img_id):
        image_path = os.path.join(os.path.dirname(__file__), '..', 'dataset/stage_2_train_images', img_id+'.dcm')
        ds = pydicom.read_file(image_path)
        image = ds.pixel_array
        # If grayscale. Convert to RGB for consistency.
        if len(image.shape) != 3 or image.shape[2] != 3:
            image = np.stack((image,) * 3, -1)
        return image

    kfold = 1
    valfps_path = os.path.join(os.path.dirname(__file__), '..', 'fold_data', 'valfps_{}.pkl'.format(kfold))
    with open(valfps_path, 'rb') as f:
        valfps = pickle.load(f)

    bboxdict_path = os.path.join(os.path.dirname(__file__), '..', 'fold_data', 'valbox_{}.pkl'.format(kfold))
    with open(bboxdict_path, 'rb') as f:
        bboxdict = pickle.load(f)

    labeldict_path = os.path.join(os.path.dirname(__file__), '..', 'fold_data', 'vallabel_{}.pkl'.format(kfold))
    with open(labeldict_path, 'rb') as f:
        labeldict = pickle.load(f)
    sample = None
    for i, nm in enumerate(valfps):
        if len(labeldict[nm]) > 1:
            sample = nm
            idx = i
            break
    # 00436515-870c-4b36-a041-de91049b9ab4
    img = load_dicom(sample)
    # [[264, 152, 476, 530], [562, 152, 817, 604]]
    bboxes = bboxdict[sample]
    # [0, 0]
    labels = labeldict[sample]
    assert img.shape == (1024, 1024, 3)
    assert len(bboxes) > 0
    assert len(bboxes[0]) == 4
    assert len(labels) == len(bboxes)

    val_aug = [Resize(*config['image_shape'], p=1.0)]
    dt = RsnaDataset(valfps, bboxdict, labeldict, aug=val_aug)
    sample = dt[idx]
    assert len(sample) == 3
    assert sample[0].shape == (3, 224, 224)
    # when `config['image_shape']` == (224, 224)
    length = (28*28+14*14+7*7+4*4+2*2)*9
    assert sample[1].shape == (length, 2)
    # assert sample[1][:, 0].sum().item() == 2
    pos_label = (sample[1][:, 1] == 1).sum().item()
    ignore_label = (sample[1][:, 1] == -1).sum().item()
    neg_label = (sample[1][:, 1] == 0).sum().item()
    assert length == pos_label + ignore_label + neg_label

    pos_reg = (sample[2][..., -1] == 1).sum().item()
    ignore_reg = (sample[2][..., -1] == -1).sum().item()
    neg_reg = (sample[2][..., -1] == 0).sum().item()
    assert length == pos_reg + ignore_reg + neg_reg

    assert pos_label == pos_reg == 64
    assert ignore_label == ignore_reg == 109
    assert neg_label == neg_reg == 9268

    regression = sample[2]
    assert regression.shape == (length, 5)
    features = []
    for sz in [28, 14, 7, 4, 2]:
        features.append(np.empty(shape=(config['batch_size'], 256, sz, sz)))
    anchors = build_anchors(features)
    assert anchors.shape == (length, 4)
    regression = regression[None, :, :4]
    assert regression.shape == (1, length, 4)
    bboxes_pred = bbox_transform_inv(anchors, regression)[0]

    # resize bbox
    bboxes = np.array(bboxes, dtype=np.float32) * 224 / 1024
    bboxes_pred = bboxes_pred[sample[1][:, -1]==1].numpy()
    assert bboxes_pred.shape == (pos_label, 4)
    assert bboxes.shape == (2, 4)
    assert (bboxes_pred[:, 0] == bboxes[0, 0]).sum() > 0
    assert (bboxes_pred[:, 1] == bboxes[0, 1]).sum() > 0
    assert (bboxes_pred[:, 2] == bboxes[0, 2]).sum() > 0
    assert (bboxes_pred[:, 3] == bboxes[0, 3]).sum() > 0

    assert (bboxes_pred[:, 0] == bboxes[1, 0]).sum() > 0
    assert (bboxes_pred[:, 1] == bboxes[1, 1]).sum() > 0
    assert (bboxes_pred[:, 2] == bboxes[1, 2]).sum() > 0
    assert (bboxes_pred[:, 3] == bboxes[1, 3]).sum() > 0
