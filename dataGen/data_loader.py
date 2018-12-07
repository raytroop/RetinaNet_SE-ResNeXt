import os
import json
import pickle
import pydicom
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from albumentations import HorizontalFlip, Rotate, Resize, Compose
from models.backbone import pretrained_settings
from .targetBuild import anchor_targets_bbox, anchors_for_shape

config_path = os.path.join(os.path.dirname(__file__), '..', 'config.json')
with open(config_path, 'r') as f:
    config = json.load(f)


def load_dicom(img_id):
    image_path = os.path.join(os.path.dirname(__file__), '..', config['dicom_train'], img_id+'.dcm')
    ds = pydicom.read_file(image_path)
    image = ds.pixel_array
    # If grayscale. Convert to RGB for consistency.
    if len(image.shape) != 3 or image.shape[2] != 3:
        image = np.stack((image,) * 3, -1)
    return image


def get_aug(aug, min_area=0., min_visibility=0.):
    return Compose(aug, bbox_params={'format': 'pascal_voc', 'min_area': min_area, 'min_visibility': min_visibility, 'label_fields': ['category_id']})


class RsnaDataset(Dataset):
    """
    A standard PyTorch definition of Dataset which defines the functions __len__ and __getitem__.
    """

    def __init__(self, filenames, gt_bboxes, gt_catids, aug=None):
        """
        Store the filenames of the jpgs to use. Specifies transforms to apply on images.

        Args:
            data_dir: (string) directory containing the dataset
            transform: (torchvision.transforms) transformation to apply on image
        """
        self.filenames = filenames
        self.gt_bboxes = gt_bboxes
        self.gt_catids = gt_catids
        self.aug = get_aug(aug) if aug is not None else get_aug([])
        model_name = config['backbone']
        # img_mean = pretrained_settings[model_name]['imagenet']['mean']
        # img_std = pretrained_settings[model_name]['imagenet']['std']
        self.img_tfs = transforms.Compose([transforms.ToTensor(), transforms.Normalize(
            mean=[config['rsna_mean']]*3, std=[config['rsna_std']]*3)])
        self.anchors = anchors_for_shape(config['image_shape'])

    def __len__(self):
        # return size of dataset
        return len(self.filenames)

    def __getitem__(self, idx):
        """
        Fetch index idx image and labels from dataset. Perform transforms on image.

        Args:
            idx: (int) index in [0, 1, ..., size_of_dataset-1]

        Returns:
            image: (Tensor) transformed image
            label: (int) corresponding label of image
        """
        fps = self.filenames[idx]
        image = load_dicom(fps)
        bboxes = self.gt_bboxes[fps]
        category_id = self.gt_catids[fps]
        augmented = self.aug(image=image, bboxes=bboxes, category_id=category_id)
        if not augmented['bboxes']:
            gt_annos = np.zeros((0, 5), dtype=np.float32)
        else:
            gt_annos = np.empty((len(augmented['bboxes']), 5), dtype=np.float32)
            gt_annos[:, :4] = augmented['bboxes']
            gt_annos[:, 4] = augmented['category_id']
        img = self.img_tfs(augmented['image'])
        labels, regression = anchor_targets_bbox(self.anchors, gt_annos)
        labels, regression = torch.tensor(labels), torch.tensor(regression)
        return img, labels, regression

trn_aug = []
if config['image_shape']:
    trn_aug.append(Resize(*config['image_shape'], p=1.0))
    val_aug = [Resize(*config['image_shape'], p=1.0)]
else:
    val_aug = []

if config['RandomRotate']:
    trn_aug.append(Rotate(limit=15, p=0.5))

if config['RandomHorizontalFlip']:
    trn_aug.append(HorizontalFlip(p=0.5))


def fetch_trn_loader(kfold, trnfps=None, bboxdict=None, labeldict=None, aug=None):
    if trnfps is None:
        trnfps_path = os.path.join(os.path.dirname(__file__), '..', 'fold_data', 'trnfps_{}.pkl'.format(kfold))
        with open(trnfps_path, 'rb') as f:
            trnfps = pickle.load(f)

    if bboxdict is None:
        bboxdict_path = os.path.join(os.path.dirname(__file__), '..', 'fold_data', 'trnbox_{}.pkl'.format(kfold))
        with open(bboxdict_path, 'rb') as f:
            bboxdict = pickle.load(f)

    if labeldict is None:
        labeldict_path = os.path.join(os.path.dirname(__file__), '..', 'fold_data', 'trnlabel_{}.pkl'.format(kfold))
        with open(labeldict_path, 'rb') as f:
            labeldict = pickle.load(f)

    if aug is None:
        aug = trn_aug

    dataset = RsnaDataset(trnfps, bboxdict, labeldict, aug)
    return DataLoader(dataset,
                    batch_size=config['batch_size'],
                    shuffle=True,
                    num_workers=config['num_workers'],
                    pin_memory=config['use_cuda'])


def fetch_val_loader(kfold, valfps=None, bboxdict=None, labeldict=None, aug=None):
    if valfps is None:
        valfps_path = os.path.join(os.path.dirname(__file__), '..', 'fold_data', 'valfps_{}.pkl'.format(kfold))
        with open(valfps_path, 'rb') as f:
            valfps = pickle.load(f)

    if bboxdict is None:
        bboxdict_path = os.path.join(os.path.dirname(__file__), '..', 'fold_data', 'valbox_{}.pkl'.format(kfold))
        with open(bboxdict_path, 'rb') as f:
            bboxdict = pickle.load(f)

    if labeldict is None:
        labeldict_path = os.path.join(os.path.dirname(__file__), '..', 'fold_data', 'vallabel_{}.pkl'.format(kfold))
        with open(labeldict_path, 'rb') as f:
            labeldict = pickle.load(f)

    if aug is None:
        aug = val_aug

    dataset = RsnaDataset(valfps, bboxdict, labeldict, aug)
    return DataLoader(dataset,
                    batch_size=config['batch_size'],
                    shuffle=False,
                    num_workers=config['num_workers'],
                    pin_memory=config['use_cuda'])

