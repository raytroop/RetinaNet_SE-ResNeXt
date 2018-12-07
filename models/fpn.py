import os
import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import backbone
from .misc import filter_detections, bbox_transform_inv, clip_boxes, build_anchors
from dataGen.targetBuild import anchors_for_shape

config_path = os.path.join(os.path.dirname(__file__), '..', 'config.json')
with open(config_path, 'r') as f:
    config = json.load(f)


class top_down(nn.Module):
    """ Creates the FPN layers on top of the backbone features.

        Args
            C3           : Feature stage C3 from the backbone.
            C4           : Feature stage C4 from the backbone.
            C5           : Feature stage C5 from the backbone.
            feature_size : The feature size to use for the resulting feature levels.

        Returns
            A list of feature levels [P3, P4, P5, P6, P7].
    """

    def __init__(self, feature_size=256):
        super(top_down, self).__init__()
        self.C5_reduced = nn.Conv2d(2048, feature_size, kernel_size=1, stride=1)
        self.P5_conv = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)
        self.C4_reduced = nn.Conv2d(1024, feature_size, kernel_size=1, stride=1)
        self.P4_conv = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)
        self.C3_reduced = nn.Conv2d(512, feature_size, kernel_size=1, stride=1)
        self.P3_conv = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)
        self.P6_conv = nn.Conv2d(2048, feature_size, kernel_size=3, stride=2, padding=1)
        self.P7_conv = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        C3, C4, C5 = x

        # upsample C5 to get P5 from the FPN paper
        P5 = self.C5_reduced(C5)
        P5_upsampled = F.interpolate(P5, scale_factor=2, mode='nearest')
        P5 = self.P5_conv(P5)

        # add P5 elementwise to C4
        P4 = self.C4_reduced(C4)
        P4 = P5_upsampled + P4
        P4_upsampled = F.interpolate(P4, scale_factor=2, mode='nearest')
        P4 = self.P4_conv(P4)

        # add P4 elementwise to C3
        P3 = self.C3_reduced(C3)
        P3 = P4_upsampled + P3
        P3 = self.P3_conv(P3)

        # "P6 is obtained via a 3x3 stride-2 conv on C5"
        P6 = self.P6_conv(C5)

        # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
        P7 = F.relu(P6)
        P7 = self.P7_conv(P7)

        return [P3, P4, P5, P6, P7]


class classification_subnet(nn.Module):
    """ the default classification submodel."""
    options = {
        'kernel_size': 3,
        'stride': 1,
        'padding': 1,
    }

    def __init__(self, num_classes=1, num_anchors=9, pyramid_feature_size=256, prior_probability=0.01):
        """
        Args
            num_classes                 : Number of classes to predict a score for at each feature level.
            num_anchors                 : Number of anchors to predict classification scores for at each feature level.
            pyramid_feature_size        : The number of filters to expect from the feature pyramid levels.
            prior_probability           : Prior probability for training stability in early training
        """
        super().__init__()
        convs = []
        for i in range(4):
            conv = nn.Conv2d(pyramid_feature_size, pyramid_feature_size, **classification_subnet.options)
            nn.init.normal_(conv.weight, mean=0.0, std=0.01)
            nn.init.zeros_(conv.bias)
            convs.append(conv)
            convs.append(nn.ReLU())
        self.feats = nn.Sequential(*convs)
        self.num_classes = num_classes
        head = nn.Conv2d(pyramid_feature_size, out_channels=num_classes * num_anchors, **classification_subnet.options)
        nn.init.normal_(head.weight, mean=0.0, std=0.01)
        nn.init.constant_(head.bias, val=-math.log((1 - prior_probability) / prior_probability))
        self.head = head

    def forward(self, x):
        outputs = self.feats(x)
        outputs = self.head(outputs)

        # reshape output and apply sigmoid
        outputs = outputs.permute(0, 2, 3, 1).contiguous()
        outputs = outputs.view(outputs.shape[0], -1, self.num_classes)
        outputs = torch.sigmoid(outputs)
        return outputs


class regression_subnet(nn.Module):
    """ Creates the default regression submodel."""
    options = {
        'kernel_size': 3,
        'stride': 1,
        'padding': 1,
    }

    def __init__(self, num_values=4, num_anchors=9, pyramid_feature_size=256):
        """
        Args
            num_values              : Number of values to regress.
            num_anchors             : Number of anchors to regress for each feature level.
            pyramid_feature_size    : The number of filters to expect from the feature pyramid levels.
        """
        super().__init__()
        self.num_values = num_values
        convs = []
        for i in range(4):
            conv = nn.Conv2d(pyramid_feature_size, pyramid_feature_size, **regression_subnet.options)
            nn.init.normal_(conv.weight, mean=0.0, std=0.01)
            nn.init.zeros_(conv.bias)
            convs.append(conv)
            convs.append(nn.ReLU())
        self.feats = nn.Sequential(*convs)

        head = nn.Conv2d(pyramid_feature_size, out_channels=num_anchors * num_values, **regression_subnet.options)
        nn.init.normal_(head.weight, mean=0.0, std=0.01)
        nn.init.zeros_(head.bias)
        self.head = head

    def forward(self, x):
        outputs = self.feats(x)
        outputs = self.head(outputs)

        # reshape
        outputs = outputs.permute(0, 2, 3, 1).contiguous()
        outputs = outputs.view(outputs.shape[0], -1, self.num_values)
        return outputs


class retinanet(nn.Module):
    """ Construct a RetinaNet model on top of a backbone, without bbox prediction transform"""

    def __init__(self, backbone_name=None, num_classes=None, num_anchors=None, pretrained_imagenet=None):
        """
         Args
            backbone_name           : backbone name,  `se_resnext50_32x4d` or `se_resnext101_32x4d`
            num_classes             : Number of classes to classify.
            num_anchors             : Number of base anchors.
        """
        super().__init__()

        if backbone_name is None:
            backbone_name = config['backbone']
        assert backbone_name in ['se_resnext50_32x4d', 'se_resnext101_32x4d'], \
            "`se_resnext50_32x4d` or `se_resnext101_32x4d`"
        bottom_up = getattr(backbone, backbone_name)

        if pretrained_imagenet is None:
            pretrained_imagenet=config['pretrained_imagenet']
        self.bottom_up = bottom_up(pretrained_imagenet)

        self.top_down = top_down(feature_size=256)

        if num_classes is None:
            num_classes = config['num_classes']
        if num_anchors is None:
            num_anchors = len(config['anchor_ratios_default']) * len(config['anchor_scales_default'])
        self.classification_subnet = classification_subnet(num_classes, num_anchors, 256, 0.01)
        self.regression_subnet = regression_subnet(4, num_anchors, 256)

    def forward(self, images):
        """
        Args:
            images: Tensor of (B, 3, H, W), where B is the batch size; H, w is image height, width
        """
        C3, C4, C5 = self.bottom_up(images)
        P3, P4, P5, P6, P7 = self.top_down((C3, C4, C5))
        classification_output = []
        regression_output = []
        for P in [P3, P4, P5, P6, P7]:
            classification_output.append(self.classification_subnet(P))
            regression_output.append(self.regression_subnet(P))

        classification = torch.cat(classification_output, dim=1)
        regression = torch.cat(regression_output, dim=1)

        return classification, regression, [P3, P4, P5, P6, P7]

    def predict(self, images):
        """
        Args:
            images: Tensor of (B, 3, H, W), where B is the batch size; H, W is image height, width

        Returns:
            list of [bboxes, labels, scores] per image
        """
        # C3, C4, C5 = self.bottom_up(images)
        # P3, P4, P5, P6, P7 = self.top_down((C3, C4, C5))
        # classification_output = []
        # regression_output = []
        # for P in [P3, P4, P5, P6, P7]:
        #     classification_output.append(self.classification_subnet(P))
        #     regression_output.append(self.regression_subnet(P))

        # classification = torch.cat(classification_output, dim=1)
        # regression = torch.cat(regression_output, dim=1)
        classification, regression, [P3, P4, P5, P6, P7] = self.__call__(images)
        print(classification.max().item())
        anchors = build_anchors(features=[P3, P4, P5, P6, P7])
        bboxes = bbox_transform_inv(anchors, regression)
        del anchors
        bboxes = clip_boxes(images, bboxes)

        return filter_detections(bboxes, classification)

    def train_extractor(self, active=True):
        if active:
            for p in self.bottom_up.parameters():
                p.requires_grad = True
            self.bottom_up.train()
        else:
            for p in self.bottom_up.parameters():
                p.requires_grad = False
            self.bottom_up.eval()
