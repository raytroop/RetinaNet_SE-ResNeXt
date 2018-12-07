import os
import json
import torch
from torch.nn import functional as F

config_path = os.path.join(os.path.dirname(__file__), '..', 'config.json')
with open(config_path, 'r') as f:
    config = json.load(f)
# torch.device object used throughout this script
device = torch.device("cuda" if config['use_cuda'] else "cpu")

def focal_loss(alpha=0.25, gamma=2.0):
    """ Create a functor for computing the focal loss.

    Args
        alpha: Scale the focal weight with alpha.
        gamma: Take the power of the focal weight with gamma.

    Returns
        A functor that computes the focal loss using the alpha and gamma.
    """

    def _focal(y_true, y_pred):
        """ Compute the focal loss given the target tensor and the predicted tensor.

        As defined in https://arxiv.org/abs/1708.02002

        Args
            y_true: Tensor of target data from the generator with shape (B, N, num_classes+1).
            y_pred: Tensor of predicted data from the network with shape (B, N, num_classes).

        Returns
            The focal loss of y_pred w.r.t. y_true.
        """
        labels = y_true[:, :, :-1]
        anchor_state = y_true[:, :, -1]  # -1 for ignore, 0 for background, 1 for object
        classification = y_pred

        # filter out "ignore" anchors
        indices = anchor_state != -1
        labels = labels[indices]
        classification = classification[indices]

        # compute the focal loss
        alpha_factor = torch.where(labels == 1, torch.tensor(alpha, dtype=torch.float32, device=device),
                                   torch.tensor(1 - alpha, dtype=torch.float32, device=device))
        focal_weight = torch.where(labels == 1, 1 - classification, classification)
        focal_weight = alpha_factor * focal_weight ** gamma

        cls_loss = focal_weight * F.binary_cross_entropy(classification, labels, reduction='none')
        # compute the normalizer: the number of positive anchors
        normalizer = torch.sum(anchor_state == 1)
        normalizer = normalizer.type(torch.float32)
        normalizer = torch.max(normalizer, torch.tensor(1.0, device=device))

        return torch.sum(cls_loss) / normalizer

    return _focal


def smooth_l1_loss(sigma=3.0):
    """ Create a smooth L1 loss functor.

    Args
        sigma: This argument defines the point where the loss changes from L2 to L1.

    Returns
        A functor for computing the smooth L1 loss given target data and predicted data.
    """
    sigma_squared = sigma ** 2

    def _smooth_l1(y_true, y_pred):
        """ Compute the smooth L1 loss of y_pred w.r.t. y_true.

        Args
            y_true: Tensor from the generator of shape (B, N, 5). The last value for each box is the state of the anchor (ignore, negative, positive).
            y_pred: Tensor from the network of shape (B, N, 4).

        Returns
            The smooth L1 loss of y_pred w.r.t. y_true.
        """
        # separate target and state
        regression = y_pred
        regression_target = y_true[:, :, :4]
        anchor_state = y_true[:, :, 4]

        # filter out "ignore" anchors and "bg"
        indices = anchor_state == 1
        regression = regression[indices]
        regression_target = regression_target[indices]

        # compute smooth L1 loss
        # f(x) = 0.5 * (sigma * x)^2          if |x| < 1 / sigma / sigma
        #        |x| - 0.5 / sigma / sigma    otherwise
        regression_diff = regression - regression_target
        regression_diff = torch.abs(regression_diff)
        regression_loss = torch.where(
            torch.lt(regression_diff, 1.0 / sigma_squared),
            0.5 * sigma_squared * torch.pow(regression_diff, 2),
            regression_diff - 0.5 / sigma_squared
        )

        # compute the normalizer: the number of positive anchors
        normalizer = torch.max(torch.tensor(1, device=device), torch.sum(indices))
        normalizer = normalizer.type(torch.float32)
        return torch.sum(regression_loss) / normalizer

    return _smooth_l1
