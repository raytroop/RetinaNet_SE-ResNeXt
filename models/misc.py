import os
import json
import numpy as np
import torch
from dataGen.targetBuild import generate_anchors, AnchorParameters

config_path = os.path.join(os.path.dirname(__file__), '..', 'config.json')
with open(config_path, 'r') as f:
    config = json.load(f)
# torch.device object used throughout this script
device = torch.device("cuda" if config['use_cuda'] else "cpu")


def shift(shape, stride, anchors):
    """ Produce shifted anchors based on shape of the map and stride size.

    Args
        shape  : Shape to shift the anchors over (H, W).
        stride : Stride to shift the anchors with over the shape.
        anchors: The anchors to apply at each location, np.ndarry.
    """
    H, W = shape
    shift_x = (torch.arange(W, dtype=torch.float32, device=device) + torch.tensor(0.5, dtype=torch.float32, device=device)) * stride
    shift_y = (torch.arange(H, dtype=torch.float32, device=device) + torch.tensor(0.5, dtype=torch.float32, device=device)) * stride
    shift_y, shift_x = torch.meshgrid([shift_y, shift_x])
    shift_x = shift_x.contiguous().view(-1)
    shift_y = shift_y.contiguous().view(-1)
    shifts = torch.stack([shift_x, shift_y, shift_x, shift_y])
    shifts = shifts.t()
    anchors = torch.tensor(anchors, dtype=torch.float32, device=device)

    shifted_anchors = torch.unsqueeze(anchors, 0) + torch.unsqueeze(shifts, 1)
    shifted_anchors = shifted_anchors.view(-1, 4)

    return shifted_anchors


def build_anchors(features, anchor_params=None):
    # if no anchor parameters are passed, use default values
    if anchor_params is None:
        anchor_params = AnchorParameters.default
    ratios = anchor_params.ratios
    scales = anchor_params.scales
    sizes = anchor_params.sizes
    strides = anchor_params.strides
    anchors_bag = []
    for feature, size, stride in zip(features, sizes, strides):
        shape = feature.shape[-2:]
        anchors = generate_anchors(size, ratios, scales)
        anchors_shift = shift(shape, stride, anchors)
        anchors_bag.append(anchors_shift)

    return torch.cat(anchors_bag, dim=0)


def bbox_transform_inv(anchors, regression, mean=None, std=None):
    """ Applies deltas (usually regression results) to boxes (usually anchors).

    Before applying the deltas to the boxes, the normalization that was previously applied (in the generator) has to be removed.
    The mean and std are the mean and std as applied in the generator. They are unnormalized in this function and then applied to the boxes.

    Args
        anchors : Tensor of shape (N, 4), N the number of boxes and 4 values for (x1, y1, x2, y2).
        regression: Tensor of (B, N, 4), where B is the batch size, N the number of boxes.
                    These deltas (d_x1, d_y1, d_x2, d_y2) are a factor of the width/height.
        mean  : The mean value used when computing deltas (defaults to [0, 0, 0, 0]).
        std   : The standard deviation used when computing deltas (defaults to [0.2, 0.2, 0.2, 0.2]).

    Returns
        A Tensor of the same shape as boxes, but with deltas applied to each box.
        The mean and std are used during training to normalize the regression values (networks love normalization).
    """

    if mean is None:
        mean = config['mean_bbox_transform']
    if std is None:
        std = config['std_bbox_transform']

    anchors = torch.unsqueeze(anchors, dim=0)  # (1, N, 4)
    width = anchors[:, :, 2] - anchors[:, :, 0]
    height = anchors[:, :, 3] - anchors[:, :, 1]

    x1 = anchors[:, :, 0] + (regression[:, :, 0] * std[0] + mean[0]) * width
    y1 = anchors[:, :, 1] + (regression[:, :, 1] * std[1] + mean[1]) * height
    x2 = anchors[:, :, 2] + (regression[:, :, 2] * std[2] + mean[2]) * width
    y2 = anchors[:, :, 3] + (regression[:, :, 3] * std[3] + mean[3]) * height

    pred_boxes = torch.stack([x1, y1, x2, y2], dim=2)

    return pred_boxes


def clip_boxes(images, boxes):
    shape = images.shape
    height = shape[-2]
    width = shape[-1]

    x1 = torch.clamp(boxes[:, :, 0], 0.0, width)
    y1 = torch.clamp(boxes[:, :, 1], 0.0, height)
    x2 = torch.clamp(boxes[:, :, 2], 0.0, width)
    y2 = torch.clamp(boxes[:, :, 3], 0.0, height)
    boxes_x1y1x2y2 = torch.stack([x1, y1, x2, y2], dim=2)
    boxes_x1y1x2y2 = boxes_x1y1x2y2.type(torch.int64)
    return boxes_x1y1x2y2


def bbox_iou(box1, box2):
    """
    Returns the IoU of two bounding boxes
    """
    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0
    )
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou


def non_max_suppression(boxes, scores, max_output_size=300, iou_threshold=0.5):
    """
    Removes detections with lower object confidence score than 'conf_thres' and performs
    Non-Maximum Suppression to further filter detections.
    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, class_score, class_pred)
    """

    # Sort the detections by maximum objectness confidence
    _, conf_sort_index = torch.sort(scores, descending=True)
    boxes = boxes[conf_sort_index]
    # Perform non-maximum suppression
    max_indexes = []
    count = 0
    while boxes.shape[0] > 0:
        # Get detection with highest confidence and save as max detection
        max_detections = boxes[0].unsqueeze(0)  # expand 1 dim
        max_indexes.append(conf_sort_index[0])
        # Stop if we're at the last detection
        if boxes.shape[0] == 1:
            break
        # Get the IOUs for all boxes with lower confidence
        ious = bbox_iou(max_detections, boxes[1:])
        # Remove detections with IoU >= NMS threshold
        boxes = boxes[1:][ious < iou_threshold]
        conf_sort_index = conf_sort_index[1:][ious < iou_threshold]
        # break when get enough bboxes
        count += 1
        if count >= max_output_size:
            break

    # max_detections = torch.cat(max_detections).data
    max_indexes = torch.stack(max_indexes).data
    return max_indexes


def filter_detections(
        boxes,
        classification,
        class_specific_filter=True,
        nms=True,
        score_threshold=0.01,
        max_detections=300,
        nms_threshold=0.5
):
    """ Filter detections using the boxes and classification values.

    Args
        boxes                 : Tensor of shape (B, num_boxes, 4) containing the boxes in (x1, y1, x2, y2) format.
        classification        : Tensor of shape (B, num_boxes, num_classes) containing the classification scores.
        class_specific_filter : Whether to perform filtering per class, or take the best scoring class and filter those.
        nms                   : Flag to enable/disable non maximum suppression.
        score_threshold       : Threshold used to prefilter the boxes with.
        max_detections        : Maximum number of detections to keep.
        nms_threshold         : Threshold for the IoU value to determine when a box should be suppressed.

    Returns
        A list of [boxes, scores, labels, other[0], other[1], ...].
        boxes is shaped (max_detections, 4) and contains the (x1, y1, x2, y2) of the non-suppressed boxes.
        scores is shaped (max_detections,) and contains the scores of the predicted class.
        labels is shaped (max_detections,) and contains the predicted label.
        other[i] is shaped (max_detections, ...) and contains the filtered other[i] data.
        In case there are less than max_detections detections, the tensors are padded with -1's.
    """

    def _filter_detections(boxes, scores, labels):
        # threshold based on score
        indices = torch.gt(scores, score_threshold).nonzero()
        if indices.shape[0] == 0:
            return torch.tensor([], dtype=torch.int64, device=device)
        indices = indices[:, 0]

        if nms:
            filtered_boxes = torch.index_select(boxes, 0, indices)
            filtered_scores = torch.index_select(scores, 0, indices)

            # perform NMS
            nms_indices = non_max_suppression(filtered_boxes, filtered_scores, max_output_size=max_detections,
                                              iou_threshold=nms_threshold)

            # filter indices based on NMS
            indices = torch.index_select(indices, 0, nms_indices)

        # add indices to list of all indices
        labels = torch.index_select(labels, 0, indices)
        indices = torch.stack([indices, labels], dim=1)

        return indices

    results = []
    for box_cur, classification_cur in zip(boxes, classification):
        if class_specific_filter:
            all_indices = []
            # perform per class filtering
            for c in range(int(classification_cur.shape[1])):
                scores = classification_cur[:, c]
                labels = torch.full_like(scores, c, dtype=torch.int64)
                all_indices.append(_filter_detections(box_cur, scores, labels))

            # concatenate indices to single tensor
            indices = torch.cat(all_indices, dim=0)
        else:
            scores, labels = torch.max(classification_cur, dim=1)
            indices = _filter_detections(box_cur, scores, labels)

        if indices.shape[0] == 0:
            results.append({'bboxes':np.zeros((0, 4)), 'scores': np.full((0, ), -1, dtype=np.float32),'category_id': np.full((0, ), -1, dtype=np.int64)})
            continue
        # select top k
        scores = classification_cur[indices[:, 0], indices[:, 1]]
        labels = indices[:, 1]
        indices = indices[:, 0]

        scores, top_indices = torch.topk(scores, k=min(max_detections, scores.shape[0]))
        # filter input using the final set of indices
        indices = indices[top_indices]
        box_cur = box_cur[indices]
        labels = labels[top_indices]
        results.append({'bboxes':box_cur.cpu().detach().numpy(),'scores': scores.cpu().detach().numpy(), 'category_id': labels.cpu().detach().numpy()})

    return results
