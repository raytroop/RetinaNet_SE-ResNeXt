import os
import pickle
import json
import torch
import numpy as np
from models.misc import filter_detections


def test_simple():
    # create simple FilterDetections layer

    # create simple input
    boxes = np.array([[
        [0, 0, 10, 10],
        [0, 0, 10, 10],  # this will be suppressed
    ]], dtype=np.float)
    boxes = torch.tensor(boxes)

    classification = np.array([[
        [0, 0.9],  # this will be suppressed
        [0, 1],
    ]], dtype=np.float)
    classification = torch.tensor(classification)

    # compute output
    results = filter_detections(boxes, classification)
    actual_boxes  = results[0][0]
    actual_scores = results[0][1]
    actual_labels = results[0][2]

    # define expected output
    expected_boxes = np.array([[0, 0, 10, 10]], dtype=np.float)

    expected_scores = np.array([1], dtype=np.float)

    expected_labels = np.array([1], dtype=np.float)

    # assert actual and expected are equal
    np.testing.assert_array_equal(actual_boxes, expected_boxes)
    np.testing.assert_array_equal(actual_scores, expected_scores)
    np.testing.assert_array_equal(actual_labels, expected_labels)


def test_mini_batch():
    # create simple FilterDetections layer

    # create input with batch_size=2
    boxes = np.array([
        [
            [0, 0, 10, 10],  # this will be suppressed
            [0, 0, 10, 10],
        ],
        [
            [100, 100, 150, 150],
            [100, 100, 150, 150],  # this will be suppressed
        ],
    ], dtype=np.float)
    boxes = torch.tensor(boxes)

    classification = np.array([
        [
            [0, 0.9],  # this will be suppressed
            [0, 1],
        ],
        [
            [1,   0],
            [0.9, 0],  # this will be suppressed
        ],
    ], dtype=np.float)
    classification = torch.tensor(classification)

    # compute output
    results = filter_detections(boxes, classification)


    # define expected output
    expected_boxes0 = np.array([[0, 0, 10, 10]], dtype=np.float)
    expected_boxes1 = np.array([[100, 100, 150, 150]], dtype=np.float)

    expected_scores0 = np.array([1], dtype=np.float)
    expected_scores1 = np.array([1], dtype=np.float)

    expected_labels0 = np.array([1], dtype=np.float)
    expected_labels1 = np.array([0], dtype=np.float)

    # assert actual and expected are equal
    np.testing.assert_array_equal(results[0][0], expected_boxes0)
    np.testing.assert_array_equal(results[0][1], expected_scores0)
    np.testing.assert_array_equal(results[0][2], expected_labels0)

    # assert actual and expected are equal
    np.testing.assert_array_equal(results[1][0], expected_boxes1)
    np.testing.assert_array_equal(results[1][1], expected_scores1)
    np.testing.assert_array_equal(results[1][2], expected_labels1)
