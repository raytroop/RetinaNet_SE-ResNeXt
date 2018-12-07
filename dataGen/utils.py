import numpy as np

def iou(box1, box2):
    """
    From Yicheng Chen's "Mean Average Precision Metric"
    https://www.kaggle.com/chenyc15/mean-average-precision-metric

    helper function to calculate IoU
    """
    x11, y11, x12, y12 = box1
    x21, y21, x22, y22 = box2
    w1, h1 = x12-x11, y12-y11
    w2, h2 = x22-x21, y22-y21

    area1, area2 = w1 * h1, w2 * h2
    xi1, yi1, xi2, yi2 = max([x11, x21]), max(
        [y11, y21]), min([x12, x22]), min([y12, y22])

    if xi2 <= xi1 or yi2 <= yi1:
        return 0
    else:
        intersect = (xi2-xi1) * (yi2-yi1)
        union = area1 + area2 - intersect
        return intersect / union


def nms(boxes, scores, overlapThresh):
    """
    adapted from non-maximum suppression by Adrian Rosebrock
    https://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
    """

    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return np.array([]).reshape(0, 4), np.array([])
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    pick = []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # compute the area of the bounding boxes
    area = (x2 - x1 + 1) * (y2 - y1 + 1)

    # sort the bounding boxes by scores in ascending order
    idxs = np.argsort(scores)

    # keep looping while indexes still remain in the indexes list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlapThresh)[0])))

    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick], scores[pick]
