import numpy as np

def bbox_overlaps(bbox1, bbox2, eps=1e-6):
    """
    :param bbox1: np.array n, 4
    :param bbox2: np.array k, 4
    :param eps:
    :return: overlaps: np.array n,k
    """

    try:
        _ = bbox1.shape[1]
    except:
        bbox1 = bbox1.reshape(-1,4)
    rows = bbox1.shape[0]
    cols = bbox2.shape[0]
    bboxes1 = bbox1.astype(np.float32)
    bboxes2 = bbox2.astype(np.float32)
    ious = np.zeros((rows, cols), dtype=np.float32)

    if rows * cols == 0:
        return ious
    exchange = False
    if bboxes1.shape[0] > bboxes2.shape[0]:
        bboxes1, bboxes2 = bboxes2, bboxes1
        ious = np.zeros((cols, rows), dtype=np.float32)
        exchange = True
    area1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (bboxes1[:, 3] - bboxes1[:, 1])
    area2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (bboxes2[:, 3] - bboxes2[:, 1])
    for i in range(bboxes1.shape[0]):
        x_start = np.maximum(bboxes1[i, 0], bboxes2[:, 0])
        y_start = np.maximum(bboxes1[i, 1], bboxes2[:, 1])
        x_end = np.minimum(bboxes1[i, 2], bboxes2[:, 2])
        y_end = np.minimum(bboxes1[i, 3], bboxes2[:, 3])
        overlap = np.maximum(x_end - x_start, 0) * np.maximum(
            y_end - y_start, 0)
        union = area1[i] + area2 - overlap

        union = np.maximum(union, eps)
        ious[i, :] = overlap / union
    if exchange:
        ious = ious.T
    return ious