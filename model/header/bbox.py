import numpy as np
import torch

class Bbox(object):
    def __init__(self,
                 means = (0., 0., 0., 0.),
                 stds = (0.1, 0.1, 0.2, 0.2)):
        self.means = means
        self.stds = stds

    def bbox_encoder(self, proposals, gt,):
        """
        :param proposals: Tensor: num_anchors, 4
        :param gt: Tensor: num_anchors, 4
        :param mean:
        :param std:
        :return:
        """
        assert proposals.size() == gt.size()

        proposals = proposals.float()
        gt = gt.float()
        px = (proposals[..., 0] + proposals[..., 2]) * 0.5
        py = (proposals[..., 1] + proposals[..., 3]) * 0.5
        pw = proposals[..., 2] - proposals[..., 0]
        ph = proposals[..., 3] - proposals[..., 1]

        gx = (gt[..., 0] + gt[..., 2]) * 0.5
        gy = (gt[..., 1] + gt[..., 3]) * 0.5
        gw = gt[..., 2] - gt[..., 0]
        gh = gt[..., 3] - gt[..., 1]

        dx = (gx - px) / pw
        dy = (gy - py) / ph
        dw = torch.log(gw / pw)
        dh = torch.log(gh / ph)
        deltas = torch.stack([dx, dy, dw, dh], dim=-1)

        means = deltas.new_tensor(self.means).unsqueeze(0)
        stds = deltas.new_tensor(self.stds).unsqueeze(0)
        deltas = deltas.sub_(means).div_(stds)
        return deltas # [num_anchors, 4]

    def bbox_decode(self,
                    bboxes,
                    pred_bboxes,
                    max_shape=None,
                    wh_ratio_clip=16 / 1000):
        """Apply transformation `pred_bboxes` to `boxes`.

        Args:
            boxes (torch.Tensor): Basic boxes.
            pred_bboxes (torch.Tensor): Encoded boxes with shape
            max_shape (tuple[int], optional): Maximum shape of boxes.
                Defaults to None.
            wh_ratio_clip (float, optional): The allowed ratio between
                width and height.

        Returns:
            torch.Tensor: Decoded boxes.
        """

        assert pred_bboxes.size(0) == bboxes.size(0)
        decoded_bboxes = delta2bbox(bboxes, pred_bboxes, self.means, self.stds,
                                    max_shape, wh_ratio_clip, True)

        return decoded_bboxes

def delta2bbox(rois,
               deltas,
               means=(0., 0., 0., 0.),
               stds=(1., 1., 1., 1.),
               max_shape=None,
               wh_ratio_clip=16 / 1000,
               clip_border=True):
    """Apply deltas to shift/scale base boxes.

    Typically the rois are anchor or proposed bounding boxes and the deltas are
    network outputs used to shift/scale those boxes.
    This is the inverse function of :func:`bbox2delta`.

    Args:
        rois (Tensor): Boxes to be transformed. Has shape (N, 4)
        deltas (Tensor): Encoded offsets with respect to each roi.
            Has shape (N, 4 * num_classes). Note N = num_anchors * W * H when
            rois is a grid of anchors. Offset encoding follows [1]_.
        means (Sequence[float]): Denormalizing means for delta coordinates
        stds (Sequence[float]): Denormalizing standard deviation for delta
            coordinates
        max_shape (tuple[int, int]): Maximum bounds for boxes. specifies (H, W)
        wh_ratio_clip (float): Maximum aspect ratio for boxes.
        clip_border (bool, optional): Whether clip the objects outside the
            border of the image. Defaults to True.

    Returns:
        Tensor: Boxes with shape (N, 4), where columns represent
            tl_x, tl_y, br_x, br_y.

    References:
        .. [1] https://arxiv.org/abs/1311.2524
    """
    means = deltas.new_tensor(means).view(1, -1).repeat(1, deltas.size(1) // 4)
    stds = deltas.new_tensor(stds).view(1, -1).repeat(1, deltas.size(1) // 4)
    denorm_deltas = deltas * stds + means
    dx = denorm_deltas[:, 0::4]
    dy = denorm_deltas[:, 1::4]
    dw = denorm_deltas[:, 2::4]
    dh = denorm_deltas[:, 3::4]
    max_ratio = np.abs(np.log(wh_ratio_clip))
    dw = dw.clamp(min=-max_ratio, max=max_ratio)
    dh = dh.clamp(min=-max_ratio, max=max_ratio)
    # Compute center of each roi
    px = ((rois[:, 0] + rois[:, 2]) * 0.5).unsqueeze(1).expand_as(dx)
    py = ((rois[:, 1] + rois[:, 3]) * 0.5).unsqueeze(1).expand_as(dy)
    # Compute width/height of each roi
    pw = (rois[:, 2] - rois[:, 0]).unsqueeze(1).expand_as(dw)
    ph = (rois[:, 3] - rois[:, 1]).unsqueeze(1).expand_as(dh)
    # Use exp(network energy) to enlarge/shrink each roi
    gw = pw * dw.exp()
    gh = ph * dh.exp()
    # Use network energy to shift the center of each roi
    gx = px + pw * dx
    gy = py + ph * dy
    # Convert center-xy/width/height to top-left, bottom-right
    x1 = gx - gw * 0.5
    y1 = gy - gh * 0.5
    x2 = gx + gw * 0.5
    y2 = gy + gh * 0.5
    if clip_border and max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1])
        y1 = y1.clamp(min=0, max=max_shape[0])
        x2 = x2.clamp(min=0, max=max_shape[1])
        y2 = y2.clamp(min=0, max=max_shape[0])
    bboxes = torch.stack([x1, y1, x2, y2], dim=-1).view(deltas.size())
    return bboxes