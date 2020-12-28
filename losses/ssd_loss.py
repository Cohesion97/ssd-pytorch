import torch

def loss(cla_scores, loc_results, gt_bboxes, gt_labels, batch_size):
    """
    :param cla_scores: list[Tensor] (B, num_anchor*num_classes+1, h, w)
    :param loc_results: list[Tensor] (B, num_anchor*4, h, w)
    :param gt_bboxes: list[Tensor]
    :param gt_labels: list[Tensor]
    :return:
    """
    multi_level_anchors = self.grid_anchors()  # list[Tensor: num_levels, featmap_size, 4]
    anchors_list = [multi_level_anchors for _ in range(batch_size)]

    (labels_list, bboxes_target_list, num_pos_total, num_neg_total) = self.match_(
        anchors_list, gt_bboxes, gt_labels, batch_size)
    num_total_samples = num_neg_total + num_pos_total
    all_cla_scores = torch.cat([
        s.permute(0, 2, 3, 1).reshape(
            batch_size, -1, self.num_classes + 1) for s in cla_scores
    ], 1)

    all_bbox_preds = torch.cat([
        b.permute(0, 2, 3, 1).reshape(batch_size, -1, 4)
        for b in loc_results
    ], -2)
    all_bbox_targets = torch.cat(bboxes_target_list, -2).view(batch_size, -1, 4)

    all_anchors = []
    for i in range(batch_size):
        all_anchors.append(torch.cat(anchors_list[i]))
    all_labels = torch.cat(labels_list, -1).view(batch_size, -1)

    # check NaN and Inf
    assert torch.isfinite(all_cla_scores).all().item(), \
        'classification scores become infinite or NaN!'
    assert torch.isfinite(all_bbox_preds).all().item(), \
        'bbox predications become infinite or NaN!'

    loss_cla, loss_bbox = multi_apply(self.loss_single_img, all_cla_scores, all_bbox_preds, all_anchors, all_labels,
                                      all_bbox_targets, num_total_samples=num_pos_total)
    return loss_cla, loss_bbox