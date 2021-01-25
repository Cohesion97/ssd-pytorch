import torch
import torch.nn.functional as F
from losses.smooth_l1_loss import smooth_l1_loss
from model.init_weights import *
from utils.IoU_calculate.IoU_assign import IoU_assigner
from functools import partial
from six.moves import map, zip
from utils.nms.bbox_nms import multiclass_nms
from .bbox import Bbox

def pairs(x):
    assert type(x) != str
    return [x,x]

def multi_apply(func, *args, **kwargs):
    """Apply function to a list of arguments.

    Note:
        This function applies the ``func`` to multiple inputs and
        map the multiple outputs of the ``func`` into different
        list. Each list contains the same type of outputs corresponding
        to different inputs.

    Args:
        func (Function): A function that will be applied to a list of
            arguments

    Returns:
        tuple(list): A tuple containing multiple list, each list contains \
            a kind of returned results by the function
    """
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results))) #转置矩阵后划分

def image_to_level(target, num_anchors_level):
    """
    Turn [anchor_img0, anchor_img1..] to [anchor_level0, anchor_level2..]
         List length=batch size           List length=level length
         (num_anchors,4)                  (batch_size, num_level_anchors, 4)
    :param target:
    :param num_anchors_lever:
    :return:
    """
    target = torch.stack(target, 0)
    level_targets = []
    start = 0
    for n in num_anchors_level:
        end = start + n
        # level_targets.append(target[:, start:end].squeeze(0))
        level_targets.append(target[:, start:end])
        start = end
    return level_targets

class ssd_header(nn.Module):

    default_inplanes = [512, 1024, 512,256,256,256]

    def __init__(self,
                 num_classes,
                 anchor_cfg={'strides':[8,16,32,64,100,300],
                             'ratios':[[2],[2,3],[2,3],[2,3],[2,],[2]],
                             'scale_range':[0.2, 0.9],
                             'input_size':300},
                 bbox_cfg={'means': (0., 0., 0., 0.), 'stds': (0.1, 0.1, 0.2, 0.2)}
                 ,
                 neg_pos_rate=3
                 ):
        super(ssd_header, self).__init__()
        self.num_classes = num_classes
        self.anchor_cfg = anchor_cfg
        assert len(self.default_inplanes)==len(self.anchor_cfg['strides'])
        self.num_multi_feats = len(self.default_inplanes)
        self.neg_pos_rate = neg_pos_rate

        # anchor config
        self.anchor_strides = self.anchor_cfg['strides']
        self.anchor_ratios = self.anchor_cfg['ratios']
        self.anchor_input_size = self.anchor_cfg['input_size']
        self.featmap_size = [38, 19, 10, 5, 3, 1]

        # gen anchors
        self.base_len = [int(0.1 * self.anchor_input_size)]
        small, big = self.anchor_cfg['scale_range']
        self.step = (big - small) / (self.num_multi_feats - 2)
        self.step_ = int(self.step * self.anchor_input_size)
        for i in range(int(small*100), int(big*100+1), int(self.step*100)):
            self.base_len.append(int(i * self.anchor_input_size / 100))
        gen_ratios = []
        gen_scales = []
        for j, item in enumerate(self.base_len):
            try:
                gen_scale = torch.Tensor([1.,
                              np.sqrt(self.base_len[j+1] /
                                      self.base_len[j])])
            except:
                gen_scale = torch.Tensor([1.,
                              np.sqrt((self.base_len[j]+self.step_) /
                                      self.base_len[j])])
            gen_scales.append(gen_scale
                )
            gen_ratio = [1.,]
            for k in self.anchor_ratios[j]:
                gen_ratio.extend([1/k, k])
            gen_ratios.append(
                torch.Tensor(gen_ratio)
            )
        self.gen_ratios = gen_ratios
        self.gen_scales = gen_scales
        self.base_anchors = self.gen_base_anchors()
        self.num_anchors = [len(num_anchor) for num_anchor in self.base_anchors ]

        # bbox config
        self.bbox_menas = bbox_cfg['means']
        self.bbox_stds = bbox_cfg['stds']
        self.bbox = Bbox(means=self.bbox_menas, stds=self.bbox_stds)

        # forward convs
        cla_convs = []
        loc_convs = []
        for i in range(len(self.default_inplanes)):
            cla_convs.append(nn.Conv2d(self.default_inplanes[i],
                                       (self.num_classes+1)*self.num_anchors[i],
                                       kernel_size=3,padding=1))
            loc_convs.append(nn.Conv2d(self.default_inplanes[i],
                                       4*self.num_anchors[i],
                                       kernel_size=3,padding=1))
        self.loc_convs = nn.ModuleList(loc_convs)
        self.cla_convs = nn.ModuleList(cla_convs)

    @property
    def num_base_anchors(self):
        return [base_anchors.size(0) for base_anchors in self.base_anchors]

    def forward(self,feats):
        cla_scores = []
        loc_results = []
        for feat, cla_layer, loc_layer in zip(feats, self.cla_convs, self.loc_convs):
            cla_scores.append(cla_layer(feat))
            loc_results.append(loc_layer(feat))
        return cla_scores, loc_results

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform', bias=0)

    def gen_anchor_single_feat(self,base_len,stride,ratios,scales):
        w, h = base_len
        w_center = int(stride/2)
        h_center = int(stride/2)

        h_ratio = torch.sqrt(ratios)
        w_ratio = 1 / h_ratio

        d_w = w * scales[:,None] * w_ratio[None,:]
        d_h = h * scales[:,None] * h_ratio[None,:]

        d_w = d_w.view(-1)
        d_h = d_h.view(-1)

        anchors = [w_center - 0.5 * d_w,
                   h_center - 0.5 * d_h,
                   w_center + 0.5 * d_w,
                   h_center + 0.5 * d_h,
                   ]

        anchors = torch.stack(anchors, dim=-1)

        return anchors

    def gen_base_anchors(self,device='cuda'):
        mulit_feat = []
        for i in range(len(self.base_len)):
            base_anchors=self.gen_anchor_single_feat(pairs(self.base_len[i]),
                                                          self.anchor_strides[i],
                                                          self.gen_ratios[i],
                                                          self.gen_scales[i])
            indices = list(range(len(self.gen_ratios[i])))
            indices.insert(1,len(indices))
            base_anchors = torch.index_select(base_anchors.to(device), 0, torch.LongTensor(indices).to(device))
            mulit_feat.append(base_anchors)
        return mulit_feat

    def grid_anchors(self,clamp=False,device='cuda'):
        """
        :argument:
            self.base_anchors: List[Tensor]
            self.anchor_strides: List[int]
        :return:
            multi_grid_anchors: list[Tensor: num_levels, featmap_size, 4]
        """
        multi_grid_anchors = []
        for num_levels in range(len(self.base_anchors)):
            x = torch.from_numpy(np.array(
                range(0, self.anchor_input_size, self.anchor_strides[num_levels]))).to(device)
            y = torch.from_numpy(np.array(
                range(0, self.anchor_input_size, self.anchor_strides[num_levels]))).to(device)
            x_, y_ = self.shift(x,y)
            shift = [x_, y_, x_, y_]
            shift = torch.stack(shift, dim=1)
            shift_=shift.type_as(self.base_anchors[num_levels].to(device))
            shift_anchors = self.base_anchors[num_levels].to(device)[None,:,:]+shift_[:,None,:]
            shift_anchors = shift_anchors.view(-1,4)
            if clamp:
                shift_anchors = shift_anchors.clamp(min=0,max=300)
            multi_grid_anchors.append(shift_anchors)
        return multi_grid_anchors

    def shift(self,x,y):
        """
        :param x: Tensor, w of featmap
        :param y: int, h of featmap
        :return: shift: Tensor, wh * 4
        """
        x_ = x.repeat(len(y))
        y_ = y.view(-1,1).repeat(1,len(x)).view(-1)

        return x_, y_

    def match_(self, anchors_list, valid_flag_list, gt_bboxes, gt_labels, batch_size):
        """
        match anchors with gt_bboxes in a batch of imgs
        :param anchors_list: list[list[Tensor: num_levels, featmap_size, 4]]
        :param gt_bboxes: list[Tensor]
        :param gt_labels: list[Tensor]
        :return:
        """
        num_anchors_level = [anchors.size(0) for anchors in anchors_list[0]]
        concat_anchor_list = []
        concat_valid_flag_list = []
        for i in range(batch_size):
            concat_anchor_list.append(torch.cat(anchors_list[i])) # list[Tensor: num_total_anchors, 4]
            concat_valid_flag_list.append(torch.cat(valid_flag_list[i]))

        result = multi_apply(self.match_single_img,
                             concat_anchor_list, concat_valid_flag_list,
                             gt_bboxes, gt_labels, batch_size=batch_size)
        (all_labels, all_label_weights, all_bbox_targets,
         all_bbox_weights, all_pos_idx, all_neg_idx) = result
        num_pos_total = sum([max(inds.numel(),1) for inds in all_pos_idx])
        num_neg_total = sum([max(inds.numel(),1) for inds in all_neg_idx])

        labels_list = image_to_level(all_labels, num_anchors_level)
        label_weights_list = image_to_level(all_label_weights, num_anchors_level)
        bboxes_target_list = image_to_level(all_bbox_targets, num_anchors_level)
        bboxes_weights_list = image_to_level(all_bbox_weights,
                                             num_anchors_level)
        return (labels_list, label_weights_list, bboxes_target_list, bboxes_weights_list, num_pos_total, num_neg_total)

    def match_single_img(self, flat_anchors, valid_flags, gt_bboxes, gt_labels, batch_size=4):
        """
        match anchors with gt_bboxes in one img
        :param flat_anchors: Tensor: num_anchors, 4
        :param gt_bboxes: Tensor: num_gt, 4
        :param gt_labels: Tensor: num_gt,
        :param batch_size: int
        :return: labels: Tensor: num_anchors, ; num_classes(background) when not assigned with gt
                bbox_targets: Tensor: num_anchors, 4; 0 when not assigned
                pos_idx: list:
        """
        inside_flags = valid_flags
        anchors = flat_anchors[inside_flags, :]
        assigner = IoU_assigner()
        # [num_anchors, ],
        assign_gt_idx, assign_label = assigner.assign(gt_bboxes, anchors, gt_labels)
        # len(pos_idx)+len(neg_idx)=num_anchors
        pos_idx = torch.nonzero(assign_gt_idx>0, as_tuple=False).squeeze(-1)
        neg_idx = torch.nonzero(assign_gt_idx==0, as_tuple=False).squeeze(-1)
        pos_bboxs = anchors[pos_idx]
        neg_bboxs = anchors[neg_idx]
        pos_gt_bboxes = gt_bboxes[assign_gt_idx[pos_idx]-1]

        num_valid_anchors = anchors.shape[0]
        bbox_targets = torch.zeros_like(anchors)
        #只有正样本的bbox需要算loss
        bbox_weights = torch.zeros_like(anchors)
        labels = anchors.new_full(
            (num_valid_anchors, ), self.num_classes , dtype=torch.long) #(num_anchors, )
        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)
        if len(pos_idx) > 0:
            pos_bbox_targets = self.bbox.bbox_encoder(pos_bboxs,pos_gt_bboxes)
            bbox_targets[pos_idx, :] = pos_bbox_targets
            bbox_weights[pos_idx, :] = 1.0
            if gt_labels is not None:
                labels[pos_idx] = gt_labels[assign_gt_idx[pos_idx]-1]
            else:
                labels[pos_idx] = 0
            label_weights[pos_idx] = 1.0
        if len(neg_idx) > 0:
            label_weights[neg_idx] = 1.0
        return (labels, label_weights, bbox_targets, bbox_weights, pos_idx, neg_idx)

    def valid_flags(self,device='cuda'):
        multi_level_flags = []
        for i in range(self.num_multi_feats):
            anchor_stride = pairs(self.anchor_strides[i])
            feat_h, feat_w = pairs(self.featmap_size[i])
            h, w = pairs(self.anchor_input_size)
            valid_feat_h = min(int(np.ceil(h / anchor_stride[1])), feat_h)
            valid_feat_w = min(int(np.ceil(w / anchor_stride[0])), feat_w)
            flags = self.single_level_valid_flags((feat_h, feat_w),
                                                  (valid_feat_h, valid_feat_w),
                                                  self.num_base_anchors[i],
                                                  device=device)
            multi_level_flags.append(flags)
        return multi_level_flags

    def single_level_valid_flags(self,
                                 featmap_size,
                                 valid_size,
                                 num_base_anchors,
                                 device='cuda'):
        feat_h, feat_w = featmap_size
        valid_h, valid_w = valid_size
        assert valid_h <= feat_h and valid_w <= feat_w
        valid_x = torch.zeros(feat_w, dtype=torch.bool, device=device)
        valid_y = torch.zeros(feat_h, dtype=torch.bool, device=device)
        valid_x[:valid_w] = 1
        valid_y[:valid_h] = 1
        valid_xx, valid_yy = self.shift(valid_x, valid_y)
        valid = valid_xx & valid_yy
        valid = valid[:, None].expand(valid.size(0),
                                      num_base_anchors).contiguous().view(-1)
        return valid

    def loss(self, cla_scores, loc_results, gt_bboxes, gt_labels):
        """
        :param cla_scores: list[Tensor] (B, num_anchor*num_classes+1, h, w)
        :param loc_results: list[Tensor] (B, num_anchor*4, h, w)
        :param gt_bboxes: list[Tensor]
        :param gt_labels: list[Tensor]
        :return:
        """
        multi_level_anchors = self.grid_anchors() # list[Tensor: num_levels, featmap_size, 4]
        batch_size = cla_scores[0].shape[0]
        anchors_list =[multi_level_anchors for _ in range(batch_size)]
        valid_flag_list=[]
        for i in range(batch_size):
            multi_level_flags = self.valid_flags()
            valid_flag_list.append(multi_level_flags)

        assert gt_labels != None
        (labels_list, labels_weight_list, bboxes_target_list,
         bboxes_weight_list, num_pos_total, num_neg_total) = \
            self.match_(anchors_list, valid_flag_list,gt_bboxes,
                        gt_labels, batch_size)
        num_total_samples = num_neg_total + num_pos_total
        all_cla_scores = torch.cat([
            s.permute(0, 2, 3, 1).reshape(
                batch_size, -1, self.num_classes+1) for s in cla_scores
        ], 1)

        all_bbox_preds = torch.cat([
            b.permute(0, 2, 3, 1).reshape(batch_size, -1, 4)
            for b in loc_results
        ], -2)
        all_bbox_targets = torch.cat(bboxes_target_list,-2).view(batch_size,-1,4)
        all_bbox_weights = torch.cat(bboxes_weight_list,-2).view(batch_size,-1,4)
        all_anchors = []
        for i in range(batch_size):
            all_anchors.append(torch.cat(anchors_list[i]))
        all_labels = torch.cat(labels_list, -1).view(batch_size, -1)
        all_label_weights = torch.cat(labels_weight_list,-1).view(batch_size, -1)


        # check NaN and Inf
        assert torch.isfinite(all_cla_scores).all().item(), \
            'classification scores become infinite or NaN!'
        assert torch.isfinite(all_bbox_preds).all().item(), \
            'bbox predications become infinite or NaN!'


        loss_cla, loss_bbox = multi_apply(self.loss_single_img, all_cla_scores, all_bbox_preds, all_bbox_weights,
                                          all_anchors, all_labels, all_label_weights,
                                           all_bbox_targets, num_total_samples=num_pos_total)
        return loss_cla, loss_bbox

    def loss_single_img(self, cla_scores,  bbox_preds, bbox_weights, anchors, labels, label_weights,
                                           bbox_targets, num_total_samples):
        """

        :param cla_scores: Tensor: num_total_anchor, num_classes+1
        :param bbox_preds: Tensor: num_total_anchor, 4 (dx,dy,dw,dh)
        :param anchors: Tensor: num_total_anchor, 4 (x1,y1,x2,y2)
        :param labels: Tensor: num_total, ; targets for anchors
        :param bbox_targets:  num_total_anchor,4 (dx,dy,dw,dh); targets for bbox preds
        :param num_total_samples:
        :return:
        """

        loss_cla_all = F.cross_entropy(cla_scores, labels, reduction='none') * label_weights
        # foreground: [0,num_class-1]; background: num_class
        pos_inds = ((labels >= 0) &
                    (labels < self.num_classes)).nonzero(as_tuple=False).reshape(-1)
        neg_inds = (labels == self.num_classes).nonzero(as_tuple=False).view(-1)
        num_pos_samples = pos_inds.size(0)
        num_neg_samples = self.neg_pos_rate * num_pos_samples
        if num_neg_samples > neg_inds.size(0):
            num_neg_samples = neg_inds.size(0)
        topk_loss_cls_neg, _ = loss_cla_all[neg_inds].topk(num_neg_samples)
        loss_cla_pos = loss_cla_all[pos_inds].sum()
        loss_cla_neg = topk_loss_cls_neg.sum()
        loss_cla = (loss_cla_pos + loss_cla_neg) / num_total_samples

        loss_bbox = smooth_l1_loss(bbox_preds, bbox_targets, bbox_weights,avg_factor=num_total_samples)
        from IPython import embed;embed()
        return loss_cla[None], loss_bbox


    def get_bboxes(self, cla_scores, bbox_preds, img_infos, with_nms=True, rescale=False):
        """
        Turn model ouputs to labeled bboxes
        :param cla_scores: list[Tensor] (B, num_anchor*num_classes+1, h, w)
        :param loc_results: list[Tensor] (B, num_anchor*4, h, w)
        :param with_nms:
        :return:
        """
        num_levels = len(cla_scores)
        device = cla_scores[0].device
        multi_level_anchors = self.grid_anchors()
        batchsize = cla_scores[0].shape[0]
        multi_levels = len(cla_scores)

        result = []
        for i in range(batchsize):
            cla_list = [cla_scores[level][i].detach() for level in range(multi_levels)]
            bbox_list = [bbox_preds[level][i].detach() for level in range(multi_levels)]
            if with_nms:
                proposal = self._get_bboxes_single(cla_list, bbox_list, multi_level_anchors,
                                                   img_shape=pairs(self.anchor_input_size),
                                                   scale_factor=img_infos[i]['scale_factor'],
                                                   rescale=rescale, with_nms=with_nms)
            result.append(proposal)

        return result

    def _get_bboxes_single(self,
                           cls_score_list,
                           bbox_pred_list,
                           mlvl_anchors,
                           img_shape,
                           scale_factor,
                           rescale=False,
                           with_nms=True):
        """Transform outputs for a single batch item into bbox predictions.

        Args:
            cls_score_list (list[Tensor]): Box scores for a single scale level
                Has shape (num_anchors * num_classes, H, W).
            bbox_pred_list (list[Tensor]): Box energies / deltas for a single
                scale level with shape (num_anchors * 4, H, W).
            mlvl_anchors (list[Tensor]): Box reference for a single scale level
                with shape (num_total_anchors, 4).
            img_shape (tuple[int]): Shape of the input image,
                (height, width, 3).
            scale_factor (ndarray): Scale factor of the image arange as
                (w_scale, h_scale, w_scale, h_scale).
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            Tensor: Labeled boxes in shape (n, 5), where the first 4 columns
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the
                5-th column is a score between 0 and 1.
        """
        #cfg = self.test_cfg if cfg is None else cfg
        assert len(cls_score_list) == len(bbox_pred_list) == len(mlvl_anchors)
        mlvl_bboxes = []
        mlvl_scores = []
        for cls_score, bbox_pred, anchors in zip(cls_score_list,
                                                 bbox_pred_list, mlvl_anchors):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            cls_score = cls_score.permute(1, 2,
                                          0).reshape(-1, self.num_classes+1)
            scores = cls_score.softmax(-1)
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            bboxes = self.bbox.bbox_decode(anchors, bbox_pred, max_shape=img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)
        mlvl_scores = torch.cat(mlvl_scores)

        if with_nms:
            det_bboxes, det_labels = multiclass_nms(mlvl_bboxes, mlvl_scores,
                                                    0.02, dict(type='nms', iou_threshold=0.45),
                                                    200)
            return det_bboxes, det_labels
        else:
            return mlvl_bboxes, mlvl_scores


