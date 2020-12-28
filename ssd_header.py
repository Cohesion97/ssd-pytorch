import torch
import torch.nn.functional as F
from losses.smooth_l1_loss import smooth_l1_loss
from init_weights import *
from IoU_assign import IoU_assigner
from functools import partial
from six.moves import map, zip

def pairs(x):
    assert type(x) != str
    return [x,x]

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

class ssd_header(nn.Module):

    default_inplanes = [512, 1024, 512,256,256,256]

    def __init__(self,
                 num_classes,
                 anchor_cfg={'strides':[8,16,32,64,100,300],
                             'ratios':[[2],[2,3],[2,3],[2,3],[2,],[2]],
                             'scale_range':[0.15, 0.9],
                             'input_size':300},
                 neg_pos_rate=3
                 ):
        super(ssd_header,self).__init__()
        self.num_classes = num_classes
        self.anchor_cfg = anchor_cfg
        assert len(self.default_inplanes)==len(self.anchor_cfg['strides'])
        self.num_multi_feats = len(self.default_inplanes)
        self.neg_pos_rate = neg_pos_rate

        self.anchor_strides = self.anchor_cfg['strides']
        self.anchor_ratios = self.anchor_cfg['ratios']
        self.anchor_input_size = self.anchor_cfg['input_size']

        self.base_len = [int(0.07 * self.anchor_input_size)]
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
                gen_ratio.extend([k, 1/k])
            gen_ratios.append(
                torch.Tensor(gen_ratio)
            )
        self.gen_ratios = gen_ratios
        self.gen_scales = gen_scales
        self.base_anchors = self.gen_anchor_single_img()
        self.num_anchors = [len(num_anchor) for num_anchor in self.base_anchors ]
        #from IPython import embed;embed()

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

        #from IPython import embed;embed()


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
        index = list(range(len(ratios)+1))
        return torch.index_select(anchors,0,torch.LongTensor(index))

    def gen_anchor_single_img(self,device='cuda'):
        mulit_feat = []
        for i in range(len(self.base_len)):
            mulit_feat.append(self.gen_anchor_single_feat(pairs(self.base_len[i]),
                                                          self.anchor_strides[i],
                                                          self.gen_ratios[i],
                                                          self.gen_scales[i]))
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
            shift_ = self.shift(x,y)
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
        x_ = x.repeat(len(y)).view(-1)
        y_ = y.view(-1,1).repeat(1,len(x)).view(-1)
        shift = [x_, y_, x_, y_]
        shift = torch.stack(shift, dim=1)

        return shift

    def match_(self, anchors_list, gt_bboxes, gt_labels, batch_size):
        """
        match anchors with gt_bboxes in a batch of imgs
        :param anchors_list: list[list[Tensor: num_levels, featmap_size, 4]]
        :param gt_bboxes: list[Tensor]
        :param gt_labels: list[Tensor]
        :return:
        """
        num_anchors_level = [anchors.size(0) for anchors in anchors_list[0]]
        concat_anchor_list = []

        for i in range(batch_size):
            concat_anchor_list.append(torch.cat(anchors_list[i])) # list[Tensor: num_total_anchors, 4]

        result = multi_apply(self.match_single_img, concat_anchor_list, gt_bboxes, gt_labels, batch_size=batch_size)
        (all_labels, all_bbox_targets, all_pos_idx, all_neg_idx) = result

        num_pos_total = sum([inds.numel() for inds in all_pos_idx])
        num_neg_total = sum([inds.numel() for inds in all_neg_idx])

        labels_list = image_to_level(all_labels, num_anchors_level)
        bboxes_target_list = image_to_level(all_bbox_targets, num_anchors_level)

        return (labels_list, bboxes_target_list, num_pos_total, num_neg_total)


    def match_single_img(self, anchors, gt_bboxes, gt_labels, batch_size=4):
        """
        match anchors with gt_bboxes in one img
        :param anchors: Tensor: num_anchors, 4
        :param gt_bboxes: Tensor: num_gt, 4
        :param gt_labels: Tensor: num_gt,
        :param batch_size: int
        :return: labels: Tensor: num_anchors, ; num_classes(background) when not assigned with gt
                bbox_targets: Tensor: num_anchors, 4; 0 when not assigned
                pos_idx: list:
        """
        assigner = IoU_assigner()
        # [num_anchors, ],
        assign_gt_idx, assign_label = assigner.assign(gt_bboxes, anchors, gt_labels)
        # len(pos_idx)+len(neg_idx)=num_anchors
        pos_idx = torch.nonzero(assign_gt_idx>0, as_tuple=False).squeeze(-1)
        neg_idx = torch.nonzero(assign_gt_idx==0, as_tuple=False).squeeze(-1)
        pos_bboxs = anchors[pos_idx]
        neg_bboxs = anchors[neg_idx]
        pos_gt = gt_bboxes[assign_gt_idx[pos_idx]-1]

        bbox_targets = torch.zeros_like(anchors)
        labels = anchors.new_full(
            (anchors.shape[0], ), self.num_classes , dtype=torch.long) #(num_anchors, )

        if len(pos_idx) > 0:
            pos_bbox_targets = self.bbox_encoder(pos_bboxs,pos_gt)
            bbox_targets[pos_idx, :] = pos_bbox_targets
            if gt_labels is not None:
                labels[pos_idx] = gt_labels[assign_gt_idx[pos_idx]-1]
            else:
                labels[pos_idx] = 0
        return (labels, bbox_targets, pos_idx, neg_idx)

    def bbox_encoder(self, anchors, gt_bboxes, mean=(0., 0., 0., 0.), std=(1., 1., 1., 1.)):
        """

        :param anchors: Tensor: num_anchors, 4
        :param gt_bboxes: Tensor: num_anchors, 4
        :param mean:
        :param std:
        :return:
        """
        anchors_centerx = 0.5*(anchors[...,0]+anchors[...,2])
        anchors_centery = 0.5*(anchors[...,1]+anchors[...,3])
        anchors_w = anchors[...,2]-anchors[...,0]
        anchors_h = anchors[...,3]-anchors[...,1]

        gt_centerx = 0.5*(gt_bboxes[...,0]+gt_bboxes[...,2])
        gt_centery = 0.5*(gt_bboxes[...,1]+gt_bboxes[...,3])
        gt_w = gt_bboxes[...,2]-gt_bboxes[...,0]
        gt_h = gt_bboxes[...,3]-gt_bboxes[...,1]

        dx = (gt_centerx - anchors_centerx) / anchors_centerx
        dy = (gt_centery - anchors_centery) / anchors_centery
        dw = torch.log(gt_w / anchors_w)
        dh = torch.log(gt_h / anchors_h)

        d = torch.stack([dx, dy, dw, dh], dim=-1)  # [num_anchors, 4]

        means = d.new_tensor(mean).unsqueeze(0)
        stds = d.new_tensor(std).unsqueeze(0)
        d = d.sub_(means).div_(stds)
        return d # [num_anchors, 4]

    def loss(self, cla_scores, loc_results, gt_bboxes, gt_labels,):
        """
        :param cla_scores: list[Tensor] (B, num_anchor*num_classes+1, h, w)
        :param loc_results: list[Tensor] (B, num_anchor*4, h, w)
        :param gt_bboxes: list[Tensor]
        :param gt_labels: list[Tensor]
        :return:
        """
        multi_level_anchors = self.grid_anchors() # list[Tensor: num_levels, featmap_size, 4]
        batch_size =  cla_scores[0].shape[0]
        anchors_list =[multi_level_anchors for _ in range(batch_size)]

        (labels_list, bboxes_target_list, num_pos_total, num_neg_total) = self.match_(
            anchors_list, gt_bboxes, gt_labels, batch_size)
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

        all_anchors = []
        for i in range(batch_size):
            all_anchors.append(torch.cat(anchors_list[i]))
        all_labels = torch.cat(labels_list, -1).view(batch_size, -1)

        # check NaN and Inf
        assert torch.isfinite(all_cla_scores).all().item(), \
            'classification scores become infinite or NaN!'
        assert torch.isfinite(all_bbox_preds).all().item(), \
            'bbox predications become infinite or NaN!'

        loss_cla, loss_bbox = multi_apply(self.loss_single_img, all_cla_scores,  all_bbox_preds,all_anchors, all_labels,
                                           all_bbox_targets, num_total_samples=num_pos_total)
        return loss_cla, loss_bbox

    def loss_single_img(self, cla_scores,  bbox_preds, anchors, labels,
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
        loss_cla_all = F.cross_entropy(cla_scores, labels, reduction='none')
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

        loss_bbox = smooth_l1_loss(bbox_preds, bbox_targets, avg_factor=num_total_samples)
        return loss_cla, loss_bbox






