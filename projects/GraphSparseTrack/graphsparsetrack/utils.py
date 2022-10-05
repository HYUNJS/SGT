import torch
import torch.nn.functional as F


def half_dict(data):
    data_t1, data_t2 = dict(), dict()
    for k in data.keys():
        data_t1[k] = data[k][::2]
        data_t2[k] = data[k][1::2]
    return data_t1, data_t2

def masking_dict(data, mask_t1, mask_t2):
    data_t1, data_t2 = dict(), dict()
    for k in data.keys():
        data_t1[k] = data[k][mask_t1]
        data_t2[k] = data[k][mask_t2]
    return data_t1, data_t2

def update_dict(src, dst):
    for k in src.keys():
        dst[k] = src[k]

def cxcywh2box(xs, ys, whs):
    return torch.cat([xs - whs[..., 0:1], ys - whs[..., 1:2], xs + whs[..., 2:3], ys + whs[..., 3:4]], dim=-1)

def ind2xy(inds, regs, w=272):
    if inds.dim() == 2:
        inds = inds.unsqueeze(2)
    xs, ys = inds % w, inds // w
    tgt_xys = torch.cat([xs, ys], dim=2) + regs # [B, 500, 2]
    return tgt_xys

def flatten_boxes_dict(boxes):
    if type(boxes).__name__ == 'dict':
        flatten_boxes = {}
        for k in boxes.keys():
            flatten_boxes[k] = boxes[k].flatten(0, 1)
    else:
        flatten_boxes = boxes.flatten(0, 1)
    return flatten_boxes

def mask_boxes_dict(boxes, mask):
    if type(boxes).__name__ == 'dict':
        masked_boxes = {}
        for k in boxes.keys():
            masked_boxes[k] = boxes[k][mask]
    else:
        masked_boxes = boxes[mask]
    return masked_boxes

def xyah2tlbr(xyah):
    xywh = xyah.clone()
    xywh[..., 2] *= xywh[..., 3]
    return torch.cat([ xywh[..., 0] - xywh[..., 2] / 2,
                        xywh[..., 1] - xywh[..., 3] / 2,
                        xywh[..., 0] - xywh[..., 2] / 2,
                        xywh[..., 1] - xywh[..., 3] / 2], dim=-1)

def xywhwh2tlbr(xywhwh):
    return torch.stack([ xywhwh[:, 0] - xywhwh[:, 2],
                         xywhwh[:, 1] - xywhwh[:, 3],
                         xywhwh[:, 0] + xywhwh[:, 4],
                         xywhwh[:, 1] + xywhwh[:, 5]], dim=1)

def tlbr2cxcywh(bboxes):
    ws = bboxes[:, 2] - bboxes[:, 0]
    hs = bboxes[:, 3] - bboxes[:, 1]
    xs = (bboxes[:, 2] + bboxes[:, 0]) / 2
    ys = (bboxes[:, 3] + bboxes[:, 1]) / 2
    return torch.stack([xs, ys, ws, hs], dim=1)

def xyxy2cxcywh(bboxes):
    ws = bboxes[:, 2] - bboxes[:, 0]
    hs = bboxes[:, 3] - bboxes[:, 1]
    xs = (bboxes[:, 2] + bboxes[:, 0]) / 2
    ys = (bboxes[:, 3] + bboxes[:, 1]) / 2
    return torch.stack([xs, ys, ws, hs], dim=1)

def gather_feature(fmap, index, mask=None, use_transform=False):
    if use_transform:
        # change a (N, C, H, W) tenor to (N, HxW, C) shape
        batch, channel = fmap.shape[:2]
        fmap = fmap.view(batch, channel, -1).permute((0, 2, 1)).contiguous()

    dim = fmap.size(-1)
    index = index.unsqueeze(len(index.shape)).expand(*index.shape, dim)
    fmap = fmap.gather(dim=1, index=index)
    if mask is not None:
        # this part is not called in Res18 dcn COCO
        mask = mask.unsqueeze(2).expand_as(fmap)
        fmap = fmap[mask]
        fmap = fmap.reshape(-1, dim)
    return fmap

def pseudo_nms(fmap, pool_size=3):
    r"""
    apply max pooling to get the same effect of nms

    Args:
        fmap(Tensor): output tensor of previous step
        pool_size(int): size of max-pooling
    """
    pad = (pool_size - 1) // 2
    fmap_max = F.max_pool2d(fmap, pool_size, stride=1, padding=pad)
    keep = (fmap_max == fmap).float()
    return fmap * keep

def topk_score(scores, K=40):
    """
    get top K point in score map
    """
    batch, channel, height, width = scores.shape

    # get topk score and its index in every H x W(channel dim) feature map
    topk_scores, topk_inds = torch.topk(scores.reshape(batch, channel, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()

    # get all topk in in a batch
    topk_score, index = torch.topk(topk_scores.reshape(batch, -1), K)
    # div by K because index is grouped by K(C x K shape)
    topk_clses = (index / K).int()
    topk_inds = gather_feature(topk_inds.view(batch, -1, 1), index).reshape(batch, K)
    topk_ys = gather_feature(topk_ys.reshape(batch, -1, 1), index).reshape(batch, K)
    topk_xs = gather_feature(topk_xs.reshape(batch, -1, 1), index).reshape(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs
