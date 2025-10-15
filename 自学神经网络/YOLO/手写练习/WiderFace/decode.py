import torch
import math


def decode(pred, conf_thr=0.3, iou_thr=0.4, S=8, B=2, C=1):
    """
    pred: N×8×8×2×6  (x,y,w,h,conf,class)
    return: K×4  原图 256×256 坐标
    """
    N, S, _, _ = pred.shape[:4]
    device = pred.device
    boxes, scores = [], []
    cell = 256 // S
    pred = pred.view(N, S, S, B, 5 + C)   # 确保 5 维

    for i in range(S):
        for j in range(S):
            for b in range(B):
                conf = pred[0, i, j, b, 4]          # 置信度
                if conf > conf_thr:
                    x = (j + pred[0, i, j, b, 0]) * cell
                    y = (i + pred[0, i, j, b, 1]) * cell
                    w = pred[0, i, j, b, 2] ** 2 * 256
                    h = pred[0, i, j, b, 3] ** 2 * 256
                    x1, y1 = x - w/2, y - h/2
                    x2, y2 = x + w/2, y + h/2
                    boxes.append([x1, y1, x2, y2])
                    scores.append(conf.item())
    if len(boxes) == 0:
        return torch.empty((0, 4))
    boxes = torch.tensor(boxes, dtype=torch.float32)
    scores = torch.tensor(scores)
    keep = nms(boxes, scores, iou_thr)
    return boxes[keep].round().int().numpy()


# 简易 CPU-NMS
def nms(boxes, scores, iou_thr):
    order = scores.argsort(descending=True)
    keep = []
    while order.numel() > 0:
        i = order[0];
        keep.append(i)
        if order.numel() == 1: break
        iou = box_iou(boxes[i].unsqueeze(0), boxes[order[1:]])
        order = order[1:][iou.squeeze() <= iou_thr]
    return torch.tensor(keep)


def box_iou(a, b):
    # a: 1×4  b: N×4
    x1 = torch.maximum(a[:, 0], b[:, 0])
    y1 = torch.maximum(a[:, 1], b[:, 1])
    x2 = torch.minimum(a[:, 2], b[:, 2])
    y2 = torch.minimum(a[:, 3], b[:, 3])
    inter = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a + area_b - inter + 1e-6)
