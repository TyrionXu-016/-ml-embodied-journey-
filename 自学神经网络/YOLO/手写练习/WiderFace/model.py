import torch.nn as nn
import torch


class TinyYOLO(nn.Module):
    def __init__(self, B=2, C=1):
        super().__init__()
        self.B, self.C = B, C
        # 5层卷积 + 2层FC
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 16, 5, 2, 2), nn.BatchNorm2d(16), nn.LeakyReLU(0, 1),
            nn.Conv2d(16, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.LeakyReLU(0, 1),
            nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.LeakyReLU(0, 1),
            nn.Conv2d(64, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0, 1),
            nn.Conv2d(128, 256, 3, 2, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0, 1),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 512), nn.LeakyReLU(0.1),
            nn.Linear(512, 8 * 8 * B * (5 + C))  # 768 节点
        )

    def forward(self, x):
        feat = self.backbone(x)
        return self.head(feat).view(-1, 8, 8, self.B, 5 + self.C)


def yolo_loss(pred, target, S=8, B=2, C=1, l_coord=5.0, l_noobj=0.5, eps=1e-6):
    """
    pred   : N×S×S×(B*5+C)  网络输出 → 内部 reshape 成 N×S×S×B×(5+C)
    target : N×S×S×5        仅“负责”网格有值，其余 0
    return : 标量 loss，已做 NaN 保护
    """
    N = pred.size(0)
    device = pred.device

    # --- 1. 负责 / 不负责掩码 ---
    mask_obj = (target[..., 0] == 1)        # N×S×S
    mask_noobj = ~mask_obj
    n_obj = mask_obj.sum().clamp(min=1)

    # --- 2. 拆成 N×S×S×B×(5+C) ---
    pred = pred.view(N, S, S, B, 5 + C)     # 现在最后一维=5+C=6

    # --- 3. 取两个框 ---
    box1 = pred[..., 0, :4]                 # 框1  x,y,w,h  N×S×S×4
    box2 = pred[..., 1, :4]                 # 框2  x,y,w,h
    conf1 = pred[..., 0, 4]                 # 框1 置信度
    conf2 = pred[..., 1, 4]                 # 框2 置信度
    cls_logits = pred[..., 0, 5:]           # 类别 logits（两框共享即可）

    # --- 4. 选负责框（IOU 更大者）---
    with torch.no_grad():
        # 计算与 gt 的 iou
        targ_xy = target[..., 1:3]  # N×S×S×2
        targ_wh = target[..., 3:5]  # N×S×S×2
        iou1 = _iou(box1, targ_xy, targ_wh)  # 不再 unsqueeze/squeeze
        iou2 = _iou(box2, targ_xy, targ_wh)
        best_box = (iou1 > iou2).long()           # 0 选框1，1 选框2
    # 负责框坐标 / 置信度
    resp_box = torch.where(best_box.unsqueeze(-1) == 0, box1, box2)
    resp_conf = torch.where(best_box == 0, conf1, conf2)

    # --- 5. 各项 loss ---
    # 坐标
    loss_xy = nn.MSELoss(reduction='sum')(
        resp_box[mask_obj][..., 0:2], target[mask_obj][..., 1:3])
    loss_wh = nn.MSELoss(reduction='sum')(
        resp_box[mask_obj][..., 2:4].clamp(min=eps).sqrt(),
        target[mask_obj][..., 3:5].clamp(min=eps).sqrt())
    # 5. 置信度 loss
    targ_conf = target[..., 0].float()
    loss_conf_obj = nn.MSELoss(reduction='sum')(
        resp_conf[mask_obj], targ_conf[mask_obj])

    conf_stack = torch.stack([conf1, conf2], dim=-1)  # N×S×S×2
    loss_conf_noobj = nn.MSELoss(reduction='sum')(
        conf_stack[mask_noobj],
        torch.zeros_like(conf_stack)[mask_noobj])
    # 类别
    loss_cls = nn.BCEWithLogitsLoss(reduction='sum')(
        cls_logits[mask_obj].squeeze(-1), targ_conf[mask_obj].clamp(0, 1))

    # --- 6. 合并 ---
    loss = (l_coord * (loss_xy + loss_wh) +
            loss_conf_obj +
            l_noobj * loss_conf_noobj +
            loss_cls) / N
    return loss


@torch.no_grad()
def _iou(box, xy, wh):
    """box: N×S×S×4 (x,y,w,h)  xy/wh: N×S×S×2"""
    x1, y1, w1, h1 = box[..., 0], box[..., 1], box[..., 2], box[..., 3]
    x2, y2, w2, h2 = xy[..., 0], xy[..., 1], wh[..., 0], wh[..., 1]

    xA = torch.maximum(x1 - w1/2, x2 - w2/2)
    yA = torch.maximum(y1 - h1/2, y2 - h2/2)
    xB = torch.minimum(x1 + w1/2, x2 + w2/2)
    yB = torch.minimum(y1 + h1/2, y2 + h2/2)

    inter = (xB - xA).clamp(min=0) * (yB - yA).clamp(min=0)
    union = w1*h1 + w2*h2 - inter + 1e-8
    return inter / union
