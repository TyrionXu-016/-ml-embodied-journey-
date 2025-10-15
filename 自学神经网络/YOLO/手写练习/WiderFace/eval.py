import torch, cv2, random, pathlib
from model import TinyYOLO
from decode import decode
from torch.utils.data import Dataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'
B, C = 2, 1


class ValSet(Dataset):
    def __init__(self, txt): self.lines = open(txt).read().splitlines()

    def __len__(self): return len(self.lines)

    def __getitem__(self, idx):
        parts = self.lines[idx].split()
        img = cv2.imread(parts[0])[:, :, ::-1] / 255.
        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)
        boxes = []
        for p in parts[1:]:
            x1, y1, x2, y2, _ = map(float, p.split(','))
            boxes.append([x1, y1, x2, y2])
        return img, torch.tensor(boxes)


def compute_iou(pred, gt):
    x1 = torch.maximum(pred[:, 0], gt[:, 0])
    y1 = torch.maximum(pred[:, 1], gt[:, 1])
    x2 = torch.minimum(pred[:, 2], gt[:, 2])
    y2 = torch.minimum(pred[:, 3], gt[:, 3])
    inter = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    union = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1]) + \
            (gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1]) - inter + 1e-8
    return inter / union


@torch.no_grad()
def eval_one(model, set, idx):
    img, gt = set[idx]
    pred = model(img.unsqueeze(0).to(device))  # 1×8×8×2×6
    boxes = decode(pred.cpu())  # K×4
    boxes = torch.tensor(boxes, dtype=torch.float32)  # ← 加这句
    if len(boxes) == 0:  return 0.0, 0.0
    ious = [compute_iou(boxes, g.unsqueeze(0)).max().item() for g in gt]
    return sum(ious) / len(ious), len(boxes)


def main():
    model = TinyYOLO(B=B, C=C).to(device)
    model.load_state_dict(torch.load('tinyyolo_last.pth', map_location=device, weights_only=True))
    model.eval()
    val = ValSet('val.txt')
    iou_list = []
    # 之前记录每张图的框数
    num_boxes = []

    for k in range(min(50, len(val))):  # 抽 50 张
        iou, n = eval_one(model, val, k)
        iou_list.append(iou)
        num_boxes.append(n)  # ← 收集
        if k < 5:  # 画 5 张图
            img, gt = val[k]
            vis = (img.permute(1, 2, 0).numpy() * 255).astype('uint8')[:, :, ::-1].copy()
            pred = model(img.unsqueeze(0).to(device))
            for b in decode(pred.cpu()):
                x1, y1, x2, y2 = map(int, b)
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            for g in gt:
                x1, y1, x2, y2 = map(int, g)
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 1)
            cv2.imwrite(f'val_{k}.jpg', vis)
    m_iou = sum(iou_list) / len(iou_list)
    print(f'>> 验证集平均IoU: {m_iou:.3f}  (目标>0.5)')
    print(f'>> 每图平均框数: {sum(num_boxes) / len(num_boxes):.1f}')


if __name__ == '__main__':
    main()
