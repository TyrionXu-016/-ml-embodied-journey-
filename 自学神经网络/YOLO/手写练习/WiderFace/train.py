import cv2
import math
import random
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from model import TinyYOLO
from model import yolo_loss
from decode import decode

import model  # 前面定义的 8×8×11 网络

B, C = 2, 1
S = 8
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# ---- 极简 Dataset ----
class FaceSet(Dataset):
    def __init__(self, txt):
        with open(txt) as f: self.lines = f.readlines()

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        parts = self.lines[idx].strip().split()
        img = cv2.imread(parts[0])[:, :, ::-1] / 255.
        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)
        label = torch.zeros(S, S, 5)  # 只存“负责”格
        for p in parts[1:]:
            x1, y1, x2, y2, _ = map(float, p.split(','))
            cx, cy, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
            cx, cy, w, h = cx / 256, cy / 256, w / 256, h / 256  # 归一化
            gridx, gridy = int(cx * S), int(cy * S)
            if label[gridy, gridx, 0] == 0:  # 只赋第一个
                w, h = max(w, 1e-4), max(h, 1e-4)  # 防止 0
                label[gridy, gridx, :] = torch.tensor([1,
                                                       cx * S - gridx,
                                                       cy * S - gridy,
                                                       w ** 0.5,  # 直接开方
                                                       h ** 0.5])
        return img, label


def visualize(model, set, epoch):
    model.eval()
    with torch.no_grad():
        img, _ = set[random.randint(0, len(set) - 1)]
        pred = model(img.unsqueeze(0).to(device))  # 8×8×11
        print('pred shape:', pred.shape)  # ← 打印形状
        boxes = decode(pred.cpu())  # 见下方 decode
        vis = (img.permute(1, 2, 0).numpy() * 255).astype('uint8')[:, :, ::-1].copy()
        for b in boxes:
            x1, y1, x2, y2 = map(int, b[:4])
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imwrite(f'epoch{epoch}.jpg', vis)
    model.train()


# ---- 训练循环 ----
def train():
    ds = FaceSet('train.txt')
    dl = DataLoader(ds, batch_size=16, shuffle=True, num_workers=0)
    model = TinyYOLO(B=B, C=C).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(1, 31):
        tot_loss = 0
        for img, gt in tqdm(dl, desc=f'E{epoch}'):
            img, gt = img.to(device), gt.to(device)
            pred = model(img)  # N×8×8×11
            loss = yolo_loss(pred, gt, S=S, B=B, C=C)
            opt.zero_grad();
            loss.backward();
            opt.step()
            tot_loss += loss.item()
        print(f'epoch {epoch}  loss={tot_loss / len(dl):.4f}')
        if epoch % 5 == 0: visualize(model, ds, epoch)
        torch.save(model.state_dict(), 'tinyyolo_last.pth')


if __name__ == '__main__':
    train()
