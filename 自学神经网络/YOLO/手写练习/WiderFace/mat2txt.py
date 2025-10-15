import scipy.io
import os
import cv2
import pathlib
from tqdm import tqdm
import numpy as np

# 读取官方mat
mat = scipy.io.loadmat('wider_face_split/wider_face_train.mat')
file_list = mat['file_list']
event_list = mat['event_list']
face_bbox_list = mat['face_bbx_list']

out_dir = pathlib.Path('wider256')
out_dir.mkdir(exist_ok=True)
f_out = open('train.txt', 'w')

target_size = 256

for idx in tqdm(range(file_list.shape[1])):
    event = event_list[0, idx][0]
    if event != '0--Parade':
        continue
    subdir = out_dir/event
    subdir.mkdir(exist_ok=True)

    filenames = file_list[0, idx]
    bboxes = face_bbox_list[0, idx]
    for j in range(filenames.shape[0]):
        fname = filenames[j, 0][0] + '.jpg'
        img_path = f'wider_train/images/{event}/{fname}'
        if not os.path.exists(img_path):
            continue
        img = cv2.imread(img_path)
        H, W = img.shape[:2]
        # 灰度缩放
        scale = min(target_size/W, target_size/H)
        newW, newH = int(W * scale), int(H * scale)
        img_res = cv2.resize(img, (newW, newH))
        canvas = 114 * np.ones((target_size, target_size, 3), dtype=np.uint8)
        dw, dh = (target_size - newW)//2, (target_size - newH)//2
        cv2.imwrite(str(subdir/fname), canvas)

        # 处理bbox
        box_mat = bboxes[j, 0]
        line = f'{subdir/fname}'
        for k in range(box_mat.shape[0]):
            x1, y1, x2, y2 = box_mat[k, :4]
            # 同步缩放+平移
            x1 = x1 * scale + dw
            y1 = y1 * scale + dh
            x2 = x2 * scale + dw
            y2 = y2 * scale + dh
            line += f' {int(x1)},{int(y1)},{int(x2)},{int(y2)},0'

        f_out.write(line + '\n')

f_out.close()
print('done -> train.txt 共', sum(1 for _ in open('train.txt')), '张图')
