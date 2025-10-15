import random
import shutil

random.seed(42)
ratio = 0.8  # 80 % 训练，20 % 验证
src = 'train.txt'
dst_train = 'train_new.txt'
dst_val = 'val.txt'

with open(src) as f:
    lines = f.readlines()

random.shuffle(lines)
n_train = int(len(lines) * ratio)

with open(dst_train, 'w') as f:
    f.writelines(lines[:n_train])

with open(dst_val, 'w') as f:
    f.writelines(lines[n_train:])

# 可选：覆盖原文件
shutil.move(dst_train, src)

print(f'>> 原数据 {len(lines)} 张')
print(f'>> 训练集 {n_train} 张 → train.txt')
print(f'>> 验证集 {len(lines) - n_train} 张 → val.txt')
