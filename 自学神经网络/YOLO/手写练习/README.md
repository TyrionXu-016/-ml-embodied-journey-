安装环境

pip install torch torchvision matplotlib opencv-python tqdm

训练数据集：
http://shuoyang1213.me/WIDERFACE/index.html

# 下载
wget -O wider_face_split.zip \
  https://github.com/NyanSwanAung/Object-Detection-Using-DETR-CustomDataset/releases/download/v1.0/wider_face_split.zip

# 校验
unzip -t wider_face_split.zip && echo "OK"