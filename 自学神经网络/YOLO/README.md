#### <a href="https://www.kimi.com/chat/d3mgdrmp4uo0a7qq1580">教程地址</a>

## 一、环境准备

```
# 安装依赖
pip install ultralytics opencv-python torch torchvision tqdm
```

## 二、准备人脸检测模型
| 镜像          | 示例（以 ultralytics 为例）                              |
| ----------- | ------------------------------------------------- |
| kkgithub    | `https://kkgithub.com/ultralytics/ultralytics`    |
| bgithub     | `https://bgithub.xyz/ultralytics/ultralytics`     |
| hub.fastgit | `https://hub.fastgit.org/ultralytics/ultralytics` |



```bash
# 已有 Socks5 本地端口 1080（V2Ray/Clash 默认）
git config --global http.proxy 'socks5://127.0.0.1:7890'
git config --global https.proxy 'socks5://127.0.0.1:7890'

# 用完取消
git config --global --unset http.proxy
git config --global --unset https.proxy
# 预训练模型
git clone https://github.com/U202142209/HumanFaceDetection.git
git clone https://github.com/AND-Q/Facial-Expression-Recognition.git
```

