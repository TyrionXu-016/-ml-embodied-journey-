# 安装环境
#### Anaconda
    mac ✅
    windows ❌

| 工具         | 一键安装命令                                                                                        | 说明                                                     |
|------------|-----------------------------------------------------------------------------------------------|--------------------------------------------------------|
| Python≥3.8 | 官网/Anaconda                                                                                   | 建议直接装 Anaconda                                         |
| PyTorch    | `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118` | 自带 GPU 加速                                              |
| Jupyter    | `pip install jupyterlab`                                                                      | 交互式改代码最直观                                              |
| 数据集        | `pip install -U torchvision`                                                                  | CIFAR-10/MNIST 内置                                      |

#### 跑项目
| #   | 项目                  | 数据集                      | 难度  | 你将学到                          | 官方/优质教程                                                                                            |
|-----|---------------------|--------------------------|-----|-------------------------------|----------------------------------------------------------------------------------------------------|
| 0   | MNIST 手写数字识别        | MNIST                    | ★☆☆ | 最简 CNN 模板、训练/验证/测试全流程         | [PyTorch 官方 60 min 入门](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)         |
| 1   | CIFAR-10 彩色图片 10 分类 | CIFAR-10                 | ★★☆ | 过拟合、Data Augmentation、LR 调度   | [CS231n 作业1](http://cs231n.github.io/assignments2023/assignment1/)                                 |
| 2   | 猫狗大战（二分类）           | Kaggle Dogs vs Cats      | ★★☆ | 迁移学习、预训练 ResNet、图片尺寸变换        | [Kaggle 官方 Notebook](https://www.kaggle.com/code/alexattia/getting-started-with-cats-dogs-dataset) |
| 3   | 交通标志识别              | GTSRB                    | ★★★ | 实际尺寸不一致、数据不平衡、可视化 feature map | [GitHub: trafficsign-CNN](https://github.com/ypwhs/trafficsign_classification)                     |
| 4   | 人脸口罩检测              | 自建 2 类或 Kaggle Face Mask | ★★★ | 数据集扩增、MobileNet 轻量化、ONNX 推理   | [GitHub: face-mask-detection](https://github.com/chandrikadeb7/Face-Mask-Detection)                |