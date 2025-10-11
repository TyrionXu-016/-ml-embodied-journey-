# ML-具身智能学习路线图

> 后端 → 机器学习 → 深度学习 → 强化学习 → 机器人仿真 → 具身智能实战  
> 每阶段完成后把 `- [ ]` 改成 `- [x]` 即可自动渲染勾选框。

---

## 阶段 0：Python & 数学速通（2 周）

| 任务                 | 输出物                       | 状态    | 资源                                                                                   |
|--------------------|---------------------------|-------|--------------------------------------------------------------------------------------|
| Python 100 题       | `python-100/` 文件夹 + 代码    | - [ ] | <a href="https://leetcode.cn/problemset/all">LeetCode-Python</a>                     |
| Numpy 100 题        | `numpy-100.ipynb`         | - [ ] | <a href="https://github.com/rougier/numpy-100">GitHub rougier/numpy-100</a>          |
| 3Blue1Brown 线性代数笔记 | `notes-linear-algebra.md` | - [ ] | <a href="https://www.bilibili.com/video/B1Ws411c7d4">Bilibili 3Blue1Brown 官方中字</a>   |
| 概率论小测 ≥80 分        | 测验截图                      | - [ ] | <a href="https://zh.khanacademy.org/math/statistics-probability">Khan Academy 概率</a> |

---

## 阶段 1：机器学习基石（6 周）

| 任务 | 输出物 | 状态 |
|---|---|---|
| 泰坦尼克 EDA + 基线 | `01-titanic-eda.ipynb` | - [ ] |
| 特征工程 & 交叉验证 | Kaggle 分数 ≥0.79 | - [ ] |
| 模型族谱对比 | `02-model-zoo.ipynb` + 曲线图 | - [ ] |
| 自动化调优 Pipeline | `titan_automl.py` | - [ ] |
| 无监督 & 降维 | `03-clustering-pca.ipynb` | - [ ] |
| 房价预测结课 | `04-house-prices/` 仓库 | - [ ] |

---

## 阶段 2：深度学习（8 周）

| 任务 | 输出物 | 状态 |
|---|---|---|
| PyTorch 张量 & Autograd | `pytorch-01-tensor.ipynb` | - [ ] |
| 全连接 Fashion-MNIST | Acc ≥92 % + TensorBoard 日志 | - [ ] |
| CNN CIFAR-10 | Acc ≥90 % + 训练曲线 | - [ ] |
| 迁移学习猫狗二分类 | `transfer_learning.py` | - [ ] |
| LSTM 生成宋词 | `lstm-poem.ipynb` | - [ ] |
| 迷你 Transformer | BLEU ≥20 英法翻译 | - [ ] |
| CLIP 图文检索 | `clip-flickr8k-demo.ipynb` | - [ ] |
| 阶段总结 | `deep-learning-portfolio/` 仓库 | - [ ] |

---

## 阶段 3：强化学习 & 模仿学习（8 周）

| 任务 | 输出物 | 状态 |
|---|---|---|
| Q-Learning 格子世界 | `q-learning-grid.ipynb` | - [ ] |
| DQN Breakout | 得分 ≥100 + 模型文件 | - [ ] |
| REINFORCE CartPole | 得分 500 完整训练曲线 | - [ ] |
| PPO HalfCheetah | 均值 ≥3000（W&B 日志） | - [ ] |
| 行为克隆 CarRacing | 得分 ≥800 | - [ ] |
| 自定义 Gym 环境 | `gym-foo/` 独立包 | - [ ] |
| 阶段总结 | `rl-portfolio/` 仓库 | - [ ] |

---

## 阶段 4：机器人仿真 & 中间件（6 周）

| 任务 | 输出物 | 状态 |
|---|---|---|
| ROS2 话题小 demo | `ros2_beginner/` 包 | - [ ] |
| 二连杆 URDF + RViz2 | `two-link/urdf/` + 截图 | - [ ] |
| MoveIt2 运动规划 | 避障成功 GIF | - [ ] |
| Isaac Sim Python API | 采集 RGB-D 脚本 | - [ ] |
| MuJoCo 3 指夹爪抓取 | SAC 训练日志 | - [ ] |
| gRPC 图像服务 | `robot_grpc/` 仓库 | - [ ] |

---

## 阶段 5：具身智能综合实战（12 周）

| 任务 | 输出物 | 状态 |
|---|---|---|
| 文献综述 | `survey-embodied-LLM.md` ≥20 篇 | - [ ] |
| 系统架构设计 | `system_design.md` + 时序图 | - [ ] |
| 多模态感知 | CLIP+PointNet 分类 90 %+ | - [ ] |
| LLM Planner | 子任务 JSON 输出示例 | - [ ] |
| 技能库封装 | 8 个 MoveIt Action | - [ ] |
| 端到端 Demo | 桌面整理视频 | - [ ] |
| 评估 & 消融 | 报告 + W&B 曲线 | - [ ] |
| 技术报告 & 分享 | `Embodied-LLM-Robot/` 仓库 + 视频 | - [ ] |

---

## 最终交付清单

- [ ] Kaggle 两竞赛银牌截图  
- [ ] `ml-foundations` 仓库  
- [ ] `deep-learning-portfolio` 仓库  
- [ ] `rl-portfolio` 仓库  
- [ ] `robotics-toolbox` 组织  
- [ ] `Embodied-LLM-Robot` 主力仓库  
- [ ] B 站 / YouTube 15 min 分享视频  

---

## 精选资源速查
| 类型     | 名称                                                                                                                                |
|--------|-----------------------------------------------------------------------------------------------------------------------------------|
| 数学     | <a href="https://www.bilibili.com/video/B1Ws411c7d4">3Blue1Brown 线性代数</a>                                                         |
| ML 入门  | <a href="https://www.coursera.org/learn/machine-learning">吴恩达 Coursera</a>                                                        |
| 书籍     | <a href="https://github.com/ageron/handson-ml3">《Hands-On ML》第二版 </a>                                                             |
| 深度学习   | <a href="https://pytorch.org/tutorials/">PyTorch 官方 Tutorial</a>                                                                  |
| RL     | <a href="https://stable-baselines3.readthedocs.io/">Stable-Baselines3 文档</a>                                                      |
| 机器人    | <a href="https://docs.ros.org/en/rolling/">ROS2 官方文档</a>                                                                          |
| 具身智能综述 | <a href="http://mp.weixin.qq.com/s?__biz=MzI1MzUyMTMwOA==&mid=2247496753&idx=1&sn=8bde6d097cba2018289956ec7dd04cd4">万字综述·魔方AI</a> |
| 论文库    | <a href="https://github.com/embodied-ai/awesome-embodied-ai">Embodied AI ArXiv 合集</a>                                             |


---

## 使用方式

1. 每完成一行任务，把 `- [ ]` 改成 `- [x]` 并 commit & push。  
2. 建议在对应阶段新建子文件夹，存放代码、Notebook、模型、报告。  
3. 养成习惯：commit 信息格式 `phase1: titanic baseline 0.803`。
