## 总述
请简练地概括项目的主要贡献，使读者可以快速理解并复现你的工作，包括：
- 原始模型的名称及链接
- 优化效果（精度和加速比），简单给出关键的数字即可，在这里不必详细展开
- 在Docker里面代码编译、运行步骤的完整说明
  - 请做到只要逐行运行你给的命令，就能把代码跑起来，比如从docker pull开始

## 原始模型
### 模型简介
Anchor DETR是由旷视科技孙剑团队于2021年9月提出并于最近开源的Transformer目标检测器。

![网络架构](img/architecture.png)

其特色是采用anchor point（锚点）查询设计，因此每个查询只预测锚点附近的目标，因此更容易优化。

并且为了降低计算的复杂度，作者团队还提出了一种注意力变体，称为行列解耦注意力（Row-Column Decoupled Attention）。
行列解耦注意力可以有效降低计算成本，同时实现与DETR中的标准注意力相似甚至更好的性能。

![计算比较](img/p0.png)

最终实验显示与DETR相比，Anchor DETR可以在减少10倍训练epoch数的情况下，获得更好的性能。

![对比实现](img/p1.png)

论文地址：<https://arxiv.org/abs/2109.07107>

开源代码地址：<https://github.com/megvii-research/AnchorDETR>

### 模型优化的难点

## 优化过程
