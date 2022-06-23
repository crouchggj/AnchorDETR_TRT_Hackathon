## 总述
本项目主要贡献包括:
- 基于TensorRT在Nvidia GPU平台实现Anchor DETR模型的转换和加速
  - 开源代码地址：<https://github.com/megvii-research/AnchorDETR>
- 优化效果（精度和加速比）
- 提供了在Docker里面代码编译、运行步骤的完整说明

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

### 模型优化的难点
- 原始模型导出ONNX时，出现错误
```
Traceback (most recent call last):
  File "export_onnx.py", line 61, in <module>
    export_onnx()
  File "export_onnx.py", line 48, in export_onnx
    model_sim, check_ok=simplify(onnx.load(onnx_path))
  File "/usr/local/lib/python3.6/dist-packages/onnx_simplifier-0.3.5-py3.6.egg/onnxsim/onnx_simplifier.py", line 483, in simplify
  File "/usr/local/lib/python3.6/dist-packages/onnx_simplifier-0.3.5-py3.6.egg/onnxsim/onnx_simplifier.py", line 384, in fixed_point
  File "/usr/local/lib/python3.6/dist-packages/onnx_simplifier-0.3.5-py3.6.egg/onnxsim/onnx_simplifier.py", line 480, in constant_folding
  File "/usr/local/lib/python3.6/dist-packages/onnx/checker.py", line 99, in check_model
    protobuf_string = model.SerializeToString()
ValueError: Message onnx.ModelProto exceeds maximum protobuf size of 2GB: 7753581684
```
    
> 通过查看github issue需要更新util/misc.py中nested_tensor_from_tensor_list方法，防止在进行ONNX导出出现异常，参考如下：
> 
> <https://github.com/megvii-research/AnchorDETR/issues/10>
> 
> <https://github.com/facebookresearch/detr/pull/173>

- 使用trtexec工具模型转换时，出现错误，需要针对算子进行改造
- ONNX模型带有大量零碎的算子，可以通过自定义插件进行整体，能进一步提升模型的运行速度
## 优化过程
### 冗余算子移除
使用onnx surgeon去除无意义的胶水算子，以方便TensorRT更好的算子融合。

![胶水算子0](img/unused_op1.png)
![胶水算子1](img/unused_op2.png)
### 构建期提前计算
针对固定表结构在构建模型时进行提前计算，减少模型运行时开销。
如下图所示，通过将2个Clip算子、Div以及Log算子的计算过程提前进行，就能有效的融合进Add算子中。
![fuse](img/fuse.png)
### LayerNorm算子融合
针对导出LayerNorm存在大量胶水算子，通过手动融合，实现自定义算子的方式加速模型运行。
下面展示算子融合前后的对比

![fuse0](img/layernorm_fuse.png)

通过迁移适配FastTransformer中LayerNorm算子，在测试显卡T4上最高实现30%以上的性能提升。

![profile](img/nsys_profile.png)

## 优化效果
### 算子数量对比
从2325降低到1985，整体算子降低14%。
### 性能和精度对比

|优化手段|硬件平台      |精度 |延迟(ms)|帧率(fps)|
|-------|-------------|----|-------|---------|
|pytorch加速|V100     |FP32|62.5   |16       |
|TensorRT加速|T4      |FP32|||
|TensorRT加速|T4      |FP16|||
|优化手段+TensorRT加速|T4      |FP16|||
|优化手段+TensorRT加速|T4      |INT8|||

## 代码框架
- AnchorDETR
  - AnchorDETR官方代码，新追加了导出脚本
- Model
  - 模型位置，通过[百度云盘](https://pan.baidu.com/share/init?surl=iB8qtVPb9dWHYgA5z1I4xg)（code: hh13）下载模型，本实验采用模型为anchor-detr-dc5。
- Trt
  - LayerNorm: layernorm自定义算子，适配自FastTransformer
  - TrtExec.cpp: TensorRT模型加速代码
  - surgeonModel.py: onnx模型融合简化脚本
  - profileModel.py: TensorRT模型性能评估脚本
  - data: npz文件为模型性能评估对比数据，calibration为模型int8量化数据

## Docker运行
* 搭建Docker运行环境

  1. docker pull

* 导出ONNX模型

  1. 通过百度云盘下载anchor-detr-dc5模型，将模型放置于Model目录。
  2. 进入AnchorDERT，执行模型导出脚本

  `python export_onnx.py --checkpoint ../Model/anchor-detr-dc5.pth`

  3. 模型导出成功，会在Model目录下创建anchor-detr-dc5.onnx。

* 模型优化
  1. 进入Trt目录，运行surgeonModel.py脚本，优化模型。

  `python surgeonModel.py ../Model/anchor-detr-dc5.onnx ../Model/dc5_new.onnx`
  
  2. 模型优化成功，会在Model目录下创建dc5_new.onnx。

* TensorRT加速
  1. 进入Trt目录，编译生成TensorRT转换可执行文件。
  `mkdir build && cd build && cmake .. && make`
  2. 运行./build/AnchorDETRTrt加速模型
    - FP32模型加速: 
    
    `./build/AnchorDETRTrt ../Model/dc5_new.onnx 0 0`
    - FP16模型加速: 
        
    `./build/AnchorDETRTrt ../Model/dc5_new.onnx 1 0`

    - INT8模型加速:

    `./build/AnchorDETRTrt ../Model/dc5_new.onnx 0 1`
* 性能和精度评估
  1. 进入Trt目录，执行profileModel.py脚本。

  `python --plan AnchorDETR.plan`

  2. 观察输出结果。



