OpenSLR.Build_Function
========================

概述
----

模型构建函数模块提供了多种先进手语识别模型的工厂函数，支持灵活的模型组合和配置。

可用模型
--------

build_slowfast
~~~~~~~~~~~~~~

构建SlowFast手语识别模型。

**函数签名**: ``build_slowfast(args, gloss_dict, loss_weights)``

**参数**:
- ``args``: 模型参数字典
- ``gloss_dict``: 词汇表字典
- ``loss_weights``: 损失权重配置

**返回**: 配置完整的SignLanguageModel实例

SlowFast网络特点：
- 双路径时序建模（慢路径+快路径）
- 多尺度特征提取
- 高效的时空特征融合

build_tlp
~~~~~~~~~

构建时序金字塔(TLP)手语识别模型。

**函数签名**: ``build_tlp(args, gloss_dict, loss_weights)``

TLP (Temporal Pyramid) 特点：
- 时序金字塔结构
- 多尺度时序建模
- ResNet骨干网络

build_vac
~~~~~~~~~

构建视觉对齐约束(VAC)手语识别模型。

**函数签名**: ``build_vac(args, gloss_dict, loss_weights)``

VAC (Visual Alignment Constraint) 特点：
- 视觉对齐约束
- 改进的时序卷积
- 归一化线性分类器

build_sen
~~~~~~~~~

构建SEN手语识别模型。

**函数签名**: ``build_sen(args, gloss_dict, loss_weights)``

SEN (Squeeze-and-Excitation Network) 特点：
- 通道注意力机制
- 自适应特征重标定
- 改进的ResNet架构

build_corrnet
~~~~~~~~~~~~~

构建相关性网络(CorrNet)手语识别模型。

**函数签名**: ``build_corrnet(args, gloss_dict, loss_weights)``

CorrNet特点：
- 相关性学习网络
- 改进的ResNet-18骨干
- 相关性感知的时序建模

模型配置
--------

通用参数
~~~~~~~~

.. code-block:: python

    model_args = {
        "num_classes": 1296,      # 词汇表大小
        "hidden_size": 1024,      # 隐藏层维度
        "c2d_type": "slowfast101", # 2D卷积类型
        "kernel_size": ["K5", "P2", "K5", "P2"],  # 卷积核配置
        "use_bn": True,           # 使用BatchNorm
    }

损失权重配置
~~~~~~~~~~~~

.. code-block:: python

    loss_weights = {
        "ConvCTC": 1.0,    # 卷积CTC损失权重
        "SeqCTC": 1.0,     # 序列CTC损失权重
        "Dist": 25.0,      # 距离损失权重
        "Cu": 0.001,       # 内容一致性损失权重
        "Cp": 0.001,       # 内容保持损失权重
    }

使用示例
--------

.. code-block:: python

    from OpenSLR.Build_Function import build_slowfast
    
    # 构建SlowFast模型
    model = build_slowfast(
        args=model_args,
        gloss_dict=gloss_dict,
        loss_weights=loss_weights
    )
    
    # 模型输出格式
    output = model(data)
    # output包含: loss, total_loss, recognized_sents

自定义模型
----------

实现新的构建函数：

.. code-block:: python

    def build_custom_model(args, gloss_dict, loss_weights):
        return SignLanguageModel(
            spatial_module_container=Container([CustomSpatialModule(args)]),
            temporal_module_container=Container([CustomTemporalModule(args)]),
            loss_module_container=Container([CustomLoss(loss_weights)]),
            decoder=CustomDecoder(args, gloss_dict)
        )