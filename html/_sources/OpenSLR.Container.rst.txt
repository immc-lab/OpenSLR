OpenSLR.Container
========================

概述
----

容器模块提供了灵活的模型组件组合机制，支持模块化的神经网络架构设计。

核心类
------

Container 类
~~~~~~~~~~~~

通用的模块包装器，支持顺序执行和数据流传递。

**构造函数**: ``Container(modules)``

**参数**:
- ``modules``: 模块列表或字典

**主要方法**:
- ``forward(data)``: 顺序执行所有模块，更新数据字典

容器类支持：
- **顺序执行**: ModuleList形式的模块序列
- **数据流传递**: 通过字典传递中间结果
- **灵活组合**: 动态添加和移除模块

SignLanguageModel 类
~~~~~~~~~~~~~~~~~~~~

手语识别模型的主类，组合四个核心容器。

**构造函数**: 
``SignLanguageModel(spatial_module_container, temporal_module_container, loss_module_container, decoder)``

**参数**:
- ``spatial_module_container``: 空间特征提取容器
- ``temporal_module_container``: 时序建模容器  
- ``loss_module_container``: 损失计算容器
- ``decoder``: 序列解码器

**主要方法**:
- ``forward(data)``: 完整的前向传播流程
- ``backward_hook(module, grad_input, grad_output)``: 梯度处理钩子

设计模式
--------

数据流设计
~~~~~~~~~~~

.. code-block:: python

    # 数据通过字典在模块间传递
    data = {
        'vid': input_video,
        'vid_lgt': video_length,
        'feature': spatial_features,      # 空间模块输出
        'logits': classification_logits,  # 时序模块输出
        'loss': total_loss,               # 损失模块输出
        'recognized_sents': final_output  # 解码器输出
    }

容器组合模式
~~~~~~~~~~~~

数据流向图::

    输入数据 → 空间容器 → 时序容器 → 损失容器 → 解码器 → 输出结果

使用示例
--------

基础使用
~~~~~~~~

.. code-block:: python

    from OpenSLR.Container import Container, SignLanguageModel
    
    # 创建自定义容器
    spatial_modules = Container([
        ResNet(args),
        AttentionModule(args)
    ])
    
    temporal_modules = Container([
        TemporalConv1D(args),
        BiLSTM(args),
        Classifier(args)
    ])
    
    # 构建完整模型
    model = SignLanguageModel(
        spatial_module_container=spatial_modules,
        temporal_module_container=temporal_modules,
        loss_module_container=loss_modules,
        decoder=decoder
    )

高级用法
~~~~~~~~

动态模块添加：

.. code-block:: python

    # 运行时添加新模块
    model.temporal_module_container.module_list.append(
        AdditionalTemporalModule(args)
    )
    
    # 替换模块
    model.spatial_module_container.module_list[0] = NewBackbone(args)

梯度处理
--------

backward_hook 方法
~~~~~~~~~~~~~~~~~~

后向钩子函数用于梯度稳定性处理：

.. code-block:: python

    def backward_hook(module, grad_input, grad_output):
        for g in grad_input:
            g[g != g] = 0  # 处理NaN梯度

最佳实践
--------

1. **模块独立性**: 每个模块应独立处理输入输出
2. **数据字典规范**: 使用一致的键名传递数据
3. **错误处理**: 模块应处理异常输入情况
4. **内存优化**: 及时释放中间结果减少内存占用