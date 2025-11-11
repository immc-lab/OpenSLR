OpenSLR.Pipline
========================

概述
----

训练流水线模块负责模型训练、验证和推理的具体实现，支持混合精度训练和多任务学习。

核心函数
--------

seq_train 函数
~~~~~~~~~~~~~~

训练流程函数。

**函数签名**: 
``seq_train(loader, model, optimizer, scheduler, device, epoch_idx, loss_weights=None)``

**参数**:
- ``loader``: 数据加载器
- ``model``: 模型实例
- ``optimizer``: 优化器
- ``scheduler``: 学习率调度器
- ``device``: 训练设备
- ``epoch_idx``: 当前epoch索引
- ``loss_weights``: 损失权重配置

训练流程：

1. 设置模型为训练模式
2. 数据加载和设备转移
3. 混合精度前向传播
4. 多任务损失计算
5. 梯度缩放和反向传播
6. 参数更新和学习率调度

seq_eval 函数
~~~~~~~~~~~~~

评估流程函数。

**函数签名**:
``seq_eval(cfg, loader, model, device, mode, epoch, work_dir)``

**参数**:
- ``cfg``: 配置对象
- ``loader``: 数据加载器
- ``model``: 模型实例
- ``device``: 评估设备
- ``mode``: 评估模式
- ``epoch``: 当前epoch
- ``work_dir``: 工作目录

评估流程：

1. 设置模型为评估模式
2. 批量推理和结果收集
3. 结果文件输出
4. WER指标计算

write2file 函数
~~~~~~~~~~~~~~~

结果写入函数。

**函数签名**: ``write2file(path, info, output)``

**参数**:
- ``path``: 输出文件路径
- ``info``: 样本信息列表
- ``output``: 识别输出列表

训练配置
--------

混合精度训练
~~~~~~~~~~~~

.. code-block:: python

    with autocast():
        ret_dict = model(data)
        loss = ret_dict['loss']
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

多任务损失
~~~~~~~~~~

支持的损失类型：

- **ConvCTC**: 卷积特征的CTC损失
- **SeqCTC**: 序列特征的CTC损失  
- **Dist**: 距离损失
- **Cu**: 内容一致性损失
- **Cp**: 内容保持损失

评估指标
--------

WER (Word Error Rate)
~~~~~~~~~~~~~~~~~~~~~

词错误率计算公式：

.. math::
    WER = \frac{S + D + I}{N} \times 100\%

其中：
- S: 替换错误数
- D: 删除错误数  
- I: 插入错误数
- N: 总词数

使用示例
--------

.. code-block:: python

    from OpenSLR.Pipline import seq_train, seq_eval
    
    # 训练一个epoch
    loss_values = seq_train(
        loader=train_loader,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epoch_idx=epoch,
        loss_weights=loss_weights
    )
    
    # 评估模型
    wer_score = seq_eval(
        cfg=config,
        loader=dev_loader,
        model=model,
        device=device,
        mode="dev",
        epoch=epoch,
        work_dir=work_dir
    )