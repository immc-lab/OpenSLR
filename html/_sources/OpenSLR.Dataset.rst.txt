OpenSLR.Dataset
========================

概述
----

数据集模块负责视频数据的加载、预处理和数据增强，支持多种数据格式和增强策略。

核心类
------

VideoDataset 类
~~~~~~~~~~~~~~~

视频数据集类，支持多种数据加载格式。

**构造函数**: 
``VideoDataset(prefix, gloss_dict, dataset, drop_ratio, num_gloss, mode, transform_mode, datatype, frame_interval, image_scale, allowable_vid_length)``

**主要参数**:
- ``prefix``: 数据集根路径
- ``gloss_dict``: 词汇表字典
- ``dataset``: 数据集名称
- ``mode``: 数据模式 (train/dev/test)
- ``datatype``: 数据格式类型
- ``transform_mode``: 是否启用数据增强

支持的数据类型：

- **video**: 直接读取视频文件
- **lmdb**: 数据库格式（高效IO）
- **memmap**: 内存映射文件（最高性能）
- **features**: 预提取特征

**主要方法**:
- ``__getitem__(idx)``: 获取指定索引的数据样本
- ``__len__()``: 返回数据集大小
- ``read_video(index)``: 读取视频文件
- ``read_memmap(index)``: 读取内存映射数据
- ``normalize(video, label)``: 数据归一化

数据增强模块
------------

video_augmentation 模块提供多种数据增强策略：

增强类列表：

- ``Compose(transforms)`` - 组合多个增强操作
- ``RandomCrop(size)`` - 随机裁剪
- ``CenterCrop(size)`` - 中心裁剪  
- ``RandomHorizontalFlip(prob)`` - 随机水平翻转
- ``TemporalRescale(temp_scaling, frame_interval)`` - 时序缩放
- ``RandomRotation(degrees)`` - 随机旋转
- ``ToTensor()`` - 转换为张量

增强策略包括：

- **空间增强**: 随机裁剪、水平翻转、中心裁剪
- **时序增强**: 时序缩放、帧采样
- **WER增强**: 模拟识别错误的数据增强

使用示例
--------

.. code-block:: python

    from OpenSLR.Dataset import VideoDataset
    
    dataset = VideoDataset(
        prefix="/path/to/dataset",
        gloss_dict=gloss_dict,
        dataset="phoenix2014",
        mode="train",
        datatype="memmap",  # 使用内存映射格式
        transform_mode=True  # 启用数据增强
    )
    
    # 获取单个样本
    video, label, info = dataset[0]

数据格式
--------

输入数据格式：

.. code-block:: python

    {
        'vid': torch.Tensor,      # [T, H, W, C] 视频帧
        'label': list,            # 标签索引序列
        'info': dict              # 元数据信息
    }

配置文件
--------

数据集配置文件示例：

.. code-block:: yaml

    dataset_root: "/path/to/dataset"
    dict_path: "/path/to/gloss_dict.npy"
    evaluation_dir: "/path/to/evaluation"
    evaluation_prefix: "phoenix2014"