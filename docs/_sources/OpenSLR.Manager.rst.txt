OpenSLR.Manager
========================

概述
----

管理器模块采用模块化设计，负责系统的各个功能组件的初始化和协调工作。

管理器列表
----------

ArgumentManager
~~~~~~~~~~~~~~~

命令行参数管理器。

**主要方法**:
- ``init()`` - 初始化参数解析器
- ``get(argument=None)`` - 获取参数值
- ``parse()`` - 解析命令行参数
- ``map(config)`` - 配置映射

功能：
- 命令行参数解析
- 参数验证和映射
- 配置文件集成

ConfigManager
~~~~~~~~~~~~~

配置管理器。

**主要方法**:
- ``init()`` - 初始化配置
- ``get(key=None)`` - 获取配置值
- ``load(config_path)`` - 加载配置文件

支持格式：YAML, JSON, TOML

LogManager
~~~~~~~~~~

日志管理器。

**主要方法**:
- ``init(output_path=None)`` - 初始化日志系统
- ``info(data)`` - 信息日志
- ``warning(data)`` - 警告日志  
- ``error(data)`` - 错误日志
- ``info_panel(data, title)`` - 面板式信息输出

特性：
- 多输出（控制台+文件）
- Weights & Biases集成
- 美观的格式化输出

DatasetManager
~~~~~~~~~~~~~~

数据集管理器。

**主要方法**:
- ``init()`` - 初始化数据集
- ``get(mode="train")`` - 获取指定模式的数据集
- ``get_gloss_dict()`` - 获取词汇表

DataloaderManager
~~~~~~~~~~~~~~~~~

数据加载器管理器。

**主要方法**:
- ``init()`` - 初始化数据加载器
- ``get(mode="train")`` - 获取指定模式的数据加载器

ModuleManager
~~~~~~~~~~~~~

模块管理器。

**主要方法**:
- ``init()`` - 初始化模型模块
- ``get(module_name)`` - 获取指定模块
- ``set(mode)`` - 设置模型模式
- ``init_model_object()`` - 初始化模型对象

DeviceManager
~~~~~~~~~~~~~

设备管理器。

**主要方法**:
- ``init(device)`` - 初始化设备设置
- ``to(data)`` - 数据转移到设备

CollectManager
~~~~~~~~~~~~~~

数据收集管理器。

**主要方法**:
- ``init(args)`` - 初始化收集器
- ``collate(batch)`` - 批次数据整理
- ``set_kernel_sizes(kernel_sizes)`` - 设置卷积核大小

ExperimentManager
~~~~~~~~~~~~~~~~~

实验管理器。

**主要方法**:
- ``init(arg)`` - 初始化实验
- ``run()`` - 运行实验
- ``run_train()`` - 运行训练
- ``run_test()`` - 运行测试
- ``run_inference(video_data, video_length)`` - 运行推理

工作流程
--------

管理器初始化顺序::

    ArgumentManager → ConfigManager → LogManager → DatasetManager 
    → DataloaderManager → ModuleManager → DeviceManager → ExperimentManager

使用示例
--------

.. code-block:: python

    from OpenSLR.Manager import (
        ArgumentManager, ConfigManager, LogManager,
        DatasetManager, DataloaderManager, ModuleManager,
        DeviceManager, ExperimentManager
    )
    
    # 初始化所有管理器
    ArgumentManager.init()
    ConfigManager.init()
    ArgumentManager.map(ConfigManager.get())
    LogManager.init()
    DatasetManager.init()
    DataloaderManager.init()
    ModuleManager.init()
    DeviceManager.init(ArgumentManager.get("device"))
    ExperimentManager.init(ArgumentManager.get())
    
    # 开始实验
    ExperimentManager.run()