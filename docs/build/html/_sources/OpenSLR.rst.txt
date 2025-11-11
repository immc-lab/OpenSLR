OpenSLR
============

概述
----

OpenSLR (Open Sign Language Recognition) 是一个基于PyTorch的连续手语识别框架，支持多种先进的模型架构和训练策略。

核心特性
--------

- **模块化设计**: 灵活的组件组合
- **多模型支持**: SlowFast、TLP、VAC、CorrNet等
- **高效训练**: 混合精度、内存映射数据加载
- **专业评估**: WER指标、Beam Search解码
- **实验管理**: Weights & Biases集成

项目结构
--------

.. code-block:: text

    OpenSLR/
    ├── main.py                 # 程序入口
    ├── Manager/               # 管理器模块
    │   ├── ArgumentManager.py
    │   ├── ConfigManager.py
    │   └── ...
    ├── Dataset/               # 数据集模块
    │   └── dataloader_video.py
    ├── Pipline/               # 训练流水线
    │   └── single.py
    ├── models/        # 模型构建
    │   └── build_function.py
    └── Container/             # 模型容器
        └── base.py

快速使用
--------

.. code-block:: python

    from OpenSLR import init, run, infer
    
    # 训练模式
    init()
    run()
    
    # 推理模式
    result = infer(video_data, video_length)

主模块API
---------

main.py 提供了以下主要函数：

- ``init()`` - 初始化所有管理器组件
- ``run()`` - 启动训练或测试流程  
- ``infer(video, video_length)`` - 视频推理接口

配置示例
--------

.. code-block:: python

    # 典型使用流程
    if __name__ == "__main__":
        init()  # 初始化系统
        run()   # 开始训练