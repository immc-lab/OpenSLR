Quickstart
============
5分钟上手OpenSLR
-----------------

1. 数据准备
~~~~~~~~~~~~

.. code-block:: python

    # 准备Phoenix2014数据集
    from OpenSLR.Dataset import VideoDataset
    
    dataset = VideoDataset(
        prefix="/path/to/phoenix2014",
        gloss_dict=gloss_dict,
        mode="train"
    )

2. 配置训练
~~~~~~~~~~~~

创建配置文件 ``configs/my_experiment.yaml``：

.. code-block:: yaml

    feeder: OpenSLR.Dataset.VideoDataset
    phase: train
    dataset: phoenix2014
    num_epoch: 80
    batch_size: 8
    
    model: OpenSLR.Build_Function.build_slowfast
    model_args:
        num_classes: 1296
        hidden_size: 1024

3. 启动训练
~~~~~~~~~~~~

.. code-block:: bash

    python main.py --config configs/my_experiment.yaml --work-dir ./work_dir/my_experiment

4. 模型推理
~~~~~~~~~~~~

.. code-block:: python

    from OpenSLR import infer
    
    # 加载视频并进行识别
    recognized_sents = infer(video_data, video_length)
    print("识别结果:", recognized_sents)

示例代码
--------

完整训练示例：

.. code-block:: python

    from OpenSLR.Manager import ArgumentManager, ConfigManager, ExperimentManager
    
    # 初始化系统
    ArgumentManager.init()
    ConfigManager.init()
    # ... 其他管理器初始化
    
    # 开始训练
    ExperimentManager.run()

常见问题
--------