Installation
============

系统要求
--------

- Python 3.7+
- PyTorch 1.8+
- CUDA 11.0+ (GPU训练)

依赖安装
--------

.. code-block:: bash

    pip install torch torchvision
    pip install opencv-python numpy scipy
    pip install loguru rich wandb tqdm pyyaml
    pip install faulthandler

从源码安装
----------

.. code-block:: bash

    git clone https://github.com/your-organization/OpenSLR.git
    cd OpenSLR
    pip install -e .

环境配置
--------

1. 设置CUDA设备：

.. code-block:: bash

    export CUDA_VISIBLE_DEVICES=0,1,2,3

2. 准备数据集配置文件：

.. code-block:: bash

    cp configs/phoenix2014.yaml.example configs/phoenix2014.yaml
    # 编辑配置文件中的路径

验证安装
--------

.. code-block:: python

    import torch
    from OpenSLR import build_slowfast
    print("OpenSLR安装成功！")