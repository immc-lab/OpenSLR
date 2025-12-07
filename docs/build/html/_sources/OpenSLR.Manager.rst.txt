OpenSLR.Manager
================

Overview
--------
The Manager component is the central coordinator of the OpenSLR toolbox. It handles global state management, configuration, logging, device management, and more.

Structure
---------
The Manager is divided into two categories:

1. **Base Managers**: Common utilities shared across all managers.
2. **Deep Learning Managers**: Specialized managers for training and evaluation.

Sub-Managers
------------

Argument Manager
~~~~~~~~~~~~~~~~
Parses and stores command-line arguments.

.. code-block:: python

    from OpenSLR.Manager import ArgumentManager
    args = ArgumentManager.get_args()

Config Manager
~~~~~~~~~~~~~~
Loads and validates configuration files.

.. code-block:: python

    from OpenSLR.Manager import ConfigManager
    cfg = ConfigManager.get_config()

Log Manager
~~~~~~~~~~~
Manages system-wide logging.

.. code-block:: python

    from OpenSLR.Manager import LogManager
    logger = LogManager.get_logger()

Device Manager
~~~~~~~~~~~~~~
Specifies GPU devices for training.

.. code-block:: python

    from OpenSLR.Manager import DeviceManager
    device = DeviceManager.get_device()

Module Manager
~~~~~~~~~~~~~~
Handles registration and instantiation of modules.

.. code-block:: python

    from OpenSLR.Manager import ModuleManager
    model = ModuleManager.get_module("ResNet")

Experiment Manager
~~~~~~~~~~~~~~~~~~
Manages experiment artifacts (directories, checkpoints).

Collect Manager
~~~~~~~~~~~~~~~
Aggregates data across processes for distributed training.

Dataloader Manager
~~~~~~~~~~~~~~~~~~
Coordinates data loader creation.

Dataset Manager
~~~~~~~~~~~~~~~
Manages dataset registration and metadata.

Global State Management
-----------------------
All managers use Python class variables and `@classmethod` decorators to ensure global access without passing object instances.