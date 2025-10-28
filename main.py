import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

from manager.argument_manager import ArgumentManger
from manager.config_manager import ConfigManager
from manager.log_manager import LogManager
from manager.experiment_manager import ExperimentManager

def init():
    ArgumentManger.init()
    ConfigManager.init()
    ArgumentManger.map(ConfigManager.get())
    LogManager.init()
    ExperimentManager.init(ArgumentManger.get())
    ExperimentManager.start()

if __name__ == "__main__":
    init()