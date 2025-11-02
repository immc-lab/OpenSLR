import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

from manager.argument_manager import ArgumentManger
from manager.config_manager import ConfigManager
from manager.log_manager import LogManager
from manager.experiment_manager import ExperimentManager
from manager.dataset_manager import DatasetManager
from manager.dataloader_manager import DataloaderManager
from manager.module_manager import ModuleManager

def init():
    ArgumentManger.init()
    ConfigManager.init()
    ArgumentManger.map(ConfigManager.get())
    LogManager.init()
    DatasetManager.init()
    ModuleManager.init ( )
    DataloaderManager.init()
    ExperimentManager.init(ArgumentManger.get())

def run():
    ExperimentManager.run()

def infer(video, video_length):
    return ExperimentManager.run_inference(video, video_length)

if __name__ == "__main__":
    init()
    run()