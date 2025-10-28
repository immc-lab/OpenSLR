# -*- encoding: utf-8 -*-
import json
import yaml
import toml
from .argument_manager import ArgumentManger

class ConfigManager:
    """
    The Class that handle all the configs files
    """

    @classmethod
    def init(cls):
        experiment_config_path = ArgumentManger.get("config")
        cls.load(experiment_config_path)

    @classmethod
    def load(cls, config_path):
        with open ( config_path , mode = "r" ) as f :
            data = yaml.load ( f , Loader = yaml.FullLoader )
        if not hasattr(cls, 'CONFIG_DATA'):
            setattr(cls, 'CONFIG_DATA', {})
        if isinstance(data, dict):
            # 将读取到的 dict 每个 key 都融合到 CONFIG_DATA 中
            getattr(cls, 'CONFIG_DATA').update(data)
        else:
            raise TypeError("仅支持字典类型的配置数据，请检查配置文件内容。")

    @classmethod
    def get(cls, key=None):
        if key is None:
            return getattr(cls, 'CONFIG_DATA')
        else:
            return getattr(cls, 'CONFIG_DATA')[key]

    # @classmethod
    # def map(cls):
    #     args = vars(ArgumentManger.ARGS)
    #     if hasattr(cls, 'CONFIG_DATA'):
    #         for section in getattr(cls, 'CONFIG_DATA'):
    #             if isinstance(getattr(cls, 'CONFIG_DATA')[section], dict):
    #                 for key in getattr(cls, 'CONFIG_DATA')[section]:
    #                     arg_name = f"{section.lower()}_{key.lower()}"
    #                     if arg_name in args and args[arg_name] is not None:
    #                         getattr(cls, 'CONFIG_DATA')[section][key] = args[arg_name]
    #                     elif key in args and args[key] is not None:
    #                         getattr(cls, 'CONFIG_DATA')[section][key] = args[key]

    @classmethod
    def __iter__(cls):
        return iter(getattr(cls, 'CONFIG_DATA'))



