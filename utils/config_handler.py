from pathlib import Path

import yaml

from .path_tool import get_abs_path

# 默认配置文件都放在项目根目录下的 config 目录中，便于开源后直接复现。
DEFAULT_PROMPTS_CONFIG_PATH = get_abs_path("config/prompts.yml")
DEFAULT_MODELS_CONFIG_PATH = get_abs_path("config/models.yml")
DEFAULT_LOGS_CONFIG_PATH = get_abs_path("config/logs.yml")


def load_yaml_config(config_path: str, encoding: str = "utf-8"):
    """读取 YAML 配置文件，统一由这里入口加载，避免各处重复拼路径。"""
    config_file = Path(config_path)
    with config_file.open("r", encoding=encoding) as file:
        return yaml.safe_load(file)


def load_prompt_config(config_path: str = DEFAULT_PROMPTS_CONFIG_PATH, encoding: str = "utf-8"):
    """加载提示词配置，包含系统提示词、情绪提示词和记忆提示词路径。"""
    return load_yaml_config(config_path, encoding=encoding)


def load_model_config(config_path: str = DEFAULT_MODELS_CONFIG_PATH, encoding: str = "utf-8"):
    """加载模型与向量库配置，项目启动时会优先读取这里。"""
    return load_yaml_config(config_path, encoding=encoding)


def load_logs_config(config_path: str = DEFAULT_LOGS_CONFIG_PATH, encoding: str = "utf-8"):
    """加载日志配置，便于开源后通过 YAML 调整日志级别和输出目录。"""
    return load_yaml_config(config_path, encoding=encoding)


# 这里直接在模块导入时完成加载，后续服务模块可以像读字典一样使用配置。
prompts_conf = load_prompt_config()
model_conf = load_model_config()
logs_conf = load_logs_config()

if __name__ == "__main__":
    print(prompts_conf["system_prompt_path"])