from pathlib import Path

import yaml

from .path_tool import get_abs_path

DEFAULT_PROMPTS_CONFIG_PATH = get_abs_path("config/prompts.yml")
DEFAULT_MODELS_CONFIG_PATH = get_abs_path("config/models.yml")
DEFAULT_LOGS_CONFIG_PATH = get_abs_path("config/logs.yml")


def load_yaml_config(config_path: str, encoding: str = "utf-8"):
    config_file = Path(config_path)
    with config_file.open("r", encoding=encoding) as file:
        return yaml.safe_load(file)

def load_prompt_config(config_path: str = DEFAULT_PROMPTS_CONFIG_PATH, encoding: str = "utf-8"):
    return load_yaml_config(config_path, encoding=encoding)


def load_model_config(config_path: str = DEFAULT_MODELS_CONFIG_PATH, encoding: str = "utf-8"):
    return load_yaml_config(config_path, encoding=encoding)


def load_logs_config(config_path: str = DEFAULT_LOGS_CONFIG_PATH, encoding: str = "utf-8"):
    return load_yaml_config(config_path, encoding=encoding)


prompts_conf = load_prompt_config()
model_conf = load_model_config()
logs_conf = load_logs_config()

if __name__ == "__main__":
    print(prompts_conf["system_prompt_path"])