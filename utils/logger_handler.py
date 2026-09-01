import logging
import os
from functools import lru_cache
from logging.handlers import RotatingFileHandler

from .config_handler import logs_conf
from .path_tool import get_abs_path

# 日志保存的根目录
LOG_ROOT_DIR = get_abs_path(logs_conf.get("log_dir", "logs"))

# 确保日志目录存在
os.makedirs(LOG_ROOT_DIR, exist_ok=True)

# 日志的格式配置
DEFAULT_LOG_FORMAT = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
)


def _resolve_log_level(value, default):
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        return logging.getLevelName(value.upper()) if value.upper() in logging._nameToLevel else default
    return default


@lru_cache(maxsize=None)
def get_logger(
        name: str="agent",
    console_level: int=logging.INFO,
    file_level: int=logging.DEBUG,
        log_file=None
) -> logging.Logger:
    """
    获取日志记录器
    Args:
        name (str): 日志记录器的名称
        console_level (int): 控制台日志级别
        file_level (int): 文件日志级别
        log_file (str): 日志文件路径，如果为 None，则使用默认路径
    Returns:
        logging.Logger: 配置好的日志记录器
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # 设置为最低级别，确保所有日志都能被处理

    console_level = _resolve_log_level(logs_conf.get("console_level", console_level), console_level)
    file_level = _resolve_log_level(logs_conf.get("file_level", file_level), file_level)

    # 避免重复添加Handler
    if logger.handlers:
        return logger

    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(DEFAULT_LOG_FORMAT)
    logger.addHandler(console_handler)

    # 文件处理器
    if log_file is None:
        log_file = os.path.join(LOG_ROOT_DIR, logs_conf.get("log_filename", f"{name}.log"))

    file_handler = RotatingFileHandler(
        log_file,
        mode="a",
        encoding="utf-8",
        maxBytes=logs_conf.get("max_bytes", 5 * 1024 * 1024),
        backupCount=logs_conf.get("backup_count", 5),
    )
    file_handler.setLevel(file_level)
    file_handler.setFormatter(DEFAULT_LOG_FORMAT)
    logger.addHandler(file_handler)

    return logger

# 快捷获取日志器
logger = get_logger()

if __name__ == "__main__":
    logger.info("This is an info message.")
    logger.debug("This is a debug message.")
    logger.error("This is an error message.")
    logger.warning("This is a warning message.")