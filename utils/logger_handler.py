import logging
import os
from functools import lru_cache
from logging.handlers import RotatingFileHandler

from .config_handler import logs_conf
from .path_tool import get_abs_path

# 日志保存到项目内固定目录，开源复现时无需额外手工创建路径。
LOG_ROOT_DIR = get_abs_path(logs_conf.get("log_dir", "logs"))

# 启动时自动创建目录，避免首次运行因为目录不存在而报错。
os.makedirs(LOG_ROOT_DIR, exist_ok=True)

# 统一日志格式，方便排查 Agent、工具调用和知识库检索问题。
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
    # 记录器本身设为 DEBUG，是否真正输出由控制台/文件 handler 控制。
    logger.setLevel(logging.DEBUG)

    console_level = _resolve_log_level(logs_conf.get("console_level", console_level), console_level)
    file_level = _resolve_log_level(logs_conf.get("file_level", file_level), file_level)

    # 模块可能被多次导入，先判断是否已经挂载过 handler。
    if logger.handlers:
        return logger

    # 控制台输出用于本地调试和复现问题。
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(DEFAULT_LOG_FORMAT)
    logger.addHandler(console_handler)

    # 文件输出用于保留历史运行轨迹，默认启用滚动，避免日志文件无限增大。
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

# 提供模块级 logger，其他文件直接导入即可使用。
logger = get_logger()

if __name__ == "__main__":
    logger.info("This is an info message.")
    logger.debug("This is a debug message.")
    logger.error("This is an error message.")
    logger.warning("This is a warning message.")