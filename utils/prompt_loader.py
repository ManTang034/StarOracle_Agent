from .config_handler import prompts_conf
from .logger_handler import logger
from .path_tool import get_abs_path


def _load_prompt(prompt_path: str, encoding: str) -> str:
    """从磁盘读取单个提示词文件，便于将提示词从代码中拆出去维护。"""
    try:
        with open(prompt_path, "r", encoding=encoding) as file:
            return file.read()
    except Exception as e:
        logger.error(f"Error loading prompt from {prompt_path}: {str(e)}")
        raise e


def load_system_prompt(encoding: str="utf-8") -> str:
    """读取系统提示词，Agent 的角色设定和上下文模板都来自这里。"""
    try:
        system_prompt_path = get_abs_path(prompts_conf["system_prompt_path"])
    except KeyError as e:
        logger.error(f"KeyError: {str(e)}. Please check the prompts configuration.")
        raise e

    return _load_prompt(system_prompt_path, encoding)

def load_emotion_prompt(encoding: str="utf-8") -> str:
    """读取情绪识别提示词，用于判断用户当前语气并切换回复风格。"""
    try:
        emotion_prompt_path = get_abs_path(prompts_conf["emotion_prompt_path"])
    except KeyError as e:
        logger.error(f"KeyError: {str(e)}. Please check the prompts configuration.")
        raise e

    return _load_prompt(emotion_prompt_path, encoding)


def load_memory_prompt(encoding: str="utf-8") -> str:
    """读取记忆抽取提示词，用于从对话中提取长期记忆片段。"""
    try:
        memory_prompt_path = get_abs_path(prompts_conf["memory_prompt_path"])
    except KeyError as e:
        logger.error(f"KeyError: {str(e)}. Please check the prompts configuration.")
        raise e

    return _load_prompt(memory_prompt_path, encoding)

if __name__ == "__main__":
    print(load_system_prompt())
    print(load_emotion_prompt())