from pathlib import Path

# 项目根目录用于把相对路径统一转换成绝对路径，避免不同启动目录导致找不到文件。
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def get_abs_path(relative_path: str) -> str:
    """把项目内相对路径转换为绝对路径，适合读取配置、提示词和本地持久化目录。"""
    return str(PROJECT_ROOT / relative_path)

if __name__ == "__main__":
    relative_path = "path_tool.py"
    abs_path = get_abs_path(relative_path)
    print(f"Relative path: {relative_path}")
    print(f"Absolute path: {abs_path}")