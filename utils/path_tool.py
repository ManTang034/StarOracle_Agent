from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def get_abs_path(relative_path: str) -> str:
    return str(PROJECT_ROOT / relative_path)

if __name__ == "__main__":
    relative_path = "path_tool.py"
    abs_path = get_abs_path(relative_path)
    print(f"Relative path: {relative_path}")
    print(f"Absolute path: {abs_path}")