import os
from datetime import datetime

import requests
from langchain_core.tools import tool
from langchain_tavily import TavilySearch

from utils.config_handler import model_conf
from utils.logger_handler import logger


@tool(description="只有需要了解实时信息或不知道的事情时才会使用这个工具")
def search(query: str) -> str:
    result = TavilySearch().run(query)
    return result


@tool(description="获取当前本机的准确日期时间，涉及今天、明天、本周、本月、现在等时间判断时必须优先使用这个工具")
def current_time() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


FORTUNE_SIGN_INDEX = {
    "白羊座": 0, "金牛座": 1, "双子座": 2, "巨蟹座": 3,
    "狮子座": 4, "处女座": 5, "天秤座": 6, "天蝎座": 7,
    "射手座": 8, "摩羯座": 9, "水瓶座": 10, "双鱼座": 11,
}

FORTUNE_PERIOD_KEY = {
    "today": "dailyFortune",
    "daily": "dailyFortune",
    "tomorrow": "tomorrowFortune",
    "today_tomorrow": "todayTomorrowFortune",
    "week": "weeklyFortune",
    "weekly": "weeklyFortune",
    "month": "monthlyFortune",
    "monthly": "monthlyFortune",
    "year": "yearlyFortune",
    "yearly": "yearlyFortune",
}


def _get_yuanfenju_api_key() -> str:
    config_key = model_conf.get("yuanfenju", {}).get("api_key", "").strip()
    if config_key:
        return config_key

    env_name = model_conf.get("yuanfenju", {}).get("api_key_env", "YUANFENJU_API_KEY")
    return os.environ.get(env_name, "").strip()


def _normalize_sign(sign: str) -> str:
    cleaned_sign = sign.strip()
    if cleaned_sign in FORTUNE_SIGN_INDEX:
        return cleaned_sign

    alias_map = {
        "白羊": "白羊座",
        "金牛": "金牛座",
        "双子": "双子座",
        "巨蟹": "巨蟹座",
        "狮子": "狮子座",
        "处女": "处女座",
        "天秤": "天秤座",
        "天蝎": "天蝎座",
        "射手": "射手座",
        "摩羯": "摩羯座",
        "水瓶": "水瓶座",
        "双鱼": "双鱼座",
    }
    return alias_map.get(cleaned_sign, cleaned_sign)


@tool(description="查询星座每日运势，支持今日、明日、今明、周运、月运、年运。参数 sign 传星座名称，如白羊座；period 传 today/tomorrow/week/month/year")
def daily_fortune(sign: str, period: str = "today") -> str:
    api_key = _get_yuanfenju_api_key()
    if not api_key:
        return "未配置元亨聚 API Key，请先在环境变量 YUANFENJU_API_KEY 或 config/models.yml 的 yuanfenju.api_key 中配置。"

    normalized_sign = _normalize_sign(sign)
    if normalized_sign not in FORTUNE_SIGN_INDEX:
        return f"不支持的星座：{sign}。请传入 12 星座之一，例如：白羊座、金牛座、双子座。"

    requested_period = period.strip().lower()
    period_key = FORTUNE_PERIOD_KEY.get(requested_period)
    if not period_key:
        return "不支持的运势周期，请使用 today/tomorrow/week/month/year。"

    payload = {
        "api_key": api_key,
        "type": 0,
        "title_yunshi": FORTUNE_SIGN_INDEX[normalized_sign],
        "lang": model_conf.get("yuanfenju", {}).get("lang", "zh-cn"),
        "parameter_style": model_conf.get("yuanfenju", {}).get("parameter_style", "english"),
    }

    try:
        response = requests.post(
            model_conf.get("yuanfenju", {}).get("endpoint", "https://api.yuanfenju.com/index.php/v1/Zhanbu/yunshi"),
            data=payload,
            timeout=15,
        )
        response.raise_for_status()
        result = response.json()
    except Exception as exc:
        logger.error(f"Error calling yuanfenju fortune api: {exc}")
        return f"查询星座运势失败：{exc}"

    if result.get("errcode") not in (0, "0"):
        return f"查询失败：{result.get('errmsg', '未知错误')}"

    data = result.get("data", {})
    if not isinstance(data, dict):
        return "查询成功，但返回数据格式异常。"

    fortune_block = data.get(period_key, {})
    if not isinstance(fortune_block, dict):
        return "查询成功，但该周期运势数据缺失。"

    lines = [
        f"星座：{data.get('fortuneType', normalized_sign)}",
        f"周期：{period}",
    ]

    for key, label in [
        ("score", "综合分数"),
        ("love", "爱情运势"),
        ("career", "事业运势"),
        ("wealth", "财富运势"),
        ("health", "健康运势"),
        ("mood", "心情运势"),
        ("luckyColor", "幸运颜色"),
        ("luckyNumber", "幸运数字"),
        ("luckyStone", "幸运宝石"),
        ("compatibleSign", "速配星座"),
        ("cautionarySign", "提防星座"),
    ]:
        value = fortune_block.get(key)
        if value:
            lines.append(f"{label}：{value}")

    content = fortune_block.get("fortune") or fortune_block.get("todayTomorrowFortune")
    if content:
        lines.append(f"运势解读：{content}")

    return "\n".join(lines)

if __name__ == "__main__":
    tool = TavilySearch(
        max_results=1,
        topic="general",
        # include_answer=False,
        # include_raw_content=False,
        # include_images=False,
        # include_image_descriptions=False,
        # search_depth="basic",
        # time_range="day",
        # include_domains=None,
        # exclude_domains=None
    )

    tool_msg = tool.invoke({"query": "What happened at the last wimbledon"})
    print(tool_msg)