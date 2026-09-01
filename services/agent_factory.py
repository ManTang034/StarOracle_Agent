from langchain.agents import create_agent
from langchain_community.chat_models.tongyi import BaseChatModel

from tools.agent_tools import current_time, daily_fortune, search
from utils.config_handler import model_conf
from utils.prompt_loader import load_system_prompt

from .agent_middleware import AgentDebugMiddleware
from .mood_config import MOODS


def build_agent(chat_model: BaseChatModel, emotion: str, memory_context: str = "", knowledge_context: str = ""):
    role_set = MOODS.get(emotion, MOODS["default"])["roleSet"]
    system_prompt = load_system_prompt().format(
        your_roleSet=role_set,
        memory_context=memory_context,
        knowledge_context=knowledge_context,
    )
    middleware = [AgentDebugMiddleware()] if model_conf.get("agent", {}).get("enable_debug_middleware", True) else []
    return create_agent(
        model=chat_model,
        system_prompt=system_prompt,
        tools=[search, current_time, daily_fortune],
        middleware=middleware,
    )
