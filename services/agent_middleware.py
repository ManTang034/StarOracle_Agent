from langchain.agents.middleware import AgentMiddleware

from utils.logger_handler import logger


class AgentDebugMiddleware(AgentMiddleware):
    def before_agent(self, state, runtime):
        # 这几个钩子主要用于记录 ReAct 调用链，方便排查模型和工具的执行顺序。
        logger.info("[agent] start")
        return None

    def before_model(self, state, runtime):
        logger.info("[agent] before model call")
        return None

    def wrap_model_call(self, request, handler):
        logger.info("[agent] calling model")
        response = handler(request)
        logger.info("[agent] model call finished")
        return response

    def wrap_tool_call(self, request, handler):
        tool_name = request.tool_call.get("name", "unknown_tool")
        tool_args = request.tool_call.get("args", {})
        logger.info(f"[tool] calling {tool_name} with args={tool_args}")
        result = handler(request)
        logger.info(f"[tool] finished {tool_name}")
        return result

    def after_agent(self, state, runtime):
        logger.info("[agent] finished")
        return None
