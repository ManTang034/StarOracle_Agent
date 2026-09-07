import time

from langchain_community.chat_models.tongyi import BaseChatModel, ChatTongyi
from langchain_core.messages import HumanMessage

from utils.config_handler import model_conf
from utils.logger_handler import logger
from utils.observability import is_trace_debug

from .agent_factory import build_agent
from .emotion_service import EmotionService
from .knowledge_service import KnowledgeService
from .memory_service import MemoryService
from .mood_config import MOODS


class Master:
    def __init__(self):
        # 三个模型分别负责聊天生成、情绪识别和长期记忆抽取，方便单独替换或调参。
        self.chat_model: BaseChatModel = ChatTongyi(**model_conf["chat_model"])
        self.emotion_model: BaseChatModel = ChatTongyi(**model_conf["emotion_model"])
        self.memory_model: BaseChatModel = ChatTongyi(**model_conf["memory_extract_model"])

        self.user_emotion = "default"
        self.emotion_service = EmotionService(self.emotion_model)
        self.memory_service = MemoryService(self.memory_model)
        self.knowledge_service = KnowledgeService()
        self.agent = build_agent(self.chat_model, self.user_emotion)

    def run(self, query: str, user_id: str = "default"):
        # 先召回长期记忆和知识库上下文，再识别情绪，最后动态重建 Agent。
        start_time = time.perf_counter()
        trace_debug = is_trace_debug()
        logger.info(f"[chat] start user_id={user_id} query={query}")

        try:
            memory_context = self.memory_service.retrieve_context(user_id, query)
            knowledge_context = self.knowledge_service.retrieve_context(query)
            user_emotion = self.emotion_service.detect(query)

            self.user_emotion = user_emotion if user_emotion in MOODS else "default"
            role_set = MOODS[self.user_emotion]["roleSet"]
            logger.info(f"[chat] emotion_detected raw={user_emotion} resolved={self.user_emotion}")

            if trace_debug:
                logger.info(f"[chat] memory_context={memory_context or '<empty>'}")
                logger.info(f"[chat] knowledge_context={knowledge_context or '<empty>'}")
                logger.info(f"[chat] role_set={role_set}")
            else:
                logger.info(
                    f"[chat] context_summary memory_chars={len(memory_context)} knowledge_chars={len(knowledge_context)} role={self.user_emotion}"
                )

            self.agent = build_agent(
                self.chat_model,
                self.user_emotion,
                memory_context=memory_context,
                knowledge_context=knowledge_context,
            )

            messages = [HumanMessage(content=query)]
            result = self.agent.invoke({"messages": messages})
            answer = result["messages"][-1].content

            if trace_debug:
                logger.info(f"[chat] answer={answer}")
            else:
                preview = answer if len(answer) <= 160 else f"{answer[:160]}..."
                logger.info(f"[chat] answer_preview={preview}")

            saved_count = self.memory_service.remember(user_id, query, answer)
            logger.info(f"[chat] memory_saved_count={saved_count}")
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            logger.info(f"[chat] finished elapsed_ms={elapsed_ms:.1f}")
            return answer
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            logger.exception(f"[chat] failed elapsed_ms={elapsed_ms:.1f} error={exc}")
            raise
