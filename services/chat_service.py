from langchain_community.chat_models.tongyi import BaseChatModel, ChatTongyi
from langchain_core.messages import HumanMessage

from utils.config_handler import model_conf

from .agent_factory import build_agent
from .emotion_service import EmotionService
from .knowledge_service import KnowledgeService
from .memory_service import MemoryService
from .mood_config import MOODS


class Master:
    def __init__(self):
        self.chat_model: BaseChatModel = ChatTongyi(**model_conf["chat_model"])
        self.emotion_model: BaseChatModel = ChatTongyi(**model_conf["emotion_model"])
        self.memory_model: BaseChatModel = ChatTongyi(**model_conf["memory_extract_model"])

        self.user_emotion = "default"
        self.emotion_service = EmotionService(self.emotion_model)
        self.memory_service = MemoryService(self.memory_model)
        self.knowledge_service = KnowledgeService()
        self.agent = build_agent(self.chat_model, self.user_emotion)

    def run(self, query: str, user_id: str = "default"):
        memory_context = self.memory_service.retrieve_context(user_id, query)
        knowledge_context = self.knowledge_service.retrieve_context(query)
        user_emotion = self.emotion_service.detect(query)
        print(f"User emotion: {user_emotion}")
        self.user_emotion = user_emotion if user_emotion in MOODS else "default"
        self.agent = build_agent(
            self.chat_model,
            self.user_emotion,
            memory_context=memory_context,
            knowledge_context=knowledge_context,
        )
        print(MOODS[self.user_emotion]["roleSet"])
        messages = [
            HumanMessage(content=query),
        ]
        result = self.agent.invoke({"messages": messages})
        answer = result["messages"][-1].content
        self.memory_service.remember(user_id, query, answer)
        return answer
