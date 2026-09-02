from langchain_community.chat_models.tongyi import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from utils.prompt_loader import load_emotion_prompt


class EmotionService:
    def __init__(self, chat_model: BaseChatModel):
        self.chat_model = chat_model

    def detect(self, query: str) -> str:
        # 情绪识别结果会映射到不同角色设定，从而改变后续回答语气。
        prompt = ChatPromptTemplate.from_template(load_emotion_prompt())
        chain = prompt | self.chat_model | StrOutputParser()
        return chain.invoke({"query": query})