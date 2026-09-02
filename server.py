import os
from contextlib import contextmanager

from fastapi import FastAPI, Header, WebSocket, WebSocketDisconnect

from services.chat_service import Master
from services.knowledge_service import KnowledgeService
from utils.logger_handler import logger

app = FastAPI()


@contextmanager
def request_api_keys(
    dashscope_api_key: str | None = None,
    yuanfenju_api_key: str | None = None,
    tavily_api_key: str | None = None,
):
    # 通过请求头临时注入 API Key，只在当前请求生命周期内生效，不写回服务器环境。
    previous_values = {}
    key_map = {
        "DASHSCOPE_API_KEY": dashscope_api_key,
        "YUANFENJU_API_KEY": yuanfenju_api_key,
        "TAVILY_API_KEY": tavily_api_key,
    }

    try:
        for env_name, value in key_map.items():
            previous_values[env_name] = os.environ.get(env_name)
            if value and value.strip():
                os.environ[env_name] = value.strip()
        yield
    finally:
        for env_name, previous_value in previous_values.items():
            if previous_value is None:
                os.environ.pop(env_name, None)
            else:
                os.environ[env_name] = previous_value

    
@app.get("/")
def read_root():
    # 健康检查接口，便于确认后端服务是否已经启动。
    return {"Hello": "World"}

@app.post("/chat")
def chat(
    query: str,
    user_id: str = "default",
    dashscope_api_key: str | None = Header(default=None, alias="X-DASHSCOPE-API-KEY"),
    yuanfenju_api_key: str | None = Header(default=None, alias="X-YUANFENJU-API-KEY"),
    tavily_api_key: str | None = Header(default=None, alias="X-TAVILY-API-KEY"),
):
    # 每次聊天都重新构建 Master，确保本次请求使用最新的上下文和密钥。
    with request_api_keys(dashscope_api_key, yuanfenju_api_key, tavily_api_key):
        master = Master()
        result = master.run(query, user_id=user_id)
    return {"message": result}

from fastapi import Body


@app.post("/add_urls")
def add_urls(
    URL: str,
    dashscope_api_key: str | None = Header(default=None, alias="X-DASHSCOPE-API-KEY"),
    yuanfenju_api_key: str | None = Header(default=None, alias="X-YUANFENJU-API-KEY"),
    tavily_api_key: str | None = Header(default=None, alias="X-TAVILY-API-KEY"),
):
    # 入库接口也允许走同样的临时密钥注入方式，适合前端直连调试。
    with request_api_keys(dashscope_api_key, yuanfenju_api_key, tavily_api_key):
        knowledge_service = KnowledgeService()
        result = knowledge_service.add_urls(URL)
    logger.info(f"URL ingest status={result['status']} chunks={result['chunk_count']} source={result['source_name']}")
    return result

@app.post("/add_pdfs")
def add_pdfs(
    pdf_path: str,
    dashscope_api_key: str | None = Header(default=None, alias="X-DASHSCOPE-API-KEY"),
    yuanfenju_api_key: str | None = Header(default=None, alias="X-YUANFENJU-API-KEY"),
    tavily_api_key: str | None = Header(default=None, alias="X-TAVILY-API-KEY"),
):
    # PDF 入库依赖本地可访问路径，前端一般会先把上传文件保存到临时目录。
    with request_api_keys(dashscope_api_key, yuanfenju_api_key, tavily_api_key):
        knowledge_service = KnowledgeService()
        result = knowledge_service.add_pdfs(pdf_path)
    logger.info(f"PDF ingest status={result['status']} chunks={result['chunk_count']} source={result['source_name']}")
    return result

@app.post("/add_texts")
def add_texts(
    text: str = Body(..., media_type="text/plain"),
    source_name: str = "manual_text",
    dashscope_api_key: str | None = Header(default=None, alias="X-DASHSCOPE-API-KEY"),
    yuanfenju_api_key: str | None = Header(default=None, alias="X-YUANFENJU-API-KEY"),
    tavily_api_key: str | None = Header(default=None, alias="X-TAVILY-API-KEY"),
):
    # 纯文本入库通常用于复制一段资料或笔记，直接按原文写入向量库。
    with request_api_keys(dashscope_api_key, yuanfenju_api_key, tavily_api_key):
        knowledge_service = KnowledgeService()
        result = knowledge_service.add_texts(text, source_name=source_name)
    logger.info(f"Text ingest status={result['status']} chunks={result['chunk_count']} source={result['source_name']}")
    return result

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    # 保留一个最小 WebSocket 示例，方便后续扩展实时交互能力。
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            await websocket.send_text(f"Message text was: {data}")
    except WebSocketDisconnect:
        print("Client disconnected")
        # await websocket.close()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)