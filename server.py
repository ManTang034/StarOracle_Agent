from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from services.chat_service import Master
from services.knowledge_service import KnowledgeService
from utils.logger_handler import logger

app = FastAPI()

knowledge_service = KnowledgeService()

    
@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/chat")
def chat(query: str, user_id: str = "default"):
    master = Master()
    result = master.run(query, user_id=user_id)
    return {"message": result}
from fastapi import Body


@app.post("/add_urls")
def add_urls(URL: str):
    result = knowledge_service.add_urls(URL)
    logger.info(f"URL ingest status={result['status']} chunks={result['chunk_count']} source={result['source_name']}")
    return result

@app.post("/add_pdfs")
def add_pdfs(pdf_path: str):
    result = knowledge_service.add_pdfs(pdf_path)
    logger.info(f"PDF ingest status={result['status']} chunks={result['chunk_count']} source={result['source_name']}")
    return result

@app.post("/add_texts")
def add_texts(text: str = Body(..., media_type="text/plain"), source_name: str = "manual_text"):
    result = knowledge_service.add_texts(text, source_name=source_name)
    logger.info(f"Text ingest status={result['status']} chunks={result['chunk_count']} source={result['source_name']}")
    return result

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
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