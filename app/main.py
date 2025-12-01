from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from app.services.rag_engine import MedicalGraphRAG
from app.models import ChatRequest, ChatResponse

rag_service = MedicalGraphRAG()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Khởi động: Load data
    rag_service.load_data_and_build_index()
    yield
    # Tắt: Dọn dẹp
    rag_service.close()

app = FastAPI(lifespan=lifespan, title="Medical AI Chatbot API")

@app.get("/")
def health_check():
    return {"status": "running", "ai_ready": rag_service.is_ready}

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    if not request.text:
        raise HTTPException(status_code=400, detail="Nội dung chat không được để trống")
    
    try:
        result = await rag_service.process_question(request.text)
        return ChatResponse(
            answer=result["answer"],
            debug_info=result.get("debug_info")
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))