from fastapi import FastAPI, HTTPException, Request, Header, Depends, status
from contextlib import asynccontextmanager
from app.services.rag_engine import MedicalGraphRAG
from app.models import ChatRequest, ChatResponse
from app.config import settings
import logging
from app.logger import setup_logging

# 1. Khởi tạo cấu hình Log ngay đầu file
setup_logging()
logger = logging.getLogger("MedicalChatApp") # Đặt tên logger cụ thể

rag_service = MedicalGraphRAG()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Log khi khởi động
    logger.info(">>> Đang khởi động hệ thống Medical AI Chatbot...")
    try:
        rag_service.load_data_and_build_index()
        logger.info(">>> Load dữ liệu hoàn tất. Sẵn sàng phục vụ!")
    except Exception as e:
        logger.error(f">>> Lỗi khi load dữ liệu: {e}", exc_info=True)
    
    yield
    
    # Log khi tắt
    logger.info(">>> Đang tắt hệ thống...")
    rag_service.close()
    logger.info(">>> Hệ thống đã tắt hoàn toàn.")

app = FastAPI(lifespan=lifespan, title="Medical AI Chatbot API")

# --- 1. FUNCTION KIỂM TRA API KEY ---
async def verify_api_key(x_api_key: str = Header(..., alias="X-API-Key")):
    """
    Kiểm tra xem Header 'X-API-Key' có khớp với cấu hình không.
    Nếu không khớp -> Trả về lỗi 401 Unauthorized.
    """
    if x_api_key != settings.INTERNAL_API_KEY:
        logger.warning(f"Unauthorized access attempt with key: {x_api_key}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key",
        )

@app.get("/")
def health_check():
    logger.info("Health check requested") # Log kiểm tra sức khoẻ
    return {"status": "running", "ai_ready": rag_service.is_ready}

@app.post("/api/chat", response_model=ChatResponse, dependencies=[Depends(verify_api_key)])
async def chat_endpoint(request: ChatRequest):
    if not request.text:
        logger.warning("Client gửi request rỗng") # Log cảnh báo
        raise HTTPException(status_code=400, detail="Nội dung chat không được để trống")
    
    # Log câu hỏi người dùng
    logger.info(f"Nhận câu hỏi: {request.text}")
    logger.info(f"Kèm theo history: {len(request.history)} messages")
    
    try:
        result = await rag_service.process_question(
            user_question=request.text,
            history=request.history
        )
        
        # Log kết quả trả về (chỉ log đoạn đầu cho đỡ dài)
        answer_preview = (result["answer"][:50] + '..') if result["answer"] else "No answer"
        logger.info(f"Trả lời: {answer_preview}")
        
        return ChatResponse(
            answer=result["answer"],
            debug_info=result.get("debug_info")
        )
    except Exception as e:
        # Quan trọng: Ghi log lỗi đầy đủ (traceback) ra file
        logger.error(f"Lỗi xử lý tại endpoint chat: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))