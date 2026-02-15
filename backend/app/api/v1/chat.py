"""
Chat API endpoints with SSE streaming support.
"""

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
import asyncio
import json
import logging

from app.db.database import get_db
from app.db.models import User
from app.models.schemas import ChatRequest, StreamChunk
from app.core.security import get_current_user
from app.core.constants import ErrorMessages
from app.ml.llm.inference import create_llm_service
from app.ml.rag.memory_engine import create_memory_engine
from app.services.subscription import create_subscription_service

router = APIRouter()
logger = logging.getLogger(__name__)

llm_service = create_llm_service()
memory_engine = create_memory_engine()
subscription_service = create_subscription_service(llm_service, memory_engine)


@router.post("/stream")
async def chat_stream(
    request: ChatRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Stream medical response in real-time using SSE.
    """
    async def event_generator():
        try:
            if not await subscription_service.check_access(current_user, "session"):
                error_chunk = StreamChunk(
                    error=ErrorMessages.SESSION_LIMIT_REACHED,
                    done=True
                )
                yield f"data: {error_chunk.json()}\n\n"
                return
            
            has_memory = await subscription_service.check_access(current_user, "memory")
            
            history = None
            if has_memory:
                relevant_history = await memory_engine.retrieve_relevant_history(
                    current_user.id,
                    " ".join(request.symptoms),
                    db
                )
                
                if relevant_history:
                    history = memory_engine.format_history_for_context(relevant_history)
            
            full_response = ""
            
            async for chunk in llm_service.generate_stream(request.symptoms, history):
                full_response += chunk
                
                stream_chunk = StreamChunk(content=chunk, done=False)
                yield f"data: {stream_chunk.json()}\n\n"
                
                await asyncio.sleep(0.01)
            
            medical_response = await llm_service.generate_medical_response(
                request.symptoms,
                history
            )
            
            session_id = await memory_engine.store_session(
                user_id=current_user.id,
                symptoms=request.symptoms,
                diagnosis=medical_response.dict(),
                db=db
            )
            
            await subscription_service.increment_session_count(current_user, db)
            
            remaining = await subscription_service.get_remaining_sessions(current_user)
            
            final_chunk = StreamChunk(
                done=True,
                structured_data=medical_response
            )
            yield f"data: {final_chunk.json()}\n\n"
            
        except Exception as e:
            logger.error(f"Stream error: {e}")
            error_chunk = StreamChunk(
                error=ErrorMessages.LLM_GENERATION_ERROR,
                done=True
            )
            yield f"data: {error_chunk.json()}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@router.post("/predict")
async def predict(
    request: ChatRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Generate medical response (non-streaming).
    """
    result = await subscription_service.generate_response(
        current_user,
        request.symptoms,
        db
    )
    
    if "error" in result:
        raise HTTPException(
            status_code=403,
            detail=result["error"]
        )
    
    return result
