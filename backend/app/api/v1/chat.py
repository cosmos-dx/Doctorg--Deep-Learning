"""
Chat API endpoints with SSE streaming support.
Multi-agent medical consultation system.
"""

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
import asyncio
import json
import logging
from typing import Optional
import uuid

from app.db.database import get_db
from app.db.models import User
from app.models.schemas import ChatRequest, StreamChunk
from app.core.security import get_current_user
from app.core.constants import ErrorMessages
from app.core.errors import EmergencyDetectedError, AgentError
from app.ml.rag.memory_engine import create_memory_engine
from app.services.openai_service import create_openai_service
from app.agents.orchestrator import AgentOrchestrator
from app.agents.rag_agent import RAGAgent

router = APIRouter()
logger = logging.getLogger(__name__)

_memory_engine = None
_orchestrator = None


def _get_orchestrator() -> AgentOrchestrator:
    """Lazy-init orchestrator so env vars are loaded before construction."""
    global _memory_engine, _orchestrator
    if _orchestrator is None:
        _memory_engine = create_memory_engine()
        openai_service = create_openai_service()
        rag_agent = RAGAgent(
            openai_service=openai_service,
            memory_engine=_memory_engine
        )
        _orchestrator = AgentOrchestrator(
            openai_service=openai_service,
            rag_agent=rag_agent,
            memory_engine=_memory_engine
        )
    return _orchestrator


def _get_memory_engine():
    _get_orchestrator()
    return _memory_engine


@router.post("/stream")
async def chat_stream(
    request: ChatRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Stream medical consultation through multi-agent system using SSE.
    """
    async def event_generator():
        try:
            session_id = request.session_id or str(uuid.uuid4())
            
            orchestrator = _get_orchestrator()
            memory_engine = _get_memory_engine()
            
            conversation_history = []
            if request.session_id:
                history_records = await memory_engine.get_user_session_history(
                    user_id=str(current_user.id),
                    db=db,
                    limit=5
                )
                
                for record in history_records:
                    if record.get("session_id") == request.session_id:
                        conversation_history.append({
                            "role": "assistant",
                            "content": str(record.get("diagnosis", {}))
                        })
            
            user_message = request.message if hasattr(request, 'message') else " ".join(request.symptoms)
            
            # Fetch Health Profile and Report History
            from app.db.models import UserHealthProfile, MedicalReport
            
            profile = db.query(UserHealthProfile).filter(UserHealthProfile.user_id == str(current_user.id)).first()
            recent_reports = db.query(MedicalReport).filter(
                MedicalReport.user_id == str(current_user.id),
                MedicalReport.is_medical == True
            ).order_by(MedicalReport.report_date.desc().nullslast()).limit(3).all()
            
            lab_history = []
            for r in recent_reports:
                for b in r.biomarkers:
                    lab_history.append(f"{b.name}: {b.value} {b.unit} ({b.status})")
            
            metadata = {
                "health_profile": {
                    "age": profile.age, "gender": profile.gender, "lifestyle_notes": profile.lifestyle_notes,
                    "chronic_conditions": profile.chronic_conditions, "allergies": profile.allergies
                } if profile else None,
                "lab_history": lab_history if lab_history else None
            }
            
            async for chunk in orchestrator.process_stream(
                user_message=user_message,
                symptoms=request.symptoms,
                user_id=str(current_user.id),
                session_id=session_id,
                conversation_history=conversation_history,
                db=db,
                metadata=metadata
            ):
                chunk_json = json.dumps(chunk)
                yield f"data: {chunk_json}\n\n"
                
                await asyncio.sleep(0.01)
            
            final_chunk = {
                "type": "done",
                "session_id": session_id,
                "content": "Consultation complete"
            }
            yield f"data: {json.dumps(final_chunk)}\n\n"
            
        except EmergencyDetectedError as e:
            logger.warning(f"Emergency detected: {e.message}")
            emergency_chunk = {
                "type": "emergency",
                "content": e.message,
                "symptoms": e.details.get("detected_symptoms", [])
            }
            yield f"data: {json.dumps(emergency_chunk)}\n\n"
            
        except AgentError as e:
            logger.error(f"Agent error: {e.message}")
            error_chunk = {
                "type": "error",
                "content": e.message,
                "error_code": e.error_code
            }
            yield f"data: {json.dumps(error_chunk)}\n\n"
            
        except Exception as e:
            logger.error(f"Unexpected stream error: {e}")
            error_chunk = {
                "type": "error",
                "content": ErrorMessages.LLM_GENERATION_ERROR
            }
            yield f"data: {json.dumps(error_chunk)}\n\n"
    
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
    Generate medical consultation (non-streaming) through multi-agent system.
    """
    try:
        orchestrator = _get_orchestrator()
        
        session_id = request.session_id or str(uuid.uuid4())
        
        conversation_history = []
        user_message = request.message if hasattr(request, 'message') else " ".join(request.symptoms)
        
        from app.db.models import UserHealthProfile, MedicalReport
        profile = db.query(UserHealthProfile).filter(UserHealthProfile.user_id == str(current_user.id)).first()
        recent_reports = db.query(MedicalReport).filter(
            MedicalReport.user_id == str(current_user.id),
            MedicalReport.is_medical == True
        ).order_by(MedicalReport.report_date.desc().nullslast()).limit(3).all()
        
        lab_history = []
        for r in recent_reports:
            for b in r.biomarkers:
                lab_history.append(f"{b.name}: {b.value} {b.unit} ({b.status})")
                
        metadata = {
            "health_profile": {
                "age": profile.age, "gender": profile.gender, "lifestyle_notes": profile.lifestyle_notes,
                "chronic_conditions": profile.chronic_conditions, "allergies": profile.allergies
            } if profile else None,
            "lab_history": lab_history if lab_history else None
        }
        
        responses = await orchestrator.process(
            user_message=user_message,
            symptoms=request.symptoms,
            user_id=str(current_user.id),
            session_id=session_id,
            conversation_history=conversation_history,
            db=db,
            metadata=metadata
        )
        
        final_response = responses.get("final")
        
        return {
            "success": True,
            "session_id": session_id,
            "response": final_response.content if final_response else "",
            "metadata": {
                "agents_used": list(responses.keys()),
                "urgency_level": responses.get("triage", type('obj', (object,), {
                    'metadata': {}
                })()).metadata.get("urgency_level", "unknown")
            }
        }
        
    except EmergencyDetectedError as e:
        return {
            "success": False,
            "emergency": True,
            "message": e.message,
            "symptoms": e.details.get("detected_symptoms", [])
        }
        
    except AgentError as e:
        logger.error(f"Agent error in predict: {e.message}")
        raise HTTPException(
            status_code=500,
            detail={
                "message": e.message,
                "error_code": e.error_code
            }
        )
        
    except Exception as e:
        logger.error(f"Unexpected error in predict: {e}")
        raise HTTPException(
            status_code=500,
            detail={"message": ErrorMessages.LLM_GENERATION_ERROR}
        )
