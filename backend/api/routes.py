from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import Optional
import uuid

from backend.agent.core import get_agent_response
from backend.agent.memory import clear_session_memory
from backend.knowledge.loader import ingest_document

router = APIRouter()


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    system_prompt: Optional[str] = None


class ChatResponse(BaseModel):
    reply: str
    session_id: str


@router.post("/chat", response_model=ChatResponse, tags=["Agent"])
async def chat(request: ChatRequest):
    """
    Send a message to the AI agent.
    Pass session_id from a previous response to maintain conversation history.
    """
    session_id = request.session_id or str(uuid.uuid4())
    try:
        reply = await get_agent_response(
            message=request.message,
            session_id=session_id,
            system_prompt=request.system_prompt,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return ChatResponse(reply=reply, session_id=session_id)


@router.post("/knowledge/upload", tags=["Knowledge Base"])
async def upload_document(
    file: UploadFile = File(...),
    collection_name: str = "default",
):
    """
    Upload a PDF, DOCX, or TXT file to the agent's knowledge base.
    """
    allowed_types = [
        "application/pdf",
        "text/plain",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. Use PDF, DOCX, or TXT.",
        )
    contents = await file.read()
    try:
        chunks_added = await ingest_document(
            file_bytes=contents,
            filename=file.filename,
            content_type=file.content_type,
            collection_name=collection_name,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {
        "message": "Document ingested successfully.",
        "filename": file.filename,
        "chunks_added": chunks_added,
        "collection": collection_name,
    }


@router.delete("/chat/{session_id}", tags=["Agent"])
async def clear_memory(session_id: str):
    """Clear conversation history for a session."""
    clear_session_memory(session_id)
    return {"message": f"Memory cleared for session: {session_id}"}


@router.get("/health", tags=["System"])
async def health_check():
    return {"status": "healthy", "service": "AI Agent API"}
