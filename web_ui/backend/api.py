from fastapi import APIRouter, UploadFile, File, HTTPException
from .models import QueryRequest, QueryResponse, UploadResponse, Citation
from .rag_service import process_document, query_rag
import shutil
import os
import uuid

router = APIRouter()

UPLOAD_DIR = "./uploads"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

@router.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    try:
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Process the file
        doc_id = str(uuid.uuid4()) # Placeholder ID, RAG generates its own
        await process_document(file_path)
        
        return UploadResponse(
            filename=file.filename,
            status="processed",
            doc_id=doc_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    try:
        answer, citations_data = await query_rag(
            request.query, 
            mode=request.mode, 
            vlm_enhanced=request.vlm_enhanced
        )
        
        # Convert citations_data to Citation models (placeholder logic)
        citations = []
        for c in citations_data:
            citations.append(Citation(**c))
            
        return QueryResponse(answer=answer, citations=citations)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
