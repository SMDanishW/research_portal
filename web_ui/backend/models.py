from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class QueryRequest(BaseModel):
    query: str
    mode: str = "hybrid"
    vlm_enhanced: bool = True

class Citation(BaseModel):
    source_id: str
    content_type: str  # "text", "image", "table"
    content: str
    file_path: str
    page_idx: Optional[int] = None
    score: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

class QueryResponse(BaseModel):
    answer: str
    citations: List[Citation]

class UploadResponse(BaseModel):
    filename: str
    status: str
    doc_id: str
