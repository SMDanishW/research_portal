import os
import sys
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
from groq import Groq
import asyncio

# Add root directory to path to import raganything
sys.path.append(str(Path(__file__).parent.parent.parent))

from raganything import RAGAnything, RAGAnythingConfig
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc

# Global instance
rag_engine: Optional[RAGAnything] = None

def get_rag_engine():
    global rag_engine
    return rag_engine

async def initialize_rag():
    global rag_engine
    
    # Configuration
    work_dir = "./rag_storage"
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)
        
    config = RAGAnythingConfig(
        working_dir=work_dir,
        enable_image_processing=True,
        enable_table_processing=True,
        enable_equation_processing=True,
    )

    # API Key setup
    groq_api_key = os.environ.get("GROQ_API_KEY")
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    
    if groq_api_key:
        print("🚀 Using Groq Cloud for LLM.")
        api_key = groq_api_key
        base_url = "https://api.groq.com/openai/v1"
        llm_model_name = "llama-3.3-70b-versatile" # Valid Groq model
        vision_model_name = "llama-3.2-90b-vision-preview" # Valid Groq Vision model
        
        # Groq doesn't support embeddings, so we MUST use Local (Ollama) or OpenAI for embeddings
        # We'll default to Local (Ollama) for embeddings to keep it free
        print("⚠️ Groq doesn't support embeddings. Using Local (Ollama) for embeddings.")
        embedding_api_key = "ollama"
        embedding_base_url = "http://localhost:11434/v1"
        embedding_model_name = "nomic-embed-text"
        embedding_dim = 768
        
    elif openai_api_key:
        print("🚀 Using OpenAI for LLM.")
        api_key = openai_api_key
        base_url = os.environ.get("OPENAI_BASE_URL")
        llm_model_name = "gpt-4o-mini"
        vision_model_name = "gpt-4o"
        
        embedding_api_key = openai_api_key
        embedding_base_url = base_url
        embedding_model_name = "text-embedding-3-large"
        embedding_dim = 3072
        
    else:
        print("⚠️ No API Key found. Defaulting to Local LLM (Ollama).")
        api_key = "ollama"
        base_url = os.environ.get("LOCAL_LLM_BASE_URL", "http://localhost:11434/v1")
        llm_model_name = os.environ.get("LOCAL_LLM_MODEL", "llama3")
        vision_model_name = os.environ.get("LOCAL_VISION_MODEL", "llava")
        
        embedding_api_key = "ollama"
        embedding_base_url = base_url
        embedding_model_name = os.environ.get("LOCAL_EMBEDDING_MODEL", "nomic-embed-text")
        embedding_dim = 768

    # Define LLM functions
    async def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
        return await openai_complete_if_cache(
            llm_model_name,
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            api_key=api_key,
            base_url=base_url,
            **kwargs,
        )

    async def vision_model_func(prompt, system_prompt=None, history_messages=[], image_data=None, messages=None, **kwargs):
        if messages:
            return await openai_complete_if_cache(
                vision_model_name,
                "",
                system_prompt=None,
                history_messages=[],
                messages=messages,
                api_key=api_key,
                base_url=base_url,
                **kwargs,
            )
        elif image_data:
            return await openai_complete_if_cache(
                vision_model_name,
                "",
                system_prompt=None,
                history_messages=[],
                messages=[
                    {"role": "system", "content": system_prompt} if system_prompt else None,
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}},
                        ],
                    }
                ],
                api_key=api_key,
                base_url=base_url,
                **kwargs,
            )
        else:
            return await llm_model_func(prompt, system_prompt, history_messages, **kwargs)

    embedding_func = EmbeddingFunc(
        embedding_dim=embedding_dim,
        max_token_size=8192,
        func=lambda texts: openai_embed(
            texts,
            model=embedding_model_name,
            api_key=embedding_api_key,
            base_url=embedding_base_url,
        ),
    )

    # Initialize RAG
    rag_engine = RAGAnything(
        config=config,
        llm_model_func=llm_model_func,
        vision_model_func=vision_model_func,
        embedding_func=embedding_func,
    )
    
    # Ensure initialization
    await rag_engine._ensure_lightrag_initialized()
    print(f"RAG Engine Initialized (Model: {llm_model_name})")

async def process_document(file_path: str):
    global rag_engine
    if not rag_engine:
        await initialize_rag()
    
    await rag_engine.process_document_complete(
        file_path=file_path,
        parse_method="auto"
    )
    return "Processed"

async def query_rag(query_text: str, mode: str = "hybrid", vlm_enhanced: bool = True):
    global rag_engine
    if not rag_engine:
        await initialize_rag()
        
    # For now, we just return the answer. 
    # To get citations, we might need to dig deeper into LightRAG's retrieve method
    # But for the first pass, let's just get the answer.
    
    # Note: To get citations, we would ideally use rag_engine.lightrag.retrieve(...) 
    # and then format the context. 
    # For this MVP, we will rely on the answer string.
    
    answer = await rag_engine.aquery(
        query_text,
        mode=mode,
        vlm_enhanced=vlm_enhanced
    )
    
    return answer, [] # Citations empty for now
