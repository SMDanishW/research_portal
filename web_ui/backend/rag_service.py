import os
import sys
import shutil
import contextlib
from pathlib import Path
from typing import List, Dict, Any, Optional
from groq import Groq
import asyncio

# Try to import torch (only needed for local model loading)
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

# Add root directory to path to import raganything
sys.path.append(str(Path(__file__).parent.parent.parent))

from raganything import RAGAnything, RAGAnythingConfig
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc

# Global instance
rag_engine: Optional[RAGAnything] = None

# Global model and tokenizer for local inference
_local_model = None
_local_tokenizer = None
_local_model_path = None

def get_rag_engine():
    global rag_engine
    return rag_engine

def load_local_model(model_path: str):
    """Load Hugging Face model and tokenizer from local path."""
    global _local_model, _local_tokenizer, _local_model_path
    
    if _local_model is not None and _local_model_path == model_path:
        print(f"✅ Using already loaded model from {model_path}")
        return _local_model, _local_tokenizer
    
    try:
        print(f"🔄 Loading model from local path: {model_path}")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        # Load tokenizer
        _local_tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True
        )
        
        # Load model
        if not TORCH_AVAILABLE:
            raise ImportError("torch is required for local model loading. Install with: pip install torch")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"📱 Using device: {device}")
        
        load_kwargs = {
            "trust_remote_code": True,
            "local_files_only": True,
        }
        
        if device == "cuda":
            load_kwargs["torch_dtype"] = torch.float16
            load_kwargs["device_map"] = "auto"
        else:
            load_kwargs["torch_dtype"] = torch.float32
        
        _local_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            **load_kwargs
        )
        
        if device == "cpu":
            _local_model = _local_model.to(device)
        
        _local_model.eval()
        _local_model_path = model_path
        print(f"✅ Model loaded successfully from {model_path}")
        return _local_model, _local_tokenizer
        
    except Exception as e:
        print(f"❌ Error loading local model: {str(e)}")
        print("💡 Make sure transformers is installed: pip install transformers torch")
        raise

async def local_model_complete(
    prompt: str,
    system_prompt: Optional[str] = None,
    history_messages: List[Dict] = [],
    model_path: str = None,
    max_tokens: int = 2048,
    temperature: float = 0.7,
    **kwargs
) -> str:
    """Generate completion using local Hugging Face model."""
    global _local_model, _local_tokenizer
    
    if model_path:
        _local_model, _local_tokenizer = load_local_model(model_path)
    
    if _local_model is None or _local_tokenizer is None:
        raise ValueError("Model not loaded. Please provide model_path or load model first.")
    
    # Format messages for Phi-3 chat template
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    
    # Add history messages
    messages.extend(history_messages)
    
    # Add current prompt
    messages.append({"role": "user", "content": prompt})
    
    # Apply chat template
    try:
        # Run tokenization and generation in executor to avoid blocking
        loop = asyncio.get_event_loop()
        
        def generate():
            # Format for Phi-3 chat template
            formatted_prompt = _local_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Tokenize
            inputs = _local_tokenizer(formatted_prompt, return_tensors="pt")
            if TORCH_AVAILABLE and torch.cuda.is_available():
                inputs = {k: v.to(_local_model.device) for k, v in inputs.items()}
            
            # Generate
            no_grad_context = torch.no_grad() if TORCH_AVAILABLE else contextlib.nullcontext()
            with no_grad_context:
                outputs = _local_model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    pad_token_id=_local_tokenizer.eos_token_id,
                    eos_token_id=_local_tokenizer.eos_token_id,
                )
            
            # Decode response
            generated_text = _local_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the assistant's response (remove the prompt)
            if formatted_prompt in generated_text:
                response = generated_text[len(formatted_prompt):].strip()
            else:
                # Fallback: try to extract after last user message
                response = generated_text.strip()
            
            return response
        
        # Run in thread pool to avoid blocking
        response = await loop.run_in_executor(None, generate)
        return response
        
    except Exception as e:
        print(f"❌ Error during generation: {str(e)}")
        raise

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
    lmstudio_base_url = os.environ.get("LMSTUDIO_BASE_URL")
    local_model_path = os.environ.get("LOCAL_MODEL_PATH")
    
    # Check if we should use direct local model loading
    if local_model_path:
        # Resolve relative path from project root
        if not os.path.isabs(local_model_path):
            project_root = Path(__file__).parent.parent.parent.parent
            local_model_path = str(project_root / local_model_path)
        
        if os.path.exists(local_model_path):
            print(f"🚀 Using Local Hugging Face Model from: {local_model_path}")
            # Load model once
            load_local_model(local_model_path)
            use_local_model = True
        else:
            print(f"⚠️ Local model path not found: {local_model_path}")
            print("⚠️ Falling back to API-based LLM.")
            use_local_model = False
    else:
        use_local_model = False
    
    if use_local_model:
        # Use local model directly
        llm_model_name = local_model_path
        vision_model_name = local_model_path  # Same model for vision
        api_key = None
        base_url = None
        
        # For embeddings, still use Ollama or OpenAI
        embedding_api_key = "ollama"
        embedding_base_url = os.environ.get("LOCAL_LLM_BASE_URL", "http://localhost:11434/v1")
        embedding_model_name = os.environ.get("LOCAL_EMBEDDING_MODEL", "nomic-embed-text")
        embedding_dim = 768
        
    elif groq_api_key:
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
        
    elif lmstudio_base_url:
        print("🚀 Using LM Studio for Local LLM (Phi-3).")
        api_key = os.environ.get("LMSTUDIO_API_KEY", "lm-studio")
        base_url = lmstudio_base_url
        llm_model_name = os.environ.get("LOCAL_LLM_MODEL", "microsoft/Phi-3-mini-128k-instruct")
        vision_model_name = os.environ.get("LOCAL_VISION_MODEL", "microsoft/Phi-3-mini-128k-instruct")
        
        embedding_api_key = api_key
        embedding_base_url = base_url
        embedding_model_name = os.environ.get("LOCAL_EMBEDDING_MODEL", "nomic-embed-text")
        embedding_dim = 768
        
    else:
        print("⚠️ No API Key found. Defaulting to Local LLM (Ollama) with Phi-3.")
        api_key = "ollama"
        base_url = os.environ.get("LOCAL_LLM_BASE_URL", "http://localhost:11434/v1")
        llm_model_name = os.environ.get("LOCAL_LLM_MODEL", "microsoft/Phi-3-mini-128k-instruct")
        vision_model_name = os.environ.get("LOCAL_VISION_MODEL", "microsoft/Phi-3-mini-128k-instruct")
        
        embedding_api_key = "ollama"
        embedding_base_url = base_url
        embedding_model_name = os.environ.get("LOCAL_EMBEDDING_MODEL", "nomic-embed-text")
        embedding_dim = 768

    # Define LLM functions
    if use_local_model:
        async def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
            return await local_model_complete(
                prompt=prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                model_path=local_model_path,
                max_tokens=kwargs.get("max_tokens", 2048),
                temperature=kwargs.get("temperature", 0.7),
                **kwargs
            )
    else:
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
        if use_local_model:
            # For local model, vision is not supported, fall back to text-only
            return await llm_model_func(prompt, system_prompt, history_messages, **kwargs)
        
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
