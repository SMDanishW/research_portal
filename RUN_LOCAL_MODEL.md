# How to Run with Local Phi-3 Model

This guide explains how to run the Research Portal application using your local Phi-3 model files.

## Prerequisites

1. Python 3.10+ installed
2. Phi-3 model files in `../llms/Phi-3-mini-128k-instruct/` folder
3. (Optional) NVIDIA GPU with CUDA support for faster inference

## Step 1: Install Dependencies

Navigate to the `RAG-Anything` directory and install required packages:

```bash
# Install main requirements
pip install -r requirements.txt

# Install backend requirements  
pip install -r web_ui/backend/requirements.txt

# Install transformers and torch for local model loading
pip install transformers torch

# Optional: For GPU support (if you have NVIDIA GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Step 2: Set Environment Variable

You need to tell the application where your local model is located. Choose one method:

### Method 1: Set Environment Variable (Windows CMD)
```cmd
cd D:\DS_Company\research_portal\RAG-Anything
set LOCAL_MODEL_PATH=..\llms\Phi-3-mini-128k-instruct
```

### Method 2: Set Environment Variable (Windows PowerShell)
```powershell
cd D:\DS_Company\research_portal\RAG-Anything
$env:LOCAL_MODEL_PATH="..\llms\Phi-3-mini-128k-instruct"
```

### Method 3: Set Environment Variable (Linux/Mac)
```bash
cd /path/to/RAG-Anything
export LOCAL_MODEL_PATH="../llms/Phi-3-mini-128k-instruct"
```

### Method 4: Use Absolute Path
```cmd
set LOCAL_MODEL_PATH=D:\DS_Company\research_portal\llms\Phi-3-mini-128k-instruct
```

## Step 3: Run the Backend Server

From the `RAG-Anything` directory, run:

```bash
# Option 1: Using Python module
python -m web_ui.backend.main

# Option 2: Using uvicorn directly
uvicorn web_ui.backend.main:app --reload --host 0.0.0.0 --port 8000
```

You should see output like:
```
🔄 Loading model from local path: D:\DS_Company\research_portal\llms\Phi-3-mini-128k-instruct
📱 Using device: cuda  (or "cpu" if no GPU)
✅ Model loaded successfully from ...
RAG Engine Initialized (Model: ...)
INFO:     Uvicorn running on http://0.0.0.0:8000
```

## Step 4: (Optional) Run the Frontend

If you want to use the web interface, open a **new terminal** and run:

```bash
cd RAG-Anything/web_ui/frontend
npm install  # Only needed first time
npm run dev
```

Then open your browser to `http://localhost:3000`

## Step 5: Test the API

You can test the backend API directly:

```bash
# Test health endpoint
curl http://localhost:8000/api/health

# Test query endpoint (if you have documents processed)
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d "{\"query\": \"What is this document about?\"}"
```

## Troubleshooting

### Model Not Found Error
- Check that `LOCAL_MODEL_PATH` is set correctly
- Verify the path exists: `ls ../llms/Phi-3-mini-128k-instruct/` (or `dir` on Windows)
- Try using an absolute path instead of relative path

### Import Errors (transformers/torch)
- Make sure you installed: `pip install transformers torch`
- If using GPU, install CUDA version of PyTorch

### Out of Memory Error
- The model is large (~7GB). Make sure you have enough RAM/VRAM
- If on CPU, it will be slower but should work
- Consider using a smaller model or reducing `max_tokens` parameter

### Model Loading Takes Long Time
- First load takes time to load the model into memory
- Subsequent requests will be faster
- Using GPU significantly speeds up inference

## Configuration Options

You can customize behavior with these environment variables:

- `LOCAL_MODEL_PATH` - Path to your local model folder (required)
- `LOCAL_LLM_BASE_URL` - Base URL for embeddings (default: `http://localhost:11434/v1`)
- `LOCAL_EMBEDDING_MODEL` - Embedding model name (default: `nomic-embed-text`)

## Notes

- The model loads once when the server starts and stays in memory
- First request may be slower as the model warms up
- For embeddings, the system still uses Ollama (or you can configure OpenAI)
- Vision/image processing falls back to text-only mode with local model

