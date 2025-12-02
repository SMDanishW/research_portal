# Research Portal

Research Portal is a comprehensive web application designed to streamline document analysis and knowledge retrieval. It serves as a centralized hub where users can upload complex documents and interact with them using natural language queries.

## Powered by RAG-Anything

This project utilizes **RAG-Anything** as its core engine. RAG-Anything is a versatile Retrieval-Augmented Generation framework that handles the heavy lifting of document processing, embedding generation, and semantic search. By integrating RAG-Anything, the Research Portal ensures accurate, context-aware responses derived directly from your uploaded materials.

## Key Features

- **Interactive Chat Interface**: Engage in a conversational dialogue with your documents. Ask questions, request summaries, and extract key insights.
- **Multi-Modal Capabilities**: The system goes beyond simple text retrieval. It can identify and present relevant images and tables from within your documents, providing a richer context.
- **Source Citations**: Trust but verify. All generated answers come with citations, allowing you to trace information back to its exact location in the source document.
- **Modern & Responsive UI**: Built with **Next.js** and **Tailwind CSS**, the frontend offers a clean, intuitive, and responsive experience across devices.
- **High-Performance Backend**: The backend is powered by **FastAPI**, ensuring fast response times and scalable architecture.
- **Flexible LLM Integration**: Supports both local LLMs (via Ollama) for privacy and offline use, as well as cloud-based models (like OpenAI) for maximum performance.

## Technology Stack

- **Frontend**: Next.js, React, Tailwind CSS, Lucide React
- **Backend**: FastAPI, Python
- **Core Engine**: RAG-Anything

## Getting Started

### Prerequisites

- Python 3.10+
- Node.js 18+
- [Ollama](https://ollama.com/) (if using local models)

### Installation

1.  **Environment Setup**
    Ensure you have a Python virtual environment set up.
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```

2.  **Install Dependencies**
    Install the required Python packages for the RAG engine and backend.
    ```bash
    pip install -r requirements.txt
    ```

3.  **Frontend Setup**
    Navigate to the frontend directory and install Node.js dependencies.
    ```bash
    cd web_ui/frontend
    npm install
    ```

### Running the Application

1.  **Start the Backend Server**
    From the root directory (`RAG-Anything`), run the FastAPI server:
    ```bash
    python -m web_ui.backend.main
    # OR
    uvicorn web_ui.backend.main:app --reload
    ```

2.  **Start the Frontend Development Server**
    In a separate terminal, navigate to `web_ui/frontend` and start the Next.js app:
    ```bash
    cd web_ui/frontend
    npm run dev
    ```

3.  **Access the Portal**
    Open your browser and go to `http://localhost:3000` to start using the Research Portal.
