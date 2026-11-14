# Vaani --- Voice-Enabled Customer Support Chatbot

Vaani is a bilingual (English and Hinglish) customer-support chatbot
built for **VoltKart**, a sample Indian electronics e-commerce store.
Customers can ask questions by text or voice, get answers grounded in
company policies and FAQs, and check order information through a
multi-step verification flow.

It addresses a common support problem: customers need quick, consistent
help without searching through long policy documents or waiting for an
agent. Vaani combines conversational AI with policy-grounded retrieval
and controlled access to order details.

## Key Features

-   **Natural customer conversations:** Allows customers to interact through text or 
    voice and receive responses in English or Hinglish.
-   **Instant policy assistance:** Helps customers understand return, refund, shipping, 
    cancellation, warranty, and installation policies without manually searching through documents.
-   **Reliable, evidence-based answers:** Provides responses supported by company policies and FAQs, 
    includes source citations, and avoids answering when sufficient evidence is unavailable.
-   **Secure order tracking:** Guides customers through order ID and identity verification before 
    sharing order status, delivery, installation, or refund information.
-   **Context-aware conversations:** Understands follow-up questions by retaining relevant 
    conversation history, so customers don't have to repeat information.
-   **Intelligent request handling:** Recognizes greetings, complaints, product and policy questions, 
    order enquiries, and requests for human assistance, while handling unrelated queries appropriately.
-   **Uninterrupted conversation flow:** Allows customers to change topics or exit an 
    order-verification process without getting trapped in repetitive prompts.
-   **Measurable response quality:** Evaluates answer accuracy, retrieval quality, conversation 
    handling, speech recognition, and response latency to identify areas for improvement.
    
## Tech Stack

  |      Area                |    Technologies     |
  | ------------------------ | ------------------- | 
  | API and UI                | FastAPI, Uvicorn, Streamlit |
  | LLM and query rewriting    | Groq API |
  | RAG                 |       Qdrant, BM25, LangChain, `BAAI/bge-m3`|
  | Reranking                 | `BAAI/bge-reranker-v2-m3`|
  | Conversation memory        | LangGraph|
  | Order and workflow state   | SQLite|
  | Speech                     | Faster-Whisper, Google Gemini TTS|
  | Evaluation and quality     | DeepEval, Pytest|
  | Packaging and CI           | Docker, Docker Compose, GitHub Actions, Ruff|

## Evaluation Results

The following results are from the current evaluation runs. Scores are
evaluator outputs on the tested datasets, not a guarantee of production
performance.

### RAG quality

**12 bilingual test cases**, evaluated with DeepEval.

| Metric                  |      Score    |
| ----------------------  | ------------  |
| Faithfulness            | **1.0000**    |
| Answer relevancy        | **0.9722**    |
| Contextual relevancy    |  **0.5273**   |
| Contextual precision    |  **0.9861**   |
| Contextual recall       |  **1.0000**   |


### Voice pipeline

**6 voice samples** were included in the benchmark.

|  Metric                         |     Result       |
|  -----------------------------  | ---------------  |
|  Mean Word Error Rate (WER)     |     **47.11%**   |
|  Average STT latency            |   **1036.37 ms**   |
|  Average support/RAG latency    |   **6853.45 ms**   |
|  Average TTS latency            |  **9091.13 ms**   |
|  Mean end-to-end latency        |  **18309.26 ms**  |
|  P50 end-to-end latency         |  **19710.12 ms**   |
|  P95 end-to-end latency         |  **22612.32 ms**  |

### Multi-turn conversation

**5 conversation scenarios**, covering follow-ups, memory, order
verification, topic switching, and role adherence.

| Metric                      |   Score      |
| --------------------------- | ------------ |
| Conversation relevancy      |   **0.76**   |
| Knowledge retention         |    **0.68**  |
| Conversation completeness   |   **0.6067** |
| Role adherence              |   **0.77**   |

**Evaluation note:** These results highlight opportunities to improve speech 
recognition accuracy, TTS reliability, and response latency. The contextual-relevancy and conversation-completeness scores also identify areas for improvement. 

## Run Locally

### Requirements

-   Python 3.11+
-   FFmpeg
-   A Groq API key
-   A Google Gemini API key for TTS
-   Sufficient disk space and memory for local embedding, reranker, and
    Whisper models

### 1. Clone the repository

``` bash
git clone https://github.com/<your-username>/Vaani.git
cd Vaani
```

### 2. Create an environment and install dependencies

``` bash
python3.11 -m venv vaani
source vaani/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Install FFmpeg if it is not already available:

``` bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt-get update && sudo apt-get install -y ffmpeg
```

### 3. Configure environment variables

Create a `.env` file in the project root:

``` env
GROQ_API_KEY=your_groq_api_key
GOOGLE_API_KEY=your_google_gemini_api_key

ENABLE_STT=true
ENABLE_TTS=true

WHISPER_MODEL_SIZE=base
WHISPER_DEVICE=cpu
WHISPER_COMPUTE_TYPE=int8

GOOGLE_TTS_MODEL=gemini-3.1-flash-tts-preview
GOOGLE_TTS_VOICE=Kore


WHISPER_MODEL_SIZE=base
WHISPER_DEVICE=cpu
WHISPER_COMPUTE_TYPE=int8

VAANI_EVAL_MODEL="openai/gpt-oss-20b"
```

Keep `.env` private and do not commit API keys.

### 4. Build the knowledge-base indexes

``` bash
python backend/rag/indexer.py
```

This indexes the policy documents and FAQ data for retrieval.

### 5. Start the application

Open two terminals with the virtual environment activated.

**Terminal 1 --- API:**

``` bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

**Terminal 2 --- Streamlit interface:**

``` bash
streamlit run frontend/streamlit_app.py
```

Open:

-   App: http://localhost:8501
-   API documentation: http://localhost:8000/docs
-   Readiness check: http://localhost:8000/api/v1/health/ready

## Run Tests and Evaluations

``` bash
# Unit and API tests
pytest tests/ -v

# RAG quality
python evaluation/evaluate_rag.py

# Voice quality and latency
python evaluation/evaluate_voice.py

# Multi-turn conversation
python evaluation/evaluate_conversation.py
```

Evaluation scripts may call external LLM or speech APIs and can consume
API quota. Configure valid API keys before running them.

## Project Status

Vaani demonstrates an end-to-end approach to building an intelligent, bilingual customer-support assistant, combining conversational AI, hybrid RAG, secure order verification, and voice interaction. The current benchmark shows strong faithfulness and retrieval recall, while contextual relevance, multi-turn task completion, voice transcription quality, latency, and API quota handling remain areas for improvement.
