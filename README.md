# TinyLlama Study Assistant

A lightweight, locally running AI chatbot built with **TinyLlama**,
**Streamlit**, **Hugging Face Transformers**, and **PyTorch**. The
project provides a simple chat interface for asking questions, learning
technical concepts, and experimenting with local language-model
inference.

## Overview

The TinyLlama Study Assistant runs a language model on your own machine
instead of relying on a hosted LLM API. It uses Streamlit for the user
interface and Hugging Face Transformers to load and run the model.

## Model

-   **Model:** `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
-   **Model family:** TinyLlama
-   **Parameters:** 1.1 billion
-   **Inference:** Local, using PyTorch and Transformers
-   **Acceleration:** CUDA when a compatible NVIDIA GPU and CUDA-enabled
    PyTorch installation are available; otherwise CPU execution can be
    configured.

## Features

-   Local chatbot interface built with Streamlit
-   TinyLlama chat model integration
-   Conversation history within the active Streamlit session
-   System prompt to guide the assistant's response style
-   Hugging Face model caching to avoid repeated downloads
-   GPU-accelerated inference when CUDA is available
-   Token streaming for progressively displayed responses (when using
    the streaming implementation)

## Tech Stack

-   Python
-   Streamlit
-   PyTorch
-   Hugging Face Transformers
-   Hugging Face Hub

## Project Structure

A simple project layout can look like this:

``` text
ChatBotv1/
├── 1st_cb.py
├── README.md
└── requirements.txt
```

Your actual filenames may differ.

## Setup

### 1. Clone the repository

``` bash
git clone <your-repository-url>
cd ChatBotv1
```

### 2. Create and activate a virtual environment

Windows PowerShell:

``` powershell
python -m venv agenv
.\agenv\Scripts\Activate.ps1
```

### 3. Install dependencies

If you have a `requirements.txt` file:

``` bash
python -m pip install -r requirements.txt
```

Otherwise, install the core packages:

``` bash
python -m pip install streamlit torch transformers huggingface_hub
```

For GPU acceleration, install a PyTorch build compatible with your
installed NVIDIA driver and CUDA support. Follow the official PyTorch
installation selector: https://pytorch.org/get-started/locally/

### 4. Run the application

``` bash
python -m streamlit run 1st_cb.py
```

Streamlit will print a local URL, usually:

``` text
http://localhost:8501
```

Open that address in your browser.

## Model Download and Cache

The first run downloads the TinyLlama model files from Hugging Face.
Later runs can reuse the local cache.

The application can use a dedicated cache directory by setting these
environment variables before importing Hugging Face libraries:

``` python
import os

os.environ["HF_HOME"] = "C:/hf_cache"
os.environ["HF_HUB_CACHE"] = "C:/hf_cache/hub"
```

Keep the cache directory between runs. If you change the cache location,
Hugging Face may download the model again into the new location.

## GPU Check

To check whether PyTorch can access CUDA, run:

``` bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CUDA not available')"
```

If CUDA is unavailable, check that the installed PyTorch build supports
CUDA and that your NVIDIA driver is working.

## Notes and Limitations

-   TinyLlama is a compact model intended for lightweight
    experimentation. Its answers may be less accurate, detailed, or
    consistent than those from larger models.
-   A system prompt can guide the model, but it cannot completely
    prevent hallucinations or guarantee reliable memory.
-   Streamlit resource caching can keep the loaded model in memory
    during the app process. Hugging Face's disk cache separately stores
    downloaded model files.
-   Local inference speed depends on the hardware, model settings,
    prompt length, and generated response length.

## Future Improvements

-   Improve response quality and instruction following
-   Add persistent chat history
-   Add controls for response length and generation settings
-   Measure inference latency and tokens per second
-   Explore retrieval-augmented generation (RAG)
-   Add tool calling and agent workflows

## Acknowledgements

-   [TinyLlama](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)
-   [Hugging Face
    Transformers](https://huggingface.co/docs/transformers)
-   [Streamlit](https://docs.streamlit.io/)
-   [PyTorch](https://pytorch.org/)
