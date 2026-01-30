# PaddleOCR Client

A FastAPI-based OCR service using PaddleOCR-VL-1.5 with vLLM backend for document parsing.

## Prerequisites

- NVIDIA GPU with CUDA support
- Docker with NVIDIA Container Toolkit
- [uv](https://docs.astral.sh/uv/) package manager

## Installation

```bash
# Clone and install dependencies
git clone <repo-url>
cd paddleocrclient
uv sync

# Install Hugging Face CLI and download model
uv tool install huggingface_hub
hf download PaddlePaddle/PaddleOCR-VL-1.5

# Fix cache permissions for Docker
chmod -R a+rX ~/.cache/huggingface
mkdir -p ~/.cache/vllm && chmod 777 ~/.cache/vllm
```

## Usage

### 1. Start the vLLM Server (Docker)

```bash
sudo docker run -it --rm --gpus all --network host \
  -v ~/.cache:/home/paddleocr/.cache \
  -e HF_HOME=/home/paddleocr/.cache/huggingface \
  ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-genai-vllm-server:latest-nvidia-gpu \
  paddleocr genai_server --model_name PaddleOCR-VL-1.5-0.9B --host 0.0.0.0 --port 8118 --backend vllm
```

### 2. Start the FastAPI Server

In a new terminal:

```bash
uv run main.py
```

The API will be available at `http://localhost:8080`

### 3. Process Documents

Using the client script:

```bash
# Place PDF files in ./demo directory
uv run client.py
```

Or via curl:

```bash
curl -X POST http://localhost:8080/ocr \
  -F "file=@document.pdf"
```

## API Endpoints

### POST /ocr

Upload and process a document.

**Supported formats:** PDF, PNG, JPG, JPEG, BMP, TIFF, WEBP

**Response:**
```json
{
  "filename": "document.pdf",
  "pages": 2,
  "results": [
    {
      "markdown": "# Page content...",
      "json": {...}
    }
  ]
}
```
