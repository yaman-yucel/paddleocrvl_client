import logging
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, UploadFile, HTTPException
from paddleocr import PaddleOCRVL

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

pipeline: PaddleOCRVL | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize OCR pipeline on startup."""
    global pipeline
    logger.info("Initializing PaddleOCR pipeline...")
    pipeline = PaddleOCRVL(
        vl_rec_backend="vllm-server",
        vl_rec_server_url="http://localhost:8118/v1",
        vl_rec_api_model_name="PaddleOCR-VL-1.5-0.9B",
    )
    logger.info("PaddleOCR pipeline ready")
    yield
    logger.info("Shutting down...")


app = FastAPI(
    title="PaddleOCR API",
    description="OCR API using PaddleOCR-VL",
    lifespan=lifespan,
)


ALLOWED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}


@app.post("/ocr")
async def process_file(file: UploadFile = File(...)) -> dict[str, Any]:
    """
    Process a single image or PDF file and return OCR results.
    
    Supported formats: PDF, PNG, JPG, JPEG, BMP, TIFF, WEBP
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="OCR pipeline not initialized")

    # Validate file extension
    file_ext = Path(file.filename or "").suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    # Save uploaded file to temp location
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        logger.info("Processing file: %s", file.filename)
        
        output = pipeline.predict(input=str(tmp_path))
        pages_res = list(output)
        restructured = pipeline.restructure_pages(pages_res)

        # Collect results from all pages
        results: list[dict[str, Any]] = []
        for page in restructured:
            page_data = {
                "markdown": page.to_markdown() if hasattr(page, "to_markdown") else None,
                "json": page.to_json() if hasattr(page, "to_json") else None,
            }
            results.append(page_data)

        logger.info("Completed processing: %s (%d pages)", file.filename, len(results))
        
        return {
            "filename": file.filename,
            "pages": len(results),
            "results": results,
        }

    except Exception as e:
        logger.exception("Error processing file: %s", file.filename)
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Clean up temp file
        tmp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)