import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import RedirectResponse
from paddleocr import PaddleOCRVL

from conf import settings
from models import BatchOCRResponse, OCRResponse, PageData

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
        vl_rec_server_url=settings.vllm_server_url,
        vl_rec_api_model_name=settings.vllm_model_name,
    )
    logger.info("PaddleOCR pipeline ready")
    yield
    logger.info("Shutting down...")


app = FastAPI(
    title="PaddleOCR API",
    description="OCR API using PaddleOCR-VL",
    lifespan=lifespan,
)


@app.get("/", include_in_schema=False)
async def root() -> RedirectResponse:
    """Redirect root to API docs."""
    return RedirectResponse(url="/docs")


ALLOWED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}


def _process_with_pipeline(
    file_paths: list[Path],
    tmp_output_dir: Path,
) -> dict[str, PageData]:
    """
    Process files with PaddleOCR pipeline and return results.
    
    Uses save_to_json and save_to_markdown to write to tmp, then reads them back.
    Returns a dict with page name as key, containing json and markdown results.
    """
    if pipeline is None:
        raise RuntimeError("Pipeline not initialized")

    # Process all files at once for efficiency
    input_paths = [str(p) for p in file_paths]
    if len(input_paths) == 1:
        output = pipeline.predict(input=input_paths[0])
    else:
        output = pipeline.predict(input=input_paths)

    pages_res = list(output)
    restructured = pipeline.restructure_pages(pages_res)

    for page in restructured:
        # Save to temp directory
        page.save_to_json(save_path=str(tmp_output_dir))
        page.save_to_markdown(save_path=str(tmp_output_dir))

    # Read back the saved files
    json_files = sorted(tmp_output_dir.glob("*_res.json"))

    # Build result dict with page as key
    results: dict[str, PageData] = {}
    for json_file in json_files:
        page_name = json_file.stem  # e.g., "9.sinif_res"
        # Markdown file doesn't have "_res" suffix
        base_name = page_name.removesuffix("_res")
        md_file = tmp_output_dir / f"{base_name}.md"

        json_data = None
        markdown = None

        if json_file.exists():
            json_data = json.loads(json_file.read_text(encoding="utf-8"))
        if md_file.exists():
            markdown = md_file.read_text(encoding="utf-8")

        results[page_name] = PageData(json=json_data, markdown=markdown)

    return results


@app.post("/ocr", response_model=OCRResponse)
async def process_file(file: UploadFile = File(...)) -> OCRResponse:
    """
    Process a single image or PDF file and return OCR results.
    
    Supported formats: PDF, PNG, JPG, JPEG, BMP, TIFF, WEBP
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="OCR pipeline not initialized")

    # Validate file extension
    #! TODO: only png, jpg, jpeg pdf tested
    file_ext = Path(file.filename or "").suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    with TemporaryDirectory() as tmp_dir:
        tmp_dir_path = Path(tmp_dir)
        tmp_input = tmp_dir_path / "input"
        tmp_output = tmp_dir_path / "output"
        tmp_input.mkdir()
        tmp_output.mkdir()

        # Use original filename to preserve name in output
        safe_name = Path(file.filename or "upload").name
        tmp_path = tmp_input / safe_name

        # Save uploaded file
        content = await file.read()
        tmp_path.write_bytes(content)

        try:
            logger.info("Processing file: %s", file.filename)
            results = _process_with_pipeline([tmp_path], tmp_output)
            logger.info("Completed processing: %s (%d results)", file.filename, len(results))

            return OCRResponse(
                filename=file.filename,
                page_count=len(results),
                pages=results,
            )

        except Exception as e:
            logger.exception("Error processing file: %s", file.filename)
            raise HTTPException(status_code=500, detail=str(e))


@app.post("/ocr/batch", response_model=BatchOCRResponse)
async def process_files(files: list[UploadFile] = File(...)) -> BatchOCRResponse:
    """
    Process multiple images/PDF files at once (more efficient than individual calls).
    
    Supported formats: PDF, PNG, JPG, JPEG, BMP, TIFF, WEBP
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="OCR pipeline not initialized")

    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    # Validate all files first
    for file in files:
        file_ext = Path(file.filename or "").suffix.lower()
        if file_ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type '{file.filename}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
            )

    with TemporaryDirectory() as tmp_dir:
        tmp_dir_path = Path(tmp_dir)
        tmp_input = tmp_dir_path / "input"
        tmp_output = tmp_dir_path / "output"
        tmp_input.mkdir()
        tmp_output.mkdir()

        # Save all uploaded files
        file_paths: list[Path] = []
        filenames: list[str] = []
        for file in files:
            safe_name = Path(file.filename or f"file_{len(file_paths)}").name
            tmp_path = tmp_input / safe_name
            content = await file.read()
            tmp_path.write_bytes(content)
            file_paths.append(tmp_path)
            filenames.append(file.filename or safe_name)

        try:
            logger.info("Processing batch of %d files", len(file_paths))
            results = _process_with_pipeline(file_paths, tmp_output)
            logger.info("Completed batch processing: %d results", len(results))

            return BatchOCRResponse(
                filenames=filenames,
                page_count=len(results),
                pages=results,
            )

        except Exception as e:
            logger.exception("Error processing batch")
            raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.host, port=settings.port)