import asyncio
import json
import logging
from pathlib import Path

import httpx

from conf import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("client.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


SUPPORTED_EXTENSIONS = {"pdf", "png", "jpg", "jpeg", "bmp", "tiff", "webp"}

MIME_TYPES = {
    "pdf": "application/pdf",
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "bmp": "image/bmp",
    "tiff": "image/tiff",
    "webp": "image/webp",
}


def save_results(
    pages: dict[str, dict],
    base_name: str,
    output_dir: Path,
) -> None:
    """Save OCR results (markdown and json) to output directory."""
    # Create subdirectory for this file's results
    file_output_dir = output_dir / base_name
    file_output_dir.mkdir(parents=True, exist_ok=True)

    for page_name, page_data in pages.items():
        markdown = page_data.get("markdown")
        json_data = page_data.get("json")

        if markdown:
            md_path = file_output_dir / f"{page_name}.md"
            md_path.write_text(markdown, encoding="utf-8")
            logger.info("Saved: %s", md_path)

        if json_data:
            json_path = file_output_dir / f"{page_name}.json"
            json_path.write_text(
                json.dumps(json_data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            logger.info("Saved: %s", json_path)


async def process_files(
    input_dir: str | None = None,
    output_dir: str | None = None,
    api_url: str | None = None,
    use_batch: bool = True,
) -> None:
    """
    Send all supported files in the input directory to the OCR API.
    
    Args:
        input_dir: Directory containing files to process
        output_dir: Directory to save results
        api_url: OCR API URL
        use_batch: If True, send all files in a single batch request (more efficient)
    """
    input_dir = input_dir or settings.input_dir
    output_dir = output_dir or settings.output_dir
    api_url = api_url or settings.ocr_api_url

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Find all supported files
    files: list[Path] = []
    for ext in SUPPORTED_EXTENSIONS:
        files.extend(input_path.glob(f"*.{ext}"))
        files.extend(input_path.glob(f"*.{ext.upper()}"))

    if not files:
        logger.warning("No supported files found in %s", input_dir)
        return

    files = sorted(files)
    logger.info("Found %d files to process", len(files))

    async with httpx.AsyncClient(timeout=settings.client_timeout) as client:
        if use_batch and len(files) > 1:
            # Batch processing - send all files at once
            await _process_batch(client, files, output_path, api_url)
        else:
            # Individual processing
            for file_path in files:
                await _process_single(client, file_path, output_path, api_url)


async def _process_single(
    client: httpx.AsyncClient,
    file_path: Path,
    output_path: Path,
    api_url: str,
) -> None:
    """Process a single file."""
    logger.info("Processing: %s", file_path.name)

    ext = file_path.suffix.lower().lstrip(".")
    mime_type = MIME_TYPES.get(ext, "application/octet-stream")

    with open(file_path, "rb") as f:
        response = await client.post(
            api_url,
            files={"file": (file_path.name, f, mime_type)},
        )

    if response.status_code == 200:
        result = response.json()
        logger.info("Completed: %s (%d pages)", file_path.name, result["page_count"])

        # Save results to output directory
        base_name = file_path.stem
        save_results(result["pages"], base_name, output_path)
    else:
        logger.error(
            "Failed: %s - %d %s",
            file_path.name,
            response.status_code,
            response.text,
        )


async def _process_batch(
    client: httpx.AsyncClient,
    file_paths: list[Path],
    output_path: Path,
    api_url: str,
) -> None:
    """Process multiple files in a single batch request."""
    logger.info("Processing batch of %d files", len(file_paths))

    # Prepare multipart files
    files_data: list[tuple[str, tuple[str, bytes, str]]] = []
    for file_path in file_paths:
        ext = file_path.suffix.lower().lstrip(".")
        mime_type = MIME_TYPES.get(ext, "application/octet-stream")
        content = file_path.read_bytes()
        files_data.append(("files", (file_path.name, content, mime_type)))

    batch_url = api_url.rstrip("/") + "/batch"
    response = await client.post(batch_url, files=files_data)

    if response.status_code == 200:
        result = response.json()
        logger.info("Batch completed: %d pages", result["page_count"])

        # Group results by source file (page names like "CV1_0" -> base "CV1")
        grouped: dict[str, dict[str, dict]] = {}
        for page_name, page_data in result["pages"].items():
            # Extract base filename (e.g., "CV1_0" -> "CV1")
            base_name = page_name.rsplit("_", 1)[0] if "_" in page_name else page_name
            if base_name not in grouped:
                grouped[base_name] = {}
            grouped[base_name][page_name] = page_data

        # Save each group
        for base_name, pages in grouped.items():
            save_results(pages, base_name, output_path)
    else:
        logger.error("Batch failed: %d %s", response.status_code, response.text)


if __name__ == "__main__":
    asyncio.run(process_files())