import asyncio
import logging
from pathlib import Path

import httpx

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("client.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


async def process_pdfs(
    input_dir: str = "./demo",
    api_url: str = "http://localhost:8080/ocr",
) -> None:
    """Send all PDFs in the input directory to the OCR API."""
    input_path = Path(input_dir)
    pdf_files = list(input_path.glob("*.pdf"))

    if not pdf_files:
        logger.warning("No PDF files found in %s", input_dir)
        return

    logger.info("Found %d PDF files to process", len(pdf_files))

    async with httpx.AsyncClient(timeout=300.0) as client:
        for pdf_file in pdf_files:
            logger.info("Processing: %s", pdf_file.name)

            with open(pdf_file, "rb") as f:
                response = await client.post(
                    api_url,
                    files={"file": (pdf_file.name, f, "application/pdf")},
                )

            if response.status_code == 200:
                result = response.json()
                logger.info("Completed: %s (%d pages)", pdf_file.name, result["pages"])

                # Log markdown content for each page
                for i, page in enumerate(result["results"]):
                    if page.get("markdown"):
                        logger.info("Page %d:\n%s", i + 1, page["markdown"])
            else:
                logger.error(
                    "Failed: %s - %d %s",
                    pdf_file.name,
                    response.status_code,
                    response.text,
                )


if __name__ == "__main__":
    asyncio.run(process_pdfs())