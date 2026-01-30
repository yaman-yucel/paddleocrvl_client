import logging
from pathlib import Path

from paddleocr import PaddleOCRVL

logger = logging.getLogger(__name__)


def setup_logging(log_file: str = "ocr_processing.log") -> None:
    """Configure logging to both file and console."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )


def process_pdfs(input_dir: str = "./demo", output_dir: str = "./output") -> None:
    """Process all PDFs in the input directory."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    pipeline = PaddleOCRVL(
        vl_rec_backend="vllm-server",
        vl_rec_server_url="http://localhost:8118/v1",
        vl_rec_api_model_name="PaddleOCR-VL-1.5-0.9B",
    )

    pdf_files = list(input_path.glob("*.pdf"))
    logger.info("Found %d PDF files to process", len(pdf_files))

    for pdf_file in pdf_files:
        logger.info("Processing: %s", pdf_file.name)

        # Create output subdirectory for each PDF
        pdf_output_dir = output_path / pdf_file.stem
        pdf_output_dir.mkdir(parents=True, exist_ok=True)

        output = pipeline.predict(input=str(pdf_file))
        pages_res = list(output)

        restructured = pipeline.restructure_pages(pages_res)

        for res in restructured:
            res.save_to_json(save_path=str(pdf_output_dir))
            res.save_to_markdown(save_path=str(pdf_output_dir))

        logger.info("Completed: %s -> %s", pdf_file.name, pdf_output_dir)


if __name__ == "__main__":
    setup_logging()
    process_pdfs()