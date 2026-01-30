from pydantic import BaseModel, Field


class ModelSettings(BaseModel):
    """OCR model configuration settings."""

    use_doc_preprocessor: bool = False
    use_layout_detection: bool = True
    use_chart_recognition: bool = False
    use_seal_recognition: bool = False
    use_ocr_for_image_block: bool = False
    format_block_content: bool = False
    merge_layout_blocks: bool = True
    markdown_ignore_labels: list[str] = []
    return_layout_polygon_points: bool = True


class ParsingBlock(BaseModel):
    """A single parsed block from OCR."""

    block_label: str
    block_content: str
    block_bbox: list[int]
    block_id: int
    block_order: int | None = None
    group_id: int | None = None
    global_block_id: int | None = None
    global_group_id: int | None = None
    block_polygon_points: list[list[float]] | None = None


class OCRResultJSON(BaseModel):
    """Full OCR result JSON structure for a page."""

    input_path: str
    page_index: int | None = None
    page_count: int | None = None
    width: int
    height: int
    model_settings: ModelSettings
    parsing_res_list: list[ParsingBlock]


class PageData(BaseModel):
    """OCR result for a single page."""

    json: OCRResultJSON | None = Field(default=None, description="Structured OCR result")
    markdown: str | None = Field(default=None, description="Markdown formatted text")


class OCRResponse(BaseModel):
    """Response for single file OCR endpoint."""

    filename: str | None = Field(description="Original filename")
    page_count: int = Field(description="Number of pages processed")
    pages: dict[str, PageData] = Field(description="OCR results keyed by page name")


class BatchOCRResponse(BaseModel):
    """Response for batch OCR endpoint."""

    filenames: list[str] = Field(description="Original filenames")
    page_count: int = Field(description="Total number of pages processed")
    pages: dict[str, PageData] = Field(description="OCR results keyed by page name")
