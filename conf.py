from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # vLLM Server
    vllm_server_url: str = "http://localhost:8118/v1"
    vllm_model_name: str = "PaddleOCR-VL-1.5-0.9B"

    # FastAPI Server
    host: str = "0.0.0.0"
    port: int = 8080

    # Client
    ocr_api_url: str = "http://localhost:8080/ocr"
    input_dir: str = "./demo"
    output_dir: str = "./output"
    client_timeout: float = 300.0


settings = Settings()
