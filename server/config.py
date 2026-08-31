from pydantic import BaseModel

class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    draft_model_name: str = "HuggingFaceTB/SmolLM-135M"
    enable_speculative: bool = True
    max_concurrency: int = 32

SERVER_CONFIG = ServerConfig()
