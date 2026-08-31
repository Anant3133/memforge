from dataclasses import dataclass
import torch

@dataclass
class EngineConfig:
    # Model configuration
    target_model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    draft_model_name: str = "meta-llama/Llama-3.2-1B-Instruct"
    fallback_target_model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    fallback_draft_model: str = "HuggingFaceTB/SmolLM-135M"

    # Memory & Paging settings
    block_size: int = 16
    num_gpu_blocks: int = 512
    max_num_seqs: int = 64
    max_model_len: int = 2048

    # Speculative decoding settings
    draft_lookahead_k: int = 4
    temperature: float = 0.7
    top_p: float = 0.9

    # Device & Precision
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float16 if torch.cuda.is_available() else torch.float32

DEFAULT_CONFIG = EngineConfig()
