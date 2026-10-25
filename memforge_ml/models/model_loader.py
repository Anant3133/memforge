"""
Model Loader: Handles Hugging Face weights, tokenizers, and fallback mechanisms
for both Target and Draft models.
"""

import torch
from typing import Tuple, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer


class ModelLoader:
    @staticmethod
    def load_model_and_tokenizer(
        model_name: str,
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
        fallback_model_name: Optional[str] = None,
    ) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """
        Loads a Hugging Face causal LM model and tokenizer, falling back to a lightweight
        model if the primary model is unavailable or cannot fit in memory.
        """
        target_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        target_dtype = torch_dtype or (torch.float16 if target_device == "cuda" else torch.float32)

        try:
            print(f"[ModelLoader] Attempting to load primary model: '{model_name}' on {target_device}...")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=target_dtype,
                device_map=target_device if target_device == "cuda" else None,
                low_cpu_mem_usage=True,
            )
            if target_device != "cuda":
                model.to(target_device)
            model.eval()
            print(f"[ModelLoader] Successfully loaded '{model_name}'.")
            return model, tokenizer

        except Exception as e:
            if fallback_model_name and fallback_model_name != model_name:
                print(f"[ModelLoader] Primary model '{model_name}' failed to load ({e}).")
                print(f"[ModelLoader] Falling back to lightweight model: '{fallback_model_name}'...")
                tokenizer = AutoTokenizer.from_pretrained(fallback_model_name)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token

                model = AutoModelForCausalLM.from_pretrained(
                    fallback_model_name,
                    torch_dtype=target_dtype,
                    device_map=target_device if target_device == "cuda" else None,
                    low_cpu_mem_usage=True,
                )
                if target_device != "cuda":
                    model.to(target_device)
                model.eval()
                print(f"[ModelLoader] Successfully loaded fallback model '{fallback_model_name}'.")
                return model, tokenizer
            else:
                raise e
