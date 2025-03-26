import torch
from transformers import pipeline
from transformers.utils import is_flash_attn_2_available
import gc
from typing import Dict
import logging

logger = logging.getLogger(__name__)


class TranscriptionService:
    def __init__(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model_cache: Dict[str, pipeline] = {}
        self.current_model = None

    def get_pipeline(self, model_name: str):
        # Only load the model if it's not already loaded
        if model_name not in self.model_cache:
            attn_implementation = (
                "flash_attention_2" if is_flash_attn_2_available() else "sdpa"
            )
            model_pipeline = pipeline(
                "automatic-speech-recognition",
                model=model_name,
                torch_dtype=torch.float16
                if torch.cuda.is_available()
                else torch.float32,
                device=self.device,
                model_kwargs={
                    "attn_implementation": attn_implementation,
                    "use_cache": True,  # Enable KV cache for efficiency
                },
            )
            self.model_cache[model_name] = model_pipeline
            self.current_model = model_name
            logger.info(f"Loaded model: {model_name}")
        return self.model_cache[model_name]

    def cleanup_request_memory(self):
        """Clean up memory specific to the current request"""
        if torch.cuda.is_available():
            # Clear CUDA cache for temporary tensors
            torch.cuda.empty_cache()
            # Force garbage collection for any temporary objects
            gc.collect()
            # Log memory usage for monitoring
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            logger.info(
                f"GPU Memory after request cleanup - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB"
            )

    def transcribe_file(
        self,
        file_path,
        model_name,
        task,
        language,
        chunk_length_s,
        batch_size,
        timestamp,
    ):
        try:
            model_pipeline = self.get_pipeline(model_name)
            generate_kwargs = {"task": task}
            if language:
                generate_kwargs["language"] = language
            ts = "word" if timestamp == "word" else True

            outputs = model_pipeline(
                file_path,
                chunk_length_s=chunk_length_s,
                batch_size=batch_size,
                generate_kwargs=generate_kwargs,
                return_timestamps=ts,
            )
            return outputs
        finally:
            self.cleanup_request_memory()

    def transcribe_stream(
        self,
        audio_buffer,
        model_name,
        task,
        language,
        chunk_length_s,
        batch_size,
        timestamp,
    ):
        try:
            model_pipeline = self.get_pipeline(model_name)
            generate_kwargs = {"task": task}
            if language:
                generate_kwargs["language"] = language
            ts = "word" if timestamp == "word" else True

            outputs = model_pipeline(
                audio_buffer,
                chunk_length_s=chunk_length_s,
                batch_size=batch_size,
                generate_kwargs=generate_kwargs,
                return_timestamps=ts,
            )
            return outputs
        finally:
            self.cleanup_request_memory()
