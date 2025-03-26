import torch
from transformers import pipeline
from transformers.utils import is_flash_attn_2_available
import gc
from typing import Dict


class TranscriptionService:
    def __init__(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model_cache: Dict[str, pipeline] = {}

    def get_pipeline(self, model_name: str):
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
        return self.model_cache[model_name]

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
        self.cleanup_memory()
        return outputs

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
        self.cleanup_memory()
        return outputs

    def cleanup_memory(self):
        """Release CUDA memory after transcription"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.empty_cache()
