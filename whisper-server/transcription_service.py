# transcription_service.py
import torch
import ray
from transformers import pipeline
from transformers.utils import is_flash_attn_2_available
from typing import Dict

# Initialize connection to Ray cluster
=ray.init(address="ray://192.168.1.65:10001")  # Use your head node IP

# Define a remote function for transcription
@ray.remote(num_gpus=1)  # Request 1 GPU
class RayTranscriptionService:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
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
                model_kwargs={"attn_implementation": attn_implementation},
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
        return outputs


# Wrapper class for local API interactions
class TranscriptionService:
    def __init__(self):
        # Create an actor pool to handle concurrent requests across the cluster
        self.services = [RayTranscriptionService.remote() for _ in range(7)]  # Up to 7 concurrent services
        self.next_service_idx = 0

    def _get_next_service(self):
        service = self.services[self.next_service_idx]
        self.next_service_idx = (self.next_service_idx + 1) % len(self.services)
        return service

    def transcribe_file(self, *args, **kwargs):
        service = self._get_next_service()
        return ray.get(service.transcribe_file.remote(*args, **kwargs))

    def transcribe_stream(self, *args, **kwargs):
        service = self._get_next_service()
        return ray.get(service.transcribe_stream.remote(*args, **kwargs))
