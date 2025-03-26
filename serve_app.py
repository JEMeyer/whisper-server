# whisper-server/serve_app.py

import ray
from ray import serve
import torch
import io
from fastapi import FastAPI, Request, UploadFile, File, Body
from fastapi.responses import JSONResponse
from transformers import pipeline
from transformers.utils import is_flash_attn_2_available
from typing import Dict
import tempfile
import os

# Connect to your existing Ray cluster
ray.init(address="ray://192.168.1.65:10001")

app = FastAPI()

@serve.deployment(
    num_replicas=1,  # Start with 1 instance
    ray_actor_options={"num_gpus": 1}  # Each instance gets 1 GPU
)
class TranscriptionService:
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
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device=self.device,
                model_kwargs={"attn_implementation": attn_implementation},
            )
            self.model_cache[model_name] = model_pipeline
        return self.model_cache[model_name]

    async def transcribe_file(
        self,
        file_bytes,
        model_name="openai/whisper-large-v3",
        task="transcribe",
        language=None,
        chunk_length_s=30,
        batch_size=24,
        timestamp="word",
    ):
        # Save temp file
        temp_dir = tempfile.mkdtemp()
        temp_file_path = os.path.join(temp_dir, "temp_audio.wav")

        with open(temp_file_path, "wb") as f:
            f.write(file_bytes)

        try:
            model_pipeline = self.get_pipeline(model_name)
            generate_kwargs = {"task": task}
            if language:
                generate_kwargs["language"] = language
            ts = "word" if timestamp == "word" else True

            outputs = model_pipeline(
                temp_file_path,
                chunk_length_s=chunk_length_s,
                batch_size=batch_size,
                generate_kwargs=generate_kwargs,
                return_timestamps=ts,
            )
            return outputs
        finally:
            # Clean up
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
            if os.path.exists(temp_dir):
                os.rmdir(temp_dir)

    async def health(self):
        return {"status": "healthy"}

# Create and deploy the service
transcription_deployment = TranscriptionService.bind()

@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    model: str = Body("openai/whisper-large-v3"),
    task: str = Body("transcribe"),
    language: str = Body(None),
    chunk_length_s: int = Body(30),
    batch_size: int = Body(24),
    timestamp: str = Body("word"),
):
    try:
        contents = await file.read()
        outputs = await transcription_deployment.transcribe_file.remote(
            contents, model, task, language, chunk_length_s, batch_size, timestamp
        )
        return JSONResponse(content=outputs)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/health")
async def health():
    result = await transcription_deployment.health.remote()
    return result

# Deploy the application
serve.run(app, host="0.0.0.0", port=8000, route_prefix="/")
