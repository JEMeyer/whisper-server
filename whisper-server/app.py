import io
import logging
import time
import os
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
import uvicorn
from .transcription_service import TranscriptionService
from .utils import save_temp_file, remove_temp_file
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

transcription_service = TranscriptionService()


class TranscriptionRequest(BaseModel):
    model: str = "openai/whisper-large-v3-turbo"
    task: str = "transcribe"
    language: str = None
    chunk_length_s: int = 30
    batch_size: int = 8
    timestamp: str = "word"


@app.middleware("http")
async def log_duration(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    logger.info(f"Request to {request.url.path} took {duration:.2f} seconds")
    return response


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...), payload: str = Form(...)):
    try:
        # Parse the JSON payload
        params = json.loads(payload)

        # Extract parameters with defaults
        model = params.get("model", "openai/whisper-large-v3-turbo")
        task = params.get("task", "transcribe")
        language = params.get("language")
        chunk_length_s = params.get("chunk_length_s", 30)
        batch_size = params.get("batch_size", 8)
        timestamp = params.get("timestamp", "word")

        contents = await file.read()
        unique_filename = f"temp_audio_{int(time.time())}_{os.urandom(4).hex()}.wav"
        temp_file_path = save_temp_file(contents, unique_filename)

        outputs = transcription_service.transcribe_file(
            temp_file_path,
            model,
            task,
            language,
            chunk_length_s,
            batch_size,
            timestamp,
        )

        remove_temp_file(temp_file_path)
        return JSONResponse(content=outputs)
    except Exception as e:
        logger.error(f"Error in transcription: {str(e)}", exc_info=True)
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.post("/transcribe/stream")
async def transcribe_stream(
    request: Request,
    model: str = Form("openai/whisper-large-v3-turbo"),
    task: str = Form("transcribe"),
    language: str = Form(None),
    chunk_length_s: int = Form(30),
    batch_size: int = Form(8),
    timestamp: str = Form("word"),
):
    try:
        audio_buffer = io.BytesIO()

        async def transcribe_generator():
            # Collect all audio data first
            async for chunk in request.stream():
                audio_buffer.write(chunk)

            # Reset buffer position
            audio_buffer.seek(0)

            # Full transcription
            outputs = transcription_service.transcribe_stream(
                audio_buffer,
                model,
                task,
                language,
                chunk_length_s,
                batch_size,
                timestamp,
            )

            yield json.dumps(outputs)

        return StreamingResponse(transcribe_generator(), media_type="application/json")
    except Exception as e:
        logger.error(f"Error in stream transcription: {str(e)}", exc_info=True)
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.get("/health")
async def health():
    return {"status": "healthy"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
