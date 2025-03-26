import io
import logging
import time
from fastapi import FastAPI, Request, UploadFile, File, Form, Depends
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
    model: str = "openai/whisper-large-v3"
    task: str = "transcribe"
    language: str = None
    chunk_length_s: int = 30
    batch_size: int = 24
    timestamp: str = "word"
    force_language_detection: bool = False


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
        model = params.get("model", "openai/whisper-large-v3")
        task = params.get("task", "transcribe")
        language = params.get("language")
        chunk_length_s = params.get("chunk_length_s", 30)
        batch_size = params.get("batch_size", 24)
        timestamp = params.get("timestamp", "word")
        force_language_detection = params.get("force_language_detection", False)

        # If force_language_detection is True, we'll explicitly set language to None
        # to force Whisper to detect the language
        if force_language_detection:
            language = None

        contents = await file.read()
        temp_file_path = save_temp_file(contents, "temp_audio.wav")

        # First pass to detect language if needed
        detected_language = None
        if language is None or force_language_detection:
            logger.info("Performing language detection")
            # Use a smaller chunk for faster language detection
            language_detection = transcription_service.transcribe_file(
                temp_file_path,
                model,
                "transcribe",  # Always use transcribe for language detection
                None,  # Force language detection
                10,  # Smaller chunk for quicker detection
                8,  # Smaller batch size
                False,  # No timestamps needed for detection
            )
            detected_language = language_detection.get("language", "en")
            logger.info(f"Detected language: {detected_language}")

            # Use detected language for the full transcription if original language was None
            if language is None and not force_language_detection:
                language = detected_language

        # Perform the full transcription with or without the detected language
        outputs = transcription_service.transcribe_file(
            temp_file_path,
            model,
            task,
            language,
            chunk_length_s,
            batch_size,
            timestamp,
        )

        # Add detected language to output
        if detected_language:
            outputs["detected_language"] = detected_language

        # Add the language that was actually used
        outputs["used_language"] = language

        remove_temp_file(temp_file_path)
        return JSONResponse(content=outputs)
    except Exception as e:
        logger.error(f"Error in transcription: {str(e)}", exc_info=True)
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.post("/transcribe/stream")
async def transcribe_stream(
    request: Request,
    model: str = Form("openai/whisper-large-v3"),
    task: str = Form("transcribe"),
    language: str = Form(None),
    chunk_length_s: int = Form(30),
    batch_size: int = Form(24),
    timestamp: str = Form("word"),
    force_language_detection: bool = Form(False),
):
    try:
        audio_buffer = io.BytesIO()

        # If force_language_detection is True, we'll explicitly set language to None
        if force_language_detection:
            language = None

        async def transcribe_generator():
            detected_language = None

            # Collect all audio data first
            async for chunk in request.stream():
                audio_buffer.write(chunk)

            # Reset buffer position
            audio_buffer.seek(0)

            # First perform language detection if needed
            if language is None or force_language_detection:
                logger.info("Performing language detection on stream")
                language_detection = transcription_service.transcribe_stream(
                    audio_buffer, model, "transcribe", None, 10, 8, False
                )
                detected_language = language_detection.get("language", "en")
                logger.info(f"Detected language in stream: {detected_language}")

                # Reset buffer position after detection
                audio_buffer.seek(0)

                # Use detected language for full transcription if original language was None
                used_language = (
                    detected_language
                    if language is None and not force_language_detection
                    else language
                )
            else:
                used_language = language

            # Full transcription
            outputs = transcription_service.transcribe_stream(
                audio_buffer,
                model,
                task,
                used_language,
                chunk_length_s,
                batch_size,
                timestamp,
            )

            # Add detected language to output
            if detected_language:
                outputs["detected_language"] = detected_language

            # Add the language that was actually used
            outputs["used_language"] = used_language

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
