# Whisper Server

This project provides a REST API for automatic speech transcription using the [OpenAI Whisper Large V3 Turbo](https://huggingface.co/openai/whisper-large-v3-turbo) model as the primary transcription model.

## Technology Stack

- **Python** for the application code.
- **FastAPI** for building the REST API: see [`whisper-server/app.py`](whisper-serverr/app.py)
- **Uvicorn** as the ASGI server.
- **PyTorch** for running the inference on CUDA when available.
- **Transformers, Optimum, and Accelerate** for model handling and optimizations.
- **Docker** for containerization, with a Dockerfile to build a container image.

## Getting Started

### Installation

1. Clone the repository.
2. Install the required dependencies via [setup.py](setup.py):

   ```sh
   pip install .
   ```

3. Start the FastAPI server using Uvicorn:

    ```sh
    uvicorn whisper-server.app:app --reload
    ```

### Usage

Once the server is running, you can access the API documentation at `http://127.0.0.1:8000/docs`.

### Example Request

To transcribe an audio file, send a POST request to the `/transcribe` endpoint with the audio file:

```sh
curl -X POST "http://127.0.0.1:8000/transcribe" -F "file=@path_to_your_audio_file.wav"
```

### Docker

To run the server using Docker:

1. Build the Docker image:

    ```sh
    docker build -t whisper-server .
    ```

2. Run the Docker container:

    ```sh
    docker run -p 8000:8000 whisper-server
    ```

### Contributing

Contributions are welcome! Please open an issue or submit a pull request.

### License

This project is licensed under the MIT License.
