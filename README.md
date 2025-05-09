# AniVoiceChanger API

API for voice conversion using RVC models.

## Environment Variables

The following environment variables can be set to configure the API:

- `MODEL_NAME`: Default model name to use (without the .pth extension).
- `PITCH_CHANGE`: Default pitch change value (integer).
- `VOLUME_ENVELOPE`: Default volume envelope value (float).
- `INDEX_RATE`: Default index rate value (float).
- `PITCH_EXTRACTION_ALGO`: Default pitch extraction algorithm (e.g., rmvpe).
- `GPU_INDEX`: GPU index to use (e.g., 0).

## Running the API

To run the API, use the following command:

```bash
python -m uvicorn main_api:app --host 0.0.0.0 --port 8000 --reload
```

This assumes you are in the `api` directory. If you are in the root directory, the command would be:

```bash
python -m uvicorn api.main_api:app --host 0.0.0.0 --port 8000 --reload
```

## API Endpoints

### Root

- **GET /**
  - Description: Root endpoint.
  - Response:
    - `200 OK`: `{ "message": "AniVoiceChanger API is running" }`

### Models

- **GET /models**
  - Description: Lists available models.
  - Response:
    - `200 OK`: `{ "models": ["model1", "model2"] }` (List of available .pth model file stems in the `weights` directory)

### Convert Voice (File Upload)

- **POST /convert**
  - Description: Converts voice from an uploaded audio file.
  - Parameters (Form Data):
    - `audio_file` (file, required): The audio file to convert.
    - `model_name` (str, optional): Name of the RVC model to use. Defaults to `MODEL_NAME` env var.
    - `pitch_change` (int, optional): Pitch change value. Defaults to `PITCH_CHANGE` env var or 0.
    - `volume_envelope` (float, optional): Volume envelope. Defaults to `VOLUME_ENVELOPE` env var or 1.0.
    - `index_rate` (float, optional): Index rate. Defaults to `INDEX_RATE` env var or 0.
    - `pitch_extraction_algo` (str, optional): Pitch extraction algorithm. Defaults to `PITCH_EXTRACTION_ALGO` env var or 'rmvpe'.
    - `f0_method_weight` (float, optional): Weight for hybrid f0 method. Defaults to 0.33.
  - Response:
    - `200 OK`: The converted audio file (`audio/wav`).
    - `404 Not Found`: If the specified model is not found.
    - `500 Internal Server Error`: If an error occurs during audio processing.

### Convert Voice (Stream Setup)

- **POST /convert/stream**
  - Description: Sets up a session for streaming audio conversion. The actual streaming is handled by the WebSocket endpoint.
  - Request Body (`application/json`):
    ```json
    {
      "model_name": "your_model_name", // Optional, defaults to MODEL_NAME env var
      "pitch_change": 0, // Optional, defaults to PITCH_CHANGE env var or 0
      "volume_envelope": 1.0, // Optional, defaults to VOLUME_ENVELOPE env var or 1.0
      "index_rate": 0.0, // Optional, defaults to INDEX_RATE env var or 0
      "pitch_extraction_algo": "rmvpe", // Optional, defaults to PITCH_EXTRACTION_ALGO env var
      "f0_method_weight": 0.33 // Optional, defaults to 0.33
    }
    ```
  - Response:
    - `200 OK`: `{ "session_id": "your_session_id", "websocket_url": "/ws/your_session_id" }`

### Convert Voice (WebSocket Stream)

- **WEBSOCKET /ws/{session_id}**
  - Description: WebSocket endpoint for real-time audio streaming and conversion.
  - Path Parameters:
    - `session_id` (str, required): The session ID obtained from the `/convert/stream` endpoint.
  - Communication Protocol:
    1. Client connects to the WebSocket.
    2. Client sends a JSON message with `VoiceConversionParams`:
       ```json
       {
         "model_name": "your_model_name",
         "pitch_change": 0,
         "volume_envelope": 1.0,
         "index_rate": 0.0,
         "pitch_extraction_algo": "rmvpe",
         "f0_method_weight": 0.33
       }
       ```
    3. Client sends audio chunks (bytes).
    4. Server sends back processed audio chunks (bytes).
    5. If an error occurs, the server sends a JSON message: `{ "error": "error_message" }`.
