import sys
import wave
import tempfile
import os
import uuid
import io
from pathlib import Path
from os import getenv
from typing import Optional, List

import torch
import numpy as np
from fastapi import (
    FastAPI,
    File,
    UploadFile,
    Form,
    WebSocket,
    BackgroundTasks,
    HTTPException,
)
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from scipy.io import wavfile
from multiprocessing import cpu_count
from dotenv import load_dotenv

# Add parent directory to path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

# Import RVC modules
from infer.modules.vc.modules import VC
from infer.lib.infer_pack.models import (
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs256NSFsid_nono,
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs768NSFsid_nono,
)
from infer.lib.audio import load_audio
from fairseq import checkpoint_utils

# Create FastAPI app
app = FastAPI(
    title="VoiceChanger API",
    description="API for voice conversion using RVC models",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Load environment variables
load_dotenv()
DEFAULT_MODEL_NAME = getenv("MODEL_NAME")
if DEFAULT_MODEL_NAME and DEFAULT_MODEL_NAME.endswith(".pth"):
    DEFAULT_MODEL_NAME = DEFAULT_MODEL_NAME[:-4]
DEFAULT_PITCH_CHANGE = int(getenv("PITCH_CHANGE", "0"))
DEFAULT_VOLUME_ENVELOPE = float(getenv("VOLUME_ENVELOPE", "1.0"))
DEFAULT_INDEX_RATE = float(getenv("INDEX_RATE", "0"))
DEFAULT_PITCH_EXTRACTION_ALGO = getenv("PITCH_EXTRACTION_ALGO", "rmvpe")
GPU_INDEX = getenv("GPU_INDEX", "0")

# Audio processing settings
CHUNK = 1024
TEMP_DIR = BASE_DIR / "voiceChanger" / "audio" / "temp"
TEMP_DIR.mkdir(parents=True, exist_ok=True)


# Input validation model
class VoiceConversionParams(BaseModel):
    model_name: str = DEFAULT_MODEL_NAME
    pitch_change: int = DEFAULT_PITCH_CHANGE
    volume_envelope: float = DEFAULT_VOLUME_ENVELOPE
    index_rate: float = DEFAULT_INDEX_RATE
    pitch_extraction_algo: str = DEFAULT_PITCH_EXTRACTION_ALGO
    f0_method_weight: float = 0.33  # Default best weight for hybrid f0 method


class Config:
    def __init__(self, device, is_half):
        self.device = device
        self.is_half = is_half
        self.n_cpu = 0
        self.gpu_name = None
        self.gpu_mem = None
        self.x_pad, self.x_query, self.x_center, self.x_max = self.device_config()

    def device_config(self) -> tuple:
        if torch.cuda.is_available():
            i_device = int(self.device.split(":")[-1])
            self.gpu_name = torch.cuda.get_device_name(i_device)
            if (
                ("16" in self.gpu_name and "V100" not in self.gpu_name.upper())
                or "P40" in self.gpu_name.upper()
                or "1060" in self.gpu_name
                or "1070" in self.gpu_name
                or "1080" in self.gpu_name
            ):
                print("16 series/10 series P40 forced single precision")
                self.is_half = False
                for config_file in ["32k.json", "40k.json", "48k.json"]:
                    with open(f"configs/{config_file}", "r") as f:
                        strr = f.read().replace("true", "false")
                    with open(f"configs/{config_file}", "w") as f:
                        f.write(strr)
                with open("trainset_preprocess_pipeline_print.py", "r") as f:
                    strr = f.read().replace("3.7", "3.0")
                with open("trainset_preprocess_pipeline_print.py", "w") as f:
                    f.write(strr)
            else:
                self.gpu_name = None
            self.gpu_mem = int(
                torch.cuda.get_device_properties(i_device).total_memory
                / 1024
                / 1024
                / 1024
                + 0.4
            )
            if self.gpu_mem <= 4:
                with open("trainset_preprocess_pipeline_print.py", "r") as f:
                    strr = f.read().replace("3.7", "3.0")
                with open("trainset_preprocess_pipeline_print.py", "w") as f:
                    f.write(strr)
        elif torch.backends.mps.is_available():
            print("No supported N-card found, use MPS for inference")
            self.device = "mps"
        else:
            print("No supported N-card found, use CPU for inference")
            self.device = "cpu"
            self.is_half = True

        if self.n_cpu == 0:
            self.n_cpu = cpu_count()

        if self.is_half:
            # 6G memory config
            x_pad = 3
            x_query = 10
            x_center = 60
            x_max = 65
        else:
            # 5G memory config
            x_pad = 1
            x_query = 6
            x_center = 38
            x_max = 41

        if self.gpu_mem != None and self.gpu_mem <= 4:
            x_pad = 1
            x_query = 5
            x_center = 30
            x_max = 32

        return x_pad, x_query, x_center, x_max


# Global variables for models
device = f"cuda:{GPU_INDEX}" if torch.cuda.is_available() else "cpu"
is_half = True if device.startswith("cuda") else False
config = Config(device, is_half)
hubert_model = None
model_cache = {}  # Cache for loaded models


def load_hubert():
    """Load the HuBERT model"""
    models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task(
        ["assets/hubert/hubert_base.pt"],
        suffix="",
    )
    hubert = models[0]
    hubert = hubert.to(device)

    if is_half:
        hubert = hubert.half()
    else:
        hubert = hubert.float()

    hubert.eval()
    return hubert


def get_vc(model_name):
    """Load and return the voice conversion model"""
    # Check if model is in cache
    if model_name in model_cache:
        return model_cache[model_name]

    # Load the model
    model_path = BASE_DIR / "assets" / "weights" / f"{model_name}.pth"
    if not model_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Model {model_name} not found. Make sure it exists in the weights directory.",
        )

    model_path = str(model_path)
    print(f"loading pth {model_path}")
    cpt = torch.load(model_path, map_location="cpu")
    tgt_sr = cpt["config"][-1]
    cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]
    if_f0 = cpt.get("f0", 1)
    version = cpt.get("version", "v1")

    if version == "v1":
        if if_f0 == 1:
            net_g = SynthesizerTrnMs256NSFsid(*cpt["config"], is_half=is_half)
        else:
            net_g = SynthesizerTrnMs256NSFsid_nono(*cpt["config"])
    elif version == "v2":
        if if_f0 == 1:
            net_g = SynthesizerTrnMs768NSFsid(*cpt["config"], is_half=is_half)
        else:
            net_g = SynthesizerTrnMs768NSFsid_nono(*cpt["config"])

    del net_g.enc_q
    net_g.load_state_dict(cpt["weight"], strict=False)
    net_g.eval().to(device)

    if is_half:
        net_g = net_g.half()
    else:
        net_g = net_g.float()

    vc = VC(tgt_sr, config)
    model_data = (cpt, version, net_g, tgt_sr, vc)

    # Cache the model
    model_cache[model_name] = model_data
    return model_data


def process_audio(input_path, output_path, params: VoiceConversionParams):
    """Process audio file with RVC"""
    # Get the voice conversion model
    cpt, version, net_g, tgt_sr, vc = get_vc(params.model_name)

    # Find index file
    logs_dir = BASE_DIR / "assets" / "weights" / params.model_name
    index_path = ""
    if logs_dir.exists():
        for file in logs_dir.iterdir():
            if file.suffix == ".index":
                index_path = str(logs_dir / file.name)
                break

    # Load and process audio
    audio = load_audio(input_path, 16000)
    times = [0, 0, 0]
    if_f0 = cpt.get("f0", 1)

    audio_opt = vc.pipeline(
        hubert_model,
        net_g,
        0,  # speaker id
        audio,
        input_path,
        times,
        params.pitch_change,
        params.pitch_extraction_algo,
        index_path,
        params.index_rate,
        if_f0,
        3,  # protection level
        tgt_sr,
        0,  # resample_sr
        params.volume_envelope,
        version,
        params.f0_method_weight,
        f0_file=None,
    )

    # Write output
    wavfile.write(output_path, tgt_sr, audio_opt)
    return output_path, tgt_sr


@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    global hubert_model
    print("Loading HuBERT model...")
    hubert_model = load_hubert()
    print("Server started and ready to process requests")


@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "AniVoiceChanger API is running"}


@app.get("/models")
async def get_models():
    """List available models"""
    models_dir = BASE_DIR / "assets" / "weights"
    if not models_dir.exists():
        return {"models": []}

    models = [file.stem for file in models_dir.iterdir() if file.suffix == ".pth"]
    return {"models": models}


@app.post("/convert")
async def convert_voice(
    audio_file: UploadFile = File(...),
    model_name: str = Form(DEFAULT_MODEL_NAME),
    pitch_change: int = Form(DEFAULT_PITCH_CHANGE),
    volume_envelope: float = Form(DEFAULT_VOLUME_ENVELOPE),
    index_rate: float = Form(DEFAULT_INDEX_RATE),
    pitch_extraction_algo: str = Form(DEFAULT_PITCH_EXTRACTION_ALGO),
    f0_method_weight: float = Form(0.33),
):
    """Convert voice from uploaded audio file"""
    # Create params
    params = VoiceConversionParams(
        model_name=model_name,
        pitch_change=pitch_change,
        volume_envelope=volume_envelope,
        index_rate=index_rate,
        pitch_extraction_algo=pitch_extraction_algo,
        f0_method_weight=f0_method_weight,
    )

    # Save uploaded file
    temp_id = str(uuid.uuid4())
    input_path = TEMP_DIR / f"input_{temp_id}.wav"
    output_path = TEMP_DIR / f"output_{temp_id}.wav"

    content = await audio_file.read()
    with open(input_path, "wb") as f:
        f.write(content)

    # Process audio
    try:
        await BackgroundTasks().add_task(
            process_audio, str(input_path), str(output_path), params
        )
        result_path, _ = process_audio(str(input_path), str(output_path), params)

        # Return processed file
        return FileResponse(
            path=result_path,
            filename="converted_voice.wav",
            media_type="audio/wav",
            background=BackgroundTasks().add_task(
                lambda: os.unlink(result_path) if os.path.exists(result_path) else None
            ),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")
    finally:
        # Clean up temporary files
        if os.path.exists(input_path):
            os.unlink(input_path)


@app.post("/convert/stream")
async def convert_voice_stream(params: VoiceConversionParams):
    """
    Setup for streaming audio conversion (WebSocket endpoint will handle actual streaming)
    Returns a session ID to use with the WebSocket
    """
    session_id = str(uuid.uuid4())
    return {"session_id": session_id, "websocket_url": f"/ws/{session_id}"}


@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time audio streaming and conversion"""
    await websocket.accept()

    # Create temporary files for this session
    input_buffer = io.BytesIO()
    frame_count = 0
    try:
        # Receive client parameters
        params_data = await websocket.receive_json()
        params = VoiceConversionParams(**params_data)

        # Process audio chunks as they come
        while True:
            # Receive audio chunk
            chunk = await websocket.receive_bytes()
            if not chunk:
                break

            # Add to buffer
            input_buffer.write(chunk)
            frame_count += 1

            # Process when we have enough data
            if frame_count >= 10:  # Process every ~10 chunks
                # Create temp files
                temp_input = TEMP_DIR / f"stream_in_{session_id}_{uuid.uuid4()}.wav"
                temp_output = TEMP_DIR / f"stream_out_{session_id}_{uuid.uuid4()}.wav"

                # Save buffer to file
                input_buffer.seek(0)
                with open(temp_input, "wb") as f:
                    f.write(input_buffer.read())

                # Process audio
                try:
                    result_path, _ = process_audio(
                        str(temp_input), str(temp_output), params
                    )

                    # Read processed audio and send back
                    with open(result_path, "rb") as f:
                        processed_audio = f.read()

                    await websocket.send_bytes(processed_audio)

                    # Clean up
                    os.unlink(temp_input)
                    os.unlink(temp_output)

                    # Reset buffer and counter
                    input_buffer = io.BytesIO()
                    frame_count = 0
                except Exception as e:
                    await websocket.send_json({"error": str(e)})

    except Exception as e:
        await websocket.send_json({"error": f"WebSocket error: {str(e)}"})
    finally:
        # Clean up any remaining temporary files
        for temp_file in TEMP_DIR.glob(f"stream_*_{session_id}_*"):
            if temp_file.exists():
                os.unlink(temp_file)


if __name__ == "__main__":
    uvicorn.run("main_api:app", host="0.0.0.0", port=8000, reload=True)
