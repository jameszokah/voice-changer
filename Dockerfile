# syntax=docker/dockerfile:1

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    ffmpeg \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements 
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create necessary directories
RUN mkdir -p weights logs voiceChanger/audio/temp assets/hubert assets/pretrained_v2 assets/uvr5_weights assets/weights

# Download all necessary models
RUN echo "Downloading HuBERT base model..." && \
    wget -q --show-progress -O "assets/hubert/hubert_base.pt" "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt" && \
    echo "Downloading RMVPE model..." && \
    wget -q --show-progress -O "rmvpe.pt" "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt" && \
    echo "Downloading pretrained models..." && \
    wget -q --show-progress -O "assets/pretrained_v2/D40k.pth" "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/D40k.pth" && \
    wget -q --show-progress -O "assets/pretrained_v2/G40k.pth" "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/G40k.pth" && \
    wget -q --show-progress -O "assets/pretrained_v2/f0D40k.pth" "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/f0D40k.pth" && \
    wget -q --show-progress -O "assets/pretrained_v2/f0G40k.pth" "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/f0G40k.pth" && \
    echo "Downloading UVR5 models..." && \
    wget -q --show-progress -O "assets/uvr5_weights/HP2-人声vocals+非人声instrumentals.pth" "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/uvr5_weights/HP2-人声vocals+非人声instrumentals.pth" && \
    wget -q --show-progress -O "assets/uvr5_weights/HP5-主旋律人声vocals+其他instrumentals.pth" "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/uvr5_weights/HP5-主旋律人声vocals+其他instrumentals.pth" && \
    echo "Downloading TITAN models..." && \
    wget -q --show-progress -O "assets/weights/titan.pth" "https://huggingface.co/blaise-tk/TITAN/resolve/main/models/medium/48k/model/f048k-Titan-Medium.pth" && \
    wget -q --show-progress -O "assets/weights/titan.index" "https://huggingface.co/blaise-tk/TITAN/resolve/main/models/medium/48k/model/added_IVF256_Flat_nprobe_1_v2.index" && \
    echo "All models downloaded successfully!"

# Copy the application code
COPY . .

# Set environment variables
ENV MODEL_NAME=""
ENV PITCH_CHANGE=0
ENV VOLUME_ENVELOPE=1.0
ENV INDEX_RATE=0.0
ENV PITCH_EXTRACTION_ALGO="rmvpe"
ENV GPU_INDEX=0

# Expose the API port
EXPOSE 8000

# Command to run the API server
CMD ["uvicorn", "api.main_api:app", "--host", "0.0.0.0", "--port", "8000"]
