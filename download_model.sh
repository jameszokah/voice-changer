#!/bin/bash

# Create directories if they don't exist
mkdir -p weights logs assets/hubert assets/pretrained_v2 assets/uvr5_weights

echo "Downloading base models required for voice conversion API..."

# Download HuBERT base model
echo "Downloading HuBERT base model..."
wget -q --show-progress -O "assets/hubert/hubert_base.pt" "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt"

# Download RMVPE model for pitch extraction
echo "Downloading RMVPE model..."
wget -q --show-progress -O "rmvpe.pt" "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt"

# Download pretrained models
echo "Downloading pretrained models..."
wget -q --show-progress -O "assets/pretrained_v2/D40k.pth" "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/D40k.pth"
wget -q --show-progress -O "assets/pretrained_v2/G40k.pth" "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/G40k.pth"
wget -q --show-progress -O "assets/pretrained_v2/f0D40k.pth" "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/f0D40k.pth"
wget -q --show-progress -O "assets/pretrained_v2/f0G40k.pth" "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/f0G40k.pth"

# Download UVR5 models for audio separation
echo "Downloading UVR5 models..."
wget -q --show-progress -O "assets/uvr5_weights/HP2-人声vocals+非人声instrumentals.pth" "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/uvr5_weights/HP2-人声vocals+非人声instrumentals.pth"
wget -q --show-progress -O "assets/uvr5_weights/HP5-主旋律人声vocals+其他instrumentals.pth" "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/uvr5_weights/HP5-主旋律人声vocals+其他instrumentals.pth"

echo "Downloading TITAN models..."
wget -q --show-progress -O "assets/weights/titan.pth" "https://huggingface.co/blaise-tk/TITAN/resolve/main/models/medium/48k/model/f048k-Titan-Medium.pth"
wget -q --show-progress -O "assets/weights/titan.index" "https://huggingface.co/blaise-tk/TITAN/resolve/main/models/medium/48k/model/added_IVF256_Flat_nprobe_1_v2.index"


echo "Download complete! All necessary models for the API have been downloaded."
echo "To use your own voice models, place .pth files in the 'weights' directory and set the MODEL_NAME environment variable."