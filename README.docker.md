# AniVoiceChanger Docker Setup

This document provides instructions for running the AniVoiceChanger API using Docker.

## Prerequisites

- Docker (https://docs.docker.com/get-docker/)
- Docker Compose (https://docs.docker.com/compose/install/)
- For GPU support: NVIDIA Container Toolkit (https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

## Quick Start

1. **Build and start the container**

   ```bash
   docker-compose up -d
   ```

   This will build the Docker image, download all necessary models automatically, and start the container in detached mode.

2. **Access the API**

   The API will be available at `http://localhost:8000`

## Configuration

### Environment Variables

You can configure the API by setting environment variables in the `.env` file or by passing them to the `docker-compose up` command:

```bash
MODEL_NAME=titan PITCH_CHANGE=12 docker-compose up -d
```

Available environment variables:

- `MODEL_NAME`: Name of the model to use (without .pth extension)
- `PITCH_CHANGE`: Default pitch change value (integer)
- `VOLUME_ENVELOPE`: Default volume envelope value (float)
- `INDEX_RATE`: Default index rate value (float)
- `PITCH_EXTRACTION_ALGO`: Default pitch extraction algorithm (e.g., rmvpe)
- `GPU_INDEX`: GPU index to use (e.g., 0)

### GPU Support

To enable GPU support, uncomment the following lines in `docker-compose.yml`:

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

## Managing Models

The Docker image comes with the TITAN model pre-installed. To use it, set the environment variable:

```bash
MODEL_NAME=titan docker-compose up -d
```

If you want to add additional models, place them in the `weights` directory of your host machine, and they will be available to the API through the volume mount.

## API Usage

See the main [README.md](README.md) for details on API endpoints and usage.
