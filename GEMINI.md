# GEMINI.md

## Project Overview

**plex-generate-previews** is a Python-based CLI tool and Docker container designed to generate video preview thumbnails (BIF files) for Plex Media Server. Its primary goal is to accelerate this process using GPU hardware acceleration (NVIDIA, AMD, Intel, Windows D3D11VA) and parallel processing, significantly outperforming Plex's built-in generation.

## Architecture

The application follows a producer-consumer model using threading for concurrency.

*   **CLI (`plex_generate_previews/cli.py`)**: The entry point. Parses arguments, initializes configuration, and starts the processing loop.
*   **Configuration (`plex_generate_previews/config.py`)**: Centralized configuration management. Loads settings from CLI arguments, environment variables (`.env`), and defaults. Enforces precedence (CLI > Env > Default) and performs robust validation.
*   **Worker System (`plex_generate_previews/worker.py`)**:
    *   **`Worker`**: Represents a single processing thread (CPU or GPU). Manages its own state (busy/idle), progress tracking, and execution of the processing task.
    *   **`WorkerPool`**: Manages the collection of `Worker` instances. It handles task assignment from a central queue, manages the fallback mechanism (GPU -> CPU on codec failure), and reports aggregate progress.
*   **Media Processing (`plex_generate_previews/media_processing.py`)**: Contains the core logic for interacting with FFmpeg to generate BIF files.
*   **Plex Client (`plex_generate_previews/plex_client.py`)**: Wrapper around `plexapi` to interact with the Plex Media Server.

## Key Directories & Files

*   **`/plex_generate_previews`**: Main source code directory.
    *   `cli.py`: Application entry point.
    *   `config.py`: Configuration logic and validation.
    *   `worker.py`: Threading and task management logic.
    *   `media_processing.py`: FFmpeg wrapper and BIF generation logic.
    *   `plex_client.py`: Plex API interaction.
    *   `gpu_detection.py`: Logic for detecting available GPUs and their capabilities.
*   **`/tests`**: Pytest test suite.
*   **`pyproject.toml`**: Project metadata, dependencies, and build configuration.
*   **`compose.yaml`**: Reference Docker Compose configuration.
*   **`Dockerfile`**: Container build definition.

## Development Workflow

### 1. Installation & Setup

The project uses `setuptools` and standard Python tooling.

```bash
# Install dependencies and the package in editable mode
pip install -e .[test]

# Install optional GPU dependencies if needed
pip install -e .[nvidia]
# or
pip install -e .[amd]
```

### 2. Running the Application

The application can be run via the installed console script or directly as a module.

```bash
# Via console script
plex-generate-previews --help

# As a module
python -m plex_generate_previews --help
```

### 3. Testing

The project uses `pytest` for testing.

```bash
# Run all tests with coverage
pytest

# Run specific tests
pytest tests/test_worker.py
```

### 4. Docker

The Docker image is built using the provided `Dockerfile`.

```bash
# Build image
docker build -t plex-generate-previews .

# Run with compose
docker compose up
```

**Note on Docker:** The container uses `s6-overlay`. Do **not** use `init: true` in Docker Compose, as it conflicts with s6's process supervision.

## Configuration

Configuration is handled by `plex_generate_previews/config.py`.

**Priority:**
1.  CLI Arguments
2.  Environment Variables
3.  Default Values

**Key Environment Variables:**
*   `PLEX_URL`: URL of the Plex Media Server.
*   `PLEX_TOKEN`: Authentication token.
*   `PLEX_CONFIG_FOLDER`: Path to the Plex config directory (crucial for identifying libraries).
*   `GPU_THREADS`: Number of concurrent GPU tasks.
*   `CPU_THREADS`: Number of concurrent CPU tasks.
*   `TMP_FOLDER`: Directory for temporary file generation.

## Common Tasks

*   **Adding a new CLI argument:** Update `plex_generate_previews/cli.py` to parse it and `plex_generate_previews/config.py` to load and validate it.
*   **Modifying FFmpeg logic:** Check `plex_generate_previews/media_processing.py`.
*   **Changing Threading behavior:** Look into `plex_generate_previews/worker.py`.

## Constraints & Conventions

*   **Python Version:** 3.7+ compatibility is required.
*   **Code Style:** Adhere to existing formatting.
*   **Path Mappings:** Critical for Docker usage. Ensure `PLEX_VIDEOS_PATH_MAPPING` and `PLEX_LOCAL_VIDEOS_PATH_MAPPING` logic in `config.py` is respected when handling file paths.
