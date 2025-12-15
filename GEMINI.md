# GEMINI.md

## Project Overview

**plex-generate-previews** is a Python-based CLI tool and Docker container designed to generate video preview thumbnails (BIF files) for Plex Media Server. Its primary goal is to accelerate this process using GPU hardware acceleration (NVIDIA, AMD, Intel, Windows D3D11VA) and parallel processing, significantly outperforming Plex's built-in generation.

## Architecture

The application operates on a producer-consumer model:

*   **Entry Point (`cli.py`)**: Parses command-line arguments and initializes the application.
*   **Configuration (`config.py`)**: Manages settings with a strict precedence order: CLI Arguments > Environment Variables > Defaults. It handles validation of paths, permissions, and Plex connectivity.
*   **Worker System (`worker.py`)**:
    *   **`WorkerPool`**: Manages a collection of `Worker` threads. It assigns tasks from a queue of media items.
    *   **`Worker`**: Represents a processing thread (GPU or CPU). It executes the FFmpeg task, tracks progress, and handles fallback logic (e.g., if a GPU worker fails on an unsupported codec, the task is handed off to a CPU worker).
*   **Media Processing (`media_processing.py`)**: Wraps FFmpeg commands to generate BIF files. It handles codec detection and hardware acceleration flags.
*   **Plex Integration (`plex_client.py`)**: Uses `plexapi` to query the Plex Server for libraries and media items.

## Key Directories & Files

*   **`plex_generate_previews/`**: Source code.
    *   `cli.py`: Main entry point.
    *   `config.py`: Configuration loading and validation logic.
    *   `worker.py`: Threading and task management.
    *   `media_processing.py`: FFmpeg wrapper (inferred).
    *   `gpu_detection.py`: Logic for detecting available GPUs.
*   **`tests/`**: Pytest suite.
*   **`pyproject.toml`**: Python project metadata, dependencies, and build configuration.
*   **`Dockerfile`**: Container definition (based on LinuxServer.io images, inferred from `s6-overlay` mentions in README).
*   **`README.md`**: Comprehensive user documentation.

## Building and Running

### Local Development (Python)

1.  **Install:**
    ```bash
    pip install -e .[test]
    ```
2.  **Run:**
    ```bash
    plex-generate-previews --help
    # or
    python -m plex_generate_previews --help
    ```
3.  **Test:**
    ```bash
    pytest
    ```

### Docker

1.  **Build:**
    ```bash
    docker build -t plex-generate-previews .
    ```
2.  **Run:**
    ```bash
    docker run --rm \
      -e PLEX_URL=... \
      -e PLEX_TOKEN=... \
      plex-generate-previews
    ```

## Configuration & Conventions

*   **Precedence:** CLI args always override environment variables.
*   **Environment Variables:** Used heavily for Docker configuration (e.g., `PLEX_URL`, `PLEX_TOKEN`, `GPU_THREADS`).
*   **Path Mappings:** Critical for Docker/Remote setups. `PLEX_VIDEOS_PATH_MAPPING` and `PLEX_LOCAL_VIDEOS_PATH_MAPPING` translate Plex's file paths to paths accessible by this tool.
*   **Logging:** Uses `loguru`. Log level can be set via `LOG_LEVEL` env var.
*   **Code Style:** Python 3.7+ compatible.
