# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Plex Generate Video Previews is a Python tool that generates video preview thumbnails (BIF files) for Plex Media Server with GPU acceleration. It significantly speeds up thumbnail generation compared to Plex's built-in system by leveraging hardware acceleration (NVIDIA CUDA, AMD/Intel VAAPI, Windows D3D11VA, Apple VideoToolbox) and parallel processing.

## Architecture

### Core Components

**CLI Entry Point** (`plex_generate_previews/cli.py`)
- Main application orchestrator with Rich-based progress UI
- Manages worker pool lifecycle and library processing
- Handles signal cleanup and temporary directory management
- Creates two progress displays: main task progress and per-worker FFmpeg progress

**Configuration** (`plex_generate_previews/config.py`)
- Loads settings from CLI arguments (priority) and environment variables
- Validates required parameters (Plex URL/token, config folder structure)
- Checks for FFmpeg availability and proper Plex directory structure
- Provides Docker-optimized help when running in containers

**Worker Pool** (`plex_generate_previews/worker.py`)
- Threading-based (not multiprocessing) worker implementation
- Separate GPU and CPU worker types with round-robin GPU assignment
- CPU fallback queue for codec errors (e.g., AV1 on older GPUs)
- Thread-safe progress updates using locks
- Dynamic task assignment - workers pull from queue as they become available

**Media Processing** (`plex_generate_previews/media_processing.py`)
- FFmpeg execution with hardware acceleration flags (before `-i` input flag)
- HDR detection and tone mapping using zscale/tonemap filters
- Skip-frame heuristic: tests `-skip_frame:v nokey` on first 10 frames, retries without if fails
- Codec error detection via stderr patterns and exit codes (-22/234/69)
- Raises `CodecNotSupportedError` for GPU workers to trigger CPU fallback
- BIF file generation from timestamp-named JPEGs

**Plex Client** (`plex_generate_previews/plex_client.py`)
- PlexAPI wrapper with retry logic for XML parsing errors
- Duplicate location filtering (multi-part episodes sharing same file)
- Library filtering by configured names
- Sort support (newest/oldest by `addedAt`)

**GPU Detection** (`plex_generate_previews/gpu_detection.py`)
- Platform-specific detection: NVIDIA (pynvml), AMD (amdsmi), Intel/AMD (VAAPI via `/dev/dri`), Windows (D3D11VA), Apple (VideoToolbox)
- Returns list of tuples: `(gpu_type, gpu_device_path, gpu_info_dict)`
- Used by CLI to select GPUs via `--gpu-selection` (all or comma-separated indices)

**Web Interface** (`plex_generate_previews/web/`)
- FastAPI application with SQLModel database (SQLite)
- Scheduler for automated processing with configurable intervals
- Plex OAuth authentication flow for setup
- Models: MediaItem, PreviewStatus, AppSettings
- Runs alongside CLI in Docker via `wrapper.sh`

### Processing Flow

1. **Startup**: CLI loads config, detects GPUs, connects to Plex, creates worker pool
2. **Library Query**: Iterate through Plex libraries, filter by configured names, apply sort order
3. **Task Distribution**: Workers pull tasks from main queue; GPU workers can fallback to CPU queue on codec errors
4. **Thumbnail Generation**:
   - Query Plex for media tree to get bundle hash and file path
   - Apply path mappings if configured (Docker volume mounts)
   - Skip if BIF already exists (unless `--regenerate-thumbnails`)
   - Run FFmpeg with GPU acceleration flags and progress monitoring
   - Generate timestamped JPEGs at configured interval
   - Create BIF file from JPEGs
5. **Progress Tracking**: Workers update progress via callbacks; CLI renders Rich UI with per-worker FFmpeg stats
6. **Cleanup**: Remove temp directories, shutdown workers gracefully

### Key Design Patterns

**Hardware Acceleration Order**: FFmpeg flags must be: `-hwaccel cuda` (or `-hwaccel vaapi -vaapi_device /dev/dri/renderD128`) **before** `-i input.mp4`

**Codec Fallback**: GPU workers raise `CodecNotSupportedError` on unsupported codecs → added to CPU fallback queue → CPU workers process without GPU flags

**Skip Frame Logic**: Test `-skip_frame:v nokey` with `-err_detect explode -xerror` on first 10 frames; if fails, retry entire encode without skip-frame

**Progress Parsing**: Poll FFmpeg stderr file every 5ms for `time=HH:MM:SS` lines; extract frame/fps/speed for Rich UI

**Path Mappings**: Convert Plex paths to local paths via `--plex-videos-path-mapping` and `--plex-local-videos-path-mapping` (critical for Docker)

## Development Commands

### Local Development

Install in editable mode with test dependencies:
```bash
pip install -e .
pip install -e ".[test]"
```

Run the CLI:
```bash
# Using console script
plex-generate-previews --plex-url http://localhost:32400 --plex-token YOUR_TOKEN --plex-config-folder /path/to/plex

# Using module
python -m plex_generate_previews --help
```

List detected GPUs:
```bash
plex-generate-previews --list-gpus
```

Run with debug logging:
```bash
plex-generate-previews --log-level DEBUG --plex-url ... --plex-token ...
```

### Testing

Run full test suite with coverage:
```bash
pytest
# Or with parallel execution
pytest -n auto
```

Run specific test file:
```bash
pytest tests/test_media_processing.py -v
```

Run with coverage report:
```bash
pytest --cov=plex_generate_previews --cov-report=html
```

Test markers:
```bash
pytest -m "not slow"  # Skip slow integration tests
```

### Docker

Build multi-arch image:
```bash
docker build -t plex-generate-previews .
```

Run with GPU (NVIDIA):
```bash
docker run --rm --gpus all \
  -e PLEX_URL=http://localhost:32400 \
  -e PLEX_TOKEN=your_token \
  -e PLEX_CONFIG_FOLDER=/config/plex/Library/Application\ Support/Plex\ Media\ Server \
  -v /path/to/plex/config:/config/plex \
  -v /path/to/media:/media \
  plex-generate-previews
```

Run with GPU (Intel/AMD VAAPI):
```bash
docker run --rm \
  --device=/dev/dri:/dev/dri \
  -e PUID=1000 -e PGID=1000 \
  -e PLEX_URL=... -e PLEX_TOKEN=... -e PLEX_CONFIG_FOLDER=... \
  -v /path/to/plex/config:/config/plex \
  -v /path/to/media:/media \
  plex-generate-previews
```

List GPUs in Docker:
```bash
docker run --rm --gpus all plex-generate-previews --list-gpus
```

### Web Interface

Start web server (for development):
```bash
plex-previews-web
# Or
python -m plex_generate_previews.web.main
```

Access at `http://localhost:8008`

In Docker, both CLI and web interface run via `wrapper.sh` which:
1. Checks for web mode (`MODE=web` or `--web` flag)
2. Runs `plex-previews-web` for web mode
3. Runs `plex-generate-previews` for CLI mode (default)

## Configuration

### Required Environment Variables
- `PLEX_URL`: Plex server URL (http://localhost:32400)
- `PLEX_TOKEN`: Plex authentication token
- `PLEX_CONFIG_FOLDER`: Path to Plex config (must contain `Media/localhost/` with BIF bundles)

### Important Environment Variables
- `GPU_THREADS`: Number of GPU workers (default: 1)
- `CPU_THREADS`: Number of CPU workers (default: 1)
- `GPU_SELECTION`: "all" or "0,1,2" (default: all)
- `PLEX_BIF_FRAME_INTERVAL`: Seconds between thumbnails (default: 5)
- `THUMBNAIL_QUALITY`: 1-10 scale, 2=highest (default: 4)
- `REGENERATE_THUMBNAILS`: Force regeneration (default: false)
- `LOG_LEVEL`: DEBUG, INFO, WARNING, ERROR (default: INFO)

### Path Mappings (Docker/Remote)
When Plex sees different paths than the tool:
- `PLEX_VIDEOS_PATH_MAPPING`: Plex's path prefix (e.g., `/server/media`)
- `PLEX_LOCAL_VIDEOS_PATH_MAPPING`: Local/container path prefix (e.g., `/media`)

Example: Plex sees `/server/media/movies/avatar.mkv`, container sees `/media/movies/avatar.mkv` → map `/server/media` to `/media`

## Common Issues

### "No GPUs detected"
- Install GPU drivers (NVIDIA/AMD/Intel)
- For Docker: use `--gpus all` (NVIDIA) or `--device=/dev/dri` (VAAPI)
- Fallback: `--gpu-threads 0 --cpu-threads 4`

### "Permission denied" (VAAPI Docker)
- Set `PUID` and `PGID` environment variables to match host user
- Add to render group: `--group-add $(getent group render | cut -d: -f3)`

### "PLEX_CONFIG_FOLDER does not exist"
- Must point to directory containing `Media/localhost/` subdirectories
- Check Docker volume mounts match expected paths
- Validate with: `ls -la /path/to/plex/Library/Application\ Support/Plex\ Media\ Server/Media/localhost`

### "Skipping as file not found"
- Path mappings incorrect - Plex paths don't match container paths
- Enable debug logging: `--log-level DEBUG` to see path conversions
- Verify media volume mounts in Docker

### FFmpeg codec errors
- GPU workers automatically fallback to CPU for unsupported codecs (e.g., AV1 on RTX 2060)
- Check `--list-gpus` output for supported acceleration types
- Ensure `CPU_THREADS > 0` to enable fallback

## Code Style Notes

- Uses `loguru` for logging (not stdlib `logging`)
- Progress UI built with `rich` library (Progress, Live, Console)
- Configuration via `dataclass` with validation helpers
- Threading (not multiprocessing) for worker pool
- Path handling: `sanitize_path()` utility for cross-platform compatibility
- FFmpeg stderr parsing: regex patterns for `frame=`, `fps=`, `time=`, `speed=`
- BIF format: Custom binary format with magic bytes `[0x89, 0x42, 0x49, 0x46, 0x0d, 0x0a, 0x1a, 0x0a]`

## Testing Patterns

Test fixtures in `tests/conftest.py`:
- `mock_config`: Pre-configured Config object
- `temp_dir`: Temporary directory with cleanup
- `fixtures_dir`: Path to test fixtures (sample videos)

Mocking FFmpeg:
```python
@patch('subprocess.Popen')
def test_ffmpeg_execution(mock_popen, mock_config, temp_dir):
    # Mock FFmpeg process with stderr file
    mock_proc = MagicMock()
    mock_proc.poll.return_value = 0  # Success
    mock_popen.return_value = mock_proc
    # Test code...
```

Testing worker pool:
```python
from plex_generate_previews.worker import WorkerPool

pool = WorkerPool(gpu_workers=2, cpu_workers=2, selected_gpus=[...])
pool.process_items(media_items, config, plex, progress_manager, title_max_width=40)
pool.shutdown()
```
