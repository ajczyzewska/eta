# EtA - EPUB to Audiobook

Convert EPUB books to audiobooks using XTTS v2 (Coqui TTS).

## Features

✅ **Automatic content filtering:**
- Skips covers, introductions, prefaces, tables of contents
- Removes page numbers
- Extracts only the actual book chapters

✅ **High-quality speech synthesis:**
- Voice cloning from your own voice sample
- Multi-language support (including Polish)
- Sentence-based text chunks (~300 characters) for fluency
- Crossfade between chunks

✅ **Checkpoint system:**
- Resume after interruption
- Safe process interruption

## Installation

### 1. Create virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate  # Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add voice sample (optional)

Place a WAV file with a voice sample (10-30 seconds) in the project directory as `speaker.wav` or `sample-agent.wav`.

## Usage

### Basic

```bash
python epub_to_audiobook.py your_book.epub
```

### With custom voice

```bash
python epub_to_audiobook.py book.epub --speaker my_voice.wav
```

### Advanced options

```bash
# Larger chunks (smoother speech)
python epub_to_audiobook.py book.epub --chunk-size 400

# Longer crossfade (smoother transitions)
python epub_to_audiobook.py book.epub --crossfade 150

# No crossfade
python epub_to_audiobook.py book.epub --crossfade 0

# Slower speech (0.5-0.9 for slower, useful if speech is too fast)
python epub_to_audiobook.py book.epub --speed 0.75

# Faster speech (1.1-2.0 for faster)
python epub_to_audiobook.py book.epub --speed 1.5

# Auto-optimize based on system capabilities (recommended)
python epub_to_audiobook.py book.epub --optimize auto

# Maximum speed (requires GPU)
python epub_to_audiobook.py book.epub --optimize speed

# Best quality (smaller chunks, smoother speech)
python epub_to_audiobook.py book.epub --optimize quality

# Show skipped elements
python epub_to_audiobook.py book.epub --verbose

# Resume from last checkpoint
python epub_to_audiobook.py book.epub --resume
```

## Parameters

| Parameter | Description | Default |
|----------|------|-----------|
| `epub_file` | Path to EPUB file | (required) |
| `--speaker` | WAV file with voice sample | sample-agent.wav |
| `--output` | Output directory | book_name_audio |
| `--chunk-size` | Chunk size (characters) | 300 |
| `--crossfade` | Crossfade in ms (0=disable) | 100 |
| `--speed` | Speech speed (0.5-2.0) | 1.0 |
| `--optimize` | Optimization profile (auto/speed/balanced/quality/off) | off |
| `--resume` | Resume from checkpoint | false |
| `--verbose` | Show details | false |

## System Requirements

### CPU
- Works on CPU (slower)
- Estimated time: ~5s per 300 character chunk

### GPU (recommended)
- NVIDIA GPU with CUDA
- 10x faster processing
- Automatic GPU detection and usage

## Output Structure

```
book_name_audio/
├── 01_Chapter_1.mp3
├── 02_Chapter_2.mp3
├── 03_Chapter_3.mp3
└── ...
```

## Performance Optimization

The `--optimize` parameter automatically detects your system capabilities and adjusts settings for optimal performance:

### Optimization Profiles

**`auto` (recommended)** - Automatically selects the best profile based on your hardware:
- **GPU with 8+ GB VRAM** → `speed` profile
- **GPU with < 8 GB VRAM** → `balanced` profile
- **CPU with 8+ cores, 16+ GB RAM** → `balanced` profile
- **Other systems** → `quality` profile

**`speed`** - Maximum generation speed (requires GPU):
- Uses larger chunks (400 characters)
- Best for: Systems with powerful GPU
- Trade-off: Slightly less natural pauses between sentences

**`balanced`** - Balance between speed and quality:
- Uses standard chunks (300 characters)
- Best for: Most systems with GPU or powerful CPU
- Default settings when optimization is off

**`quality`** - Best audio quality:
- Uses smaller chunks (200 characters)
- Results in smoother, more natural speech
- Best for: Final production, when quality is priority
- Trade-off: Slower generation

**`off`** - Manual settings (default):
- Use your own `--chunk-size` parameter
- No automatic adjustments

### Examples

```bash
# Recommended: Let the system choose optimal settings
python epub_to_audiobook.py book.epub --optimize auto

# Force maximum speed (GPU required)
python epub_to_audiobook.py book.epub --optimize speed

# Prioritize quality over speed
python epub_to_audiobook.py book.epub --optimize quality

# Combine optimization with other options
python epub_to_audiobook.py book.epub --optimize auto --speed 0.8
```

### System Detection

The tool automatically detects:
- **GPU availability** (NVIDIA CUDA)
- **GPU memory** (VRAM)
- **CPU cores**
- **System RAM**

This information is displayed at startup and used to recommend optimal settings.

## Speech Speed Control

The `--speed` parameter allows you to adjust the playback speed of generated speech:

- **Values < 1.0** slow down speech (e.g., `--speed 0.75` = 25% slower)
- **Values > 1.0** speed up speech (e.g., `--speed 1.25` = 25% faster)
- **Range:** 0.5 to 2.0
- **Default:** 1.0 (normal speed)

**Examples:**
```bash
# Slow down speech by 25% (useful if current speed is too fast)
python epub_to_audiobook.py book.epub --speed 0.75

# Slow down speech by 50%
python epub_to_audiobook.py book.epub --speed 0.5

# Speed up by 25%
python epub_to_audiobook.py book.epub --speed 1.25
```

The speed adjustment is applied during post-processing without changing the pitch, maintaining natural voice quality.

## Troubleshooting

### Warning: "The text length exceeds the character limit of 224"

This is a normal warning - the model handles longer texts through internal streaming. Check if the generated audio is complete. If yes - you can ignore the warning.

If audio is cut off:
```bash
python epub_to_audiobook.py book.epub --chunk-size 200
```

### No GPU

If you see "Model loaded on CPU" but you have an NVIDIA GPU:
1. Check if you have CUDA installed
2. Install PyTorch with CUDA support:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

## TTS Model

The project uses **XTTS v2** (Coqui TTS):
- Model: `tts_models/multilingual/multi-dataset/xtts_v2`
- Multilingual (supports Polish and other languages)
- Voice cloning from your own voice sample
- High-quality speech synthesis

## License

This project uses the XTTS v2 model from Coqui TTS, which is available under the Mozilla Public License 2.0.

## Authors
Project created based on an assignment from the Developer Jutra course by [Tomasz Ducin](https://developerjutra.pl/)

Developed by: [Aga Czyżewska](https://www.czyzewska.pro)

Project created with help from Claude. 
