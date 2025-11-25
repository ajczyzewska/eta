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
- Large text chunks (3000 characters) for fluency
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
python epub_to_audiobook.py book.epub --chunk-size 5000

# Longer crossfade (smoother transitions)
python epub_to_audiobook.py book.epub --crossfade 150

# No crossfade
python epub_to_audiobook.py book.epub --crossfade 0

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
| `--chunk-size` | Chunk size (characters) | 3000 |
| `--crossfade` | Crossfade in ms (0=disable) | 100 |
| `--resume` | Resume from checkpoint | false |
| `--verbose` | Show details | false |

## System Requirements

### CPU
- Works on CPU (slower)
- Estimated time: ~20s per 3000 character chunk

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
