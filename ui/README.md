# UI for ETa (EPUB -> Audiobook)

This folder contains a minimal Flask-based local web UI to run the existing `epub_to_audiobook.py` script without using terminal commands.

Quick start (macOS):

1. Create a Python virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Run the UI:

```bash
python ui/app.py
```

3. Open your browser at `http://127.0.0.1:5000` and use the form to upload an EPUB and (optionally) a speaker WAV.

Notes:
- The UI runs the CLI script (`epub_to_audiobook.py`) as a background process and writes logs to `ui/run.log`.
- Uploaded files are saved to `uploads/` next to the repository root.
- This UI is intended for local use only. Do not expose it to the public internet without adding authentication.
