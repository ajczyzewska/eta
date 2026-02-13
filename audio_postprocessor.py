"""
Audio post-processing for audiobook quality improvement.

Uses FFmpeg to clean up TTS-generated audio:
- Denoising (vocoder artifacts)
- DC offset removal (boundary clicks)
- Click/pop removal (crossfade artifacts)
- Silence normalization
- Loudness normalization (EBU R128 for audiobooks)
"""

import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional


class AudioPostprocessor:
    """Post-process TTS audio to remove artifacts and improve quality."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self._check_ffmpeg()

    def _check_ffmpeg(self):
        """Check if FFmpeg is available."""
        try:
            subprocess.run(
                ['ffmpeg', '-version'],
                capture_output=True,
                check=True
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise RuntimeError(
                "FFmpeg not found. Please install:\n"
                "  macOS: brew install ffmpeg\n"
                "  Ubuntu/Debian: sudo apt-get install ffmpeg\n"
                "  Windows: Download from ffmpeg.org"
            )

    def process_audio(
        self,
        input_path: str,
        output_path: str,
        denoise: bool = True,
        declick: bool = True,
        normalize: bool = True,
    ) -> bool:
        """
        Process audio file through cleanup pipeline.

        Args:
            input_path: Input audio file
            output_path: Output audio file
            denoise: Apply denoising filter
            declick: Remove clicks and pops
            normalize: Normalize loudness (EBU R128)

        Returns:
            True if successful
        """
        filters = []

        # Step 1: Denoise (remove TTS vocoder artifacts)
        if denoise:
            filters.append("afftdn=nf=-25:tn=1")

        # Step 2: Remove DC offset (prevents boundary clicks)
        filters.append("highpass=f=40,lowpass=f=15000")

        # Step 3: Remove clicks and pops
        if declick:
            filters.append("adeclick=threshold=0.05:window=55:overlap=75")
            filters.append("acompressor=threshold=-20dB:ratio=4:attack=5:release=50")

        # Step 4: Loudness normalization (EBU R128 for audiobooks)
        if normalize:
            filters.append("loudnorm=I=-16:TP=-1.5:LRA=11")

        # Build FFmpeg command with codec matching output format
        output_ext = Path(output_path).suffix.lower()
        if output_ext == '.wav':
            codec_args = ['-c:a', 'pcm_s16le']
        else:
            codec_args = ['-c:a', 'libmp3lame', '-b:a', '192k']

        cmd = [
            'ffmpeg',
            '-i', input_path,
            '-af', ','.join(filters),
            *codec_args,
            '-y',
            output_path,
        ]

        if not self.verbose:
            cmd.extend(['-loglevel', 'error'])

        try:
            subprocess.run(cmd, check=True, capture_output=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"FFmpeg error: {e.stderr.decode()}")
            return False

    def process_chapter(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        backup: bool = True,
    ) -> str:
        """
        Process a complete chapter file.

        Args:
            input_path: Chapter audio file
            output_path: Output path (defaults to replacing input in-place)
            backup: Create backup of original

        Returns:
            Path to processed file
        """
        input_path = Path(input_path)

        if output_path is None:
            output_path = input_path
        output_path = Path(output_path)

        # Backup original if requested
        if backup:
            backup_path = input_path.parent / f"{input_path.stem}_original{input_path.suffix}"
            if not backup_path.exists():
                shutil.copy2(input_path, backup_path)
                if self.verbose:
                    print(f"Backup created: {backup_path}")

        # Process to temp file then replace
        temp_path = input_path.parent / f"{input_path.stem}_temp{input_path.suffix}"

        if self.verbose:
            print(f"Processing: {input_path.name}")

        success = self.process_audio(str(input_path), str(temp_path))

        if success:
            os.replace(str(temp_path), str(output_path))
            if self.verbose:
                print(f"Saved: {output_path}")
            return str(output_path)
        else:
            if temp_path.exists():
                os.remove(temp_path)
            raise RuntimeError(f"Failed to process {input_path}")
