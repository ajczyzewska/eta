#!/usr/bin/env python3
"""
Batch post-process existing audiobook chapters.

Usage:
    python batch_postprocess.py audiobook_directory/
    python batch_postprocess.py chapter_01.mp3 chapter_02.mp3
    python batch_postprocess.py audiobook_directory/ --no-backup
"""

import argparse
import sys
from pathlib import Path

from audio_postprocessor import AudioPostprocessor


def main():
    parser = argparse.ArgumentParser(
        description='Post-process audiobook chapters to improve quality'
    )
    parser.add_argument(
        'paths',
        nargs='+',
        help='Audio files or directories to process'
    )
    parser.add_argument(
        '--no-backup',
        action='store_true',
        help='Do not create backup files'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Verbose output'
    )

    args = parser.parse_args()

    # Collect all audio files
    audio_files = []
    for path_str in args.paths:
        path = Path(path_str)

        if path.is_file() and path.suffix in ['.mp3', '.wav']:
            audio_files.append(path)
        elif path.is_dir():
            audio_files.extend(path.glob('*.mp3'))
            audio_files.extend(path.glob('*.wav'))
        else:
            print(f"Skipping: {path_str} (not a file or directory)")

    if not audio_files:
        print("No audio files found!")
        return 1

    print(f"Processing {len(audio_files)} files...")

    processor = AudioPostprocessor(verbose=args.verbose)

    success_count = 0
    for audio_file in sorted(audio_files):
        try:
            processor.process_chapter(
                str(audio_file),
                backup=not args.no_backup
            )
            success_count += 1
        except Exception as e:
            print(f"Error processing {audio_file.name}: {e}")
            continue

    print(f"Done! Processed {success_count}/{len(audio_files)} files.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
