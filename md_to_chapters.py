#!/usr/bin/env python3
"""
Markdown to Chapters JSON converter.

Splits a Markdown file on level-1 headings (# ) into a JSON array
matching the chapter dict format expected by epub_to_audiobook.py.

Usage:
    python md_to_chapters.py book.md --output chapters.json
    python md_to_chapters.py book.md  # prints JSON to stdout

Zero external dependencies — stdlib only.
"""

import argparse
import json
import re
import sys


def clean_text(text: str) -> str:
    """Normalize whitespace and strip common artifacts."""
    text = re.sub(r'\r\n', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'&nbsp;', ' ', text)
    text = re.sub(r'<[^>]+>', '', text)
    return text.strip()


def parse_markdown_chapters(markdown_text: str) -> list:
    """
    Split markdown on level-1 headings.

    Returns list of dicts: [{"title": "...", "content": "..."}, ...]
    """
    parts = re.split(r'^# ', markdown_text, flags=re.MULTILINE)

    chapters = []
    for part in parts:
        part = part.strip()
        if not part:
            continue

        lines = part.split('\n', 1)
        title = lines[0].strip()
        content = clean_text(lines[1]) if len(lines) > 1 else ''

        if not content:
            continue

        chapters.append({
            'title': title,
            'content': content,
        })

    return chapters


def _title_from_filename(filename: str) -> str:
    """Derive a chapter title from a filename like '01_intro.md' → 'intro'."""
    stem = filename.rsplit('.', 1)[0]  # remove extension
    # Strip leading numeric prefix (e.g. "01_", "03-")
    stem = re.sub(r'^\d+[_\-\s]*', '', stem)
    # Replace underscores and hyphens with spaces
    stem = stem.replace('_', ' ').replace('-', ' ')
    return stem.strip() or filename


def load_chapters_from_md_folder(folder_path: str) -> list:
    """
    Load .md files from a folder, each file becoming one chapter.

    Files are sorted alphabetically (use numeric prefixes to control order).
    Title is taken from the first # heading, or derived from the filename.

    Returns list of dicts: [{"title": "...", "content": "..."}, ...]
    """
    import os
    import glob as glob_mod

    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    md_files = sorted(glob_mod.glob(os.path.join(folder_path, '*.md')))

    chapters = []
    for filepath in md_files:
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()

        if not text.strip():
            continue

        # Check for a leading # heading
        match = re.match(r'^# ([^\n]+)\n?(.*)', text, re.DOTALL)
        if match:
            title = match.group(1).strip()
            content = clean_text(match.group(2))
        else:
            title = _title_from_filename(os.path.basename(filepath))
            content = clean_text(text)

        if not content:
            continue

        chapters.append({
            'title': title,
            'content': content,
        })

    return chapters


def main():
    parser = argparse.ArgumentParser(
        description="Convert Markdown with # headings to chapters JSON"
    )
    parser.add_argument("markdown_file", help="Path to Markdown file")
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output JSON file (default: stdout)"
    )
    args = parser.parse_args()

    try:
        with open(args.markdown_file, 'r', encoding='utf-8') as f:
            markdown_text = f.read()
    except FileNotFoundError:
        print(f"Error: File not found: {args.markdown_file}", file=sys.stderr)
        sys.exit(1)

    chapters = parse_markdown_chapters(markdown_text)

    if not chapters:
        print("Error: No chapters found. Expected level-1 headings (# Chapter Title)", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(chapters)} chapter(s)", file=sys.stderr)

    output_json = json.dumps(chapters, ensure_ascii=False, indent=2)

    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(output_json)
        print(f"Written to {args.output}", file=sys.stderr)
    else:
        print(output_json)


if __name__ == "__main__":
    main()
