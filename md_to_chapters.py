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
