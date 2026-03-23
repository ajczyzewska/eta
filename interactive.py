#!/usr/bin/env python3
"""
Interactive wizard for EtA — guides users through input selection and settings.

Usage:
    config = run_wizard()
    # config is a dict with keys: input_type, path, title, author, etc.
"""

MENU = """
  [1] EPUB file
  [2] Markdown folder (each .md = one chapter)
  [3] Generate book from prompt
"""

INPUT_TYPES = {
    "1": "epub",
    "2": "md_folder",
    "3": "generate",
}


def _ask_choice() -> str:
    """Ask user to pick an input type. Re-prompts on invalid input."""
    print("\nEtA — Text to Audiobook\n")
    print("What would you like to convert?")
    print(MENU)

    while True:
        choice = input("> ").strip()
        if choice in INPUT_TYPES:
            return INPUT_TYPES[choice]
        print(f"Invalid choice '{choice}'. Please enter 1, 2, or 3.")


def _ask_epub() -> dict:
    path = input("Path to EPUB file: ").strip()
    return {"input_type": "epub", "path": path}


def _ask_md_folder() -> dict:
    path = input("Path to markdown folder: ").strip()
    title = input("Book title (Enter to skip): ").strip() or "Unknown title"
    author = input("Author (Enter to skip): ").strip() or "Unknown author"
    return {"input_type": "md_folder", "path": path, "title": title, "author": author}


def _ask_generate() -> dict:
    path = input("Path to prompts folder: ").strip()
    return {"input_type": "generate", "path": path}


_HANDLERS = {
    "epub": _ask_epub,
    "md_folder": _ask_md_folder,
    "generate": _ask_generate,
}


def run_wizard() -> dict:
    """Run the interactive wizard. Returns a config dict."""
    input_type = _ask_choice()
    return _HANDLERS[input_type]()
