"""Tests for md_to_chapters.py — MD folder loading and markdown parsing."""

import pytest

from md_to_chapters import load_chapters_from_md_folder, parse_markdown_chapters


# === load_chapters_from_md_folder tests ===


class TestLoadChaptersFromMdFolder:
    def test_loads_files_sorted_alphabetically(self, tmp_path):
        (tmp_path / "02_second.md").write_text("# Second\nContent B", encoding="utf-8")
        (tmp_path / "01_first.md").write_text("# First\nContent A", encoding="utf-8")

        chapters = load_chapters_from_md_folder(str(tmp_path))

        assert len(chapters) == 2
        assert chapters[0]["title"] == "First"
        assert chapters[1]["title"] == "Second"

    def test_title_from_heading(self, tmp_path):
        (tmp_path / "chapter.md").write_text("# My Title\nSome content here.", encoding="utf-8")

        chapters = load_chapters_from_md_folder(str(tmp_path))

        assert chapters[0]["title"] == "My Title"
        assert "My Title" not in chapters[0]["content"]
        assert "Some content here." in chapters[0]["content"]

    def test_title_from_filename(self, tmp_path):
        (tmp_path / "01_intro.md").write_text("No heading here, just content.", encoding="utf-8")

        chapters = load_chapters_from_md_folder(str(tmp_path))

        assert chapters[0]["title"] == "intro"
        assert "No heading here" in chapters[0]["content"]

    def test_title_from_filename_strips_prefix_and_cleans(self, tmp_path):
        (tmp_path / "03-my-great-chapter.md").write_text("Content only.", encoding="utf-8")

        chapters = load_chapters_from_md_folder(str(tmp_path))

        assert chapters[0]["title"] == "my great chapter"

    def test_skips_empty_files(self, tmp_path):
        (tmp_path / "01_real.md").write_text("# Real\nContent.", encoding="utf-8")
        (tmp_path / "02_empty.md").write_text("", encoding="utf-8")
        (tmp_path / "03_whitespace.md").write_text("   \n\n  ", encoding="utf-8")

        chapters = load_chapters_from_md_folder(str(tmp_path))

        assert len(chapters) == 1
        assert chapters[0]["title"] == "Real"

    def test_skips_non_md_files(self, tmp_path):
        (tmp_path / "chapter.md").write_text("# Chapter\nContent.", encoding="utf-8")
        (tmp_path / "notes.txt").write_text("Not a chapter.", encoding="utf-8")
        (tmp_path / "image.png").write_bytes(b"\x89PNG")

        chapters = load_chapters_from_md_folder(str(tmp_path))

        assert len(chapters) == 1

    def test_content_cleaned(self, tmp_path):
        (tmp_path / "ch.md").write_text(
            "# Title\nSome &nbsp; text   with  <b>html</b>  artifacts.",
            encoding="utf-8",
        )

        chapters = load_chapters_from_md_folder(str(tmp_path))

        assert "&nbsp;" not in chapters[0]["content"]
        assert "<b>" not in chapters[0]["content"]
        assert "html" in chapters[0]["content"]

    def test_empty_folder_returns_empty_list(self, tmp_path):
        chapters = load_chapters_from_md_folder(str(tmp_path))
        assert chapters == []

    def test_nonexistent_folder_raises(self):
        with pytest.raises(FileNotFoundError):
            load_chapters_from_md_folder("/nonexistent/path/that/does/not/exist")

    def test_output_format(self, tmp_path):
        (tmp_path / "ch.md").write_text("# Title\nContent here.", encoding="utf-8")

        chapters = load_chapters_from_md_folder(str(tmp_path))

        assert len(chapters) == 1
        assert set(chapters[0].keys()) == {"title", "content"}
        assert isinstance(chapters[0]["title"], str)
        assert isinstance(chapters[0]["content"], str)


# === parse_markdown_chapters regression ===


class TestParseMarkdownChapters:
    def test_basic(self):
        md = "# Chapter One\nFirst content.\n\n# Chapter Two\nSecond content."
        chapters = parse_markdown_chapters(md)

        assert len(chapters) == 2
        assert chapters[0]["title"] == "Chapter One"
        assert "First content." in chapters[0]["content"]
        assert chapters[1]["title"] == "Chapter Two"
        assert "Second content." in chapters[1]["content"]
