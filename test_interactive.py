"""Tests for interactive.py — the wizard that guides users through input selection."""

from unittest.mock import patch, call

import pytest

from interactive import run_wizard


class TestWizardInputSelection:
    @patch("builtins.input", side_effect=["1", "/path/to/book.epub"])
    def test_wizard_select_epub(self, mock_input):
        config = run_wizard()

        assert config["input_type"] == "epub"
        assert config["path"] == "/path/to/book.epub"

    @patch("builtins.input", side_effect=["2", "/path/to/md_folder", "My Book", "Author Name"])
    def test_wizard_select_md_folder(self, mock_input):
        config = run_wizard()

        assert config["input_type"] == "md_folder"
        assert config["path"] == "/path/to/md_folder"
        assert config["title"] == "My Book"
        assert config["author"] == "Author Name"

    @patch("builtins.input", side_effect=["3", "/path/to/prompts"])
    def test_wizard_select_generate(self, mock_input):
        config = run_wizard()

        assert config["input_type"] == "generate"
        assert config["path"] == "/path/to/prompts"


class TestWizardConfig:
    @patch("builtins.input", side_effect=["1", "/path/to/book.epub"])
    def test_returns_correct_keys(self, mock_input):
        config = run_wizard()

        assert "input_type" in config
        assert "path" in config

    @patch("builtins.input", side_effect=["2", "/folder", "Title", "Author"])
    def test_md_folder_has_metadata(self, mock_input):
        config = run_wizard()

        assert config["title"] == "Title"
        assert config["author"] == "Author"

    @patch("builtins.input", side_effect=["2", "/folder", "", ""])
    def test_md_folder_empty_metadata_uses_defaults(self, mock_input):
        config = run_wizard()

        assert config["title"] is not None
        assert config["author"] is not None


class TestWizardInputValidation:
    @patch("builtins.input", side_effect=["invalid", "1", "/path/to/book.epub"])
    def test_invalid_choice_reprompts(self, mock_input):
        config = run_wizard()

        assert config["input_type"] == "epub"
        # Should have been called 3 times: bad choice, good choice, path
        assert mock_input.call_count == 3
