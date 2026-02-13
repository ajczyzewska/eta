# Changelog

All notable changes to this project will be documented in this file.

## [1.0.0] - 2024-11-25

### Added
- Initial release
- EPUB to audiobook conversion with XTTS v2
- Automatic content filtering (covers, prefaces, TOC, page numbers)
- Voice cloning from custom WAV samples
- Checkpoint system for resume capability
- Crossfade between audio chunks for smooth transitions
- GPU acceleration support
- Polish language support
- Configurable chunk size (default 300 characters)
- Verbose mode to show skipped content
- Metadata extraction (title, author)

### Features
- ✅ Smart chapter detection
- ✅ Automatic page number removal
- ✅ Sentence-based chunk sizes (~300 chars) for natural speech flow
- ✅ Crossfade between chunks (100ms default)
- ✅ GPU auto-detection
- ✅ Resume from checkpoint after interruption
- ✅ Progress tracking with Rich console
- ✅ MP3 output at 192kbps

### Technical
- Uses XTTS v2 multilingual model
- Supports streaming for texts longer than model limit
- Hierarchical text splitting (paragraphs → sentences → words)
- BeautifulSoup for HTML parsing from EPUB
- Pydub for audio processing
- Rich for beautiful CLI output
