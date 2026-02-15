# Quick Start Guide

## 🚀 5-minutowy start

### 1. Sklonuj repo i przejdź do folderu

```bash
git clone <your-repo-url>
cd EtA
```

### 2. Wymagania wstępne

- **Python 3.9-3.11** (nie kompatybilne z Python 3.12+)
- **FFmpeg** - wymagany do eksportu MP3:
  ```bash
  # macOS
  brew install ffmpeg
  # Ubuntu/Debian
  sudo apt-get install ffmpeg
  ```

### 3. Setup środowiska

```bash
# Utwórz wirtualne środowisko
python3 -m venv .venv

# Aktywuj środowisko
source .venv/bin/activate  # Linux/Mac
# lub
.venv\Scripts\activate     # Windows

# Zaktualizuj pip (ważne - stary pip może mieć problemy z pobieraniem dużych pakietów)
pip install --upgrade pip

# Zainstaluj zależności
pip install -r requirements.txt
```

### 4. Przygotuj głos (opcjonalnie)

Jeśli chcesz użyć własnego głosu:
- Nagraj 10-30 sekund czystego nagrania głosu
- Zapisz jako `speaker.wav` w katalogu projektu

Lub użyj domyślnego głosu `sample-agent.wav`

### 5. Konwertuj książkę!

```bash
python epub_to_audiobook.py twoja_książka.epub
```

To wszystko! 🎉

## 📦 Co otrzymasz?

```
twoja_książka_audio/
├── 01_Rozdzial_1.mp3
├── 02_Rozdzial_2.mp3
├── 03_Rozdzial_3.mp3
└── ...
```

## 🎛️ Podstawowe opcje

```bash
# Z własnym głosem
python epub_to_audiobook.py książka.epub --speaker moj_glos.wav

# Większe fragmenty = płynniejsza mowa
python epub_to_audiobook.py książka.epub --chunk-size 400

# Zobacz co zostało pominięte
python epub_to_audiobook.py książka.epub --verbose
```

## ❓ Problemy?

### "Model załadowany na CPU" - wolno działa
- ✅ To normalne, przetwarzanie na CPU trwa dłużej
- 💡 Użyj GPU z CUDA dla 10x szybszego przetwarzania

### Ostrzeżenie o limicie 224 znaków
- ✅ Możesz zignorować - model radzi sobie z dłuższymi tekstami
- 🔍 Sprawdź czy wygenerowane audio jest kompletne
- 📝 Jeśli ucięte, użyj: `--chunk-size 200`

### Przerwałem proces
- ✅ Użyj `--resume` aby wznowić od ostatniego checkpointu
```bash
python epub_to_audiobook.py książka.epub --resume
```

## 📚 Więcej info

- Pełna dokumentacja: [README.md](README.md)
- Zgłoś problem: [Issues](../../issues)
