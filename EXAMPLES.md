# Przykłady użycia

## 📚 Podstawowe scenariusze

### 1. Najprostsze użycie (domyślny głos)

```bash
python epub_to_audiobook.py moja_książka.epub
```

**Co się stanie:**
- Użyje domyślnego głosu `sample-agent.wav`
- Fragmenty po 3000 znaków
- Crossfade 100ms między fragmentami
- Wyjście: `moja_książka_audio/`

---

### 2. Z własnym głosem

```bash
python epub_to_audiobook.py książka.epub --speaker głos_narratora.wav
```

**Wymagania dla pliku głosu:**
- Format: WAV (16kHz lub 22.05kHz)
- Długość: 10-30 sekund
- Jakość: Czyste nagranie bez szumów
- Zawartość: Jedna osoba mówiąca po polsku

---

### 3. Maksymalna płynność (duże fragmenty)

```bash
python epub_to_audiobook.py książka.epub \
    --chunk-size 5000 \
    --crossfade 150
```

**Efekt:**
- Dłuższe fragmenty = lepsza intonacja i naturalność
- Dłuższy crossfade = jeszcze płynniejsze przejścia
- ⚠️ Może być wolniejsze

---

### 4. Szybkie fragmenty (mniejsze pliki)

```bash
python epub_to_audiobook.py książka.epub \
    --chunk-size 1000 \
    --crossfade 50
```

**Efekt:**
- Szybsze generowanie
- Mniejsze zużycie pamięci
- Może być mniej płynne

---

### 5. Bez crossfade (ostre przejścia)

```bash
python epub_to_audiobook.py książka.epub --crossfade 0
```

**Kiedy użyć:**
- Chcesz zaoszczędzić czas procesowania
- Testujesz różne ustawienia
- Preferujesz wyraźne przerwy między fragmentami

---

### 6. Tryb verbose (diagnostyka)

```bash
python epub_to_audiobook.py książka.epub --verbose
```

**Co zobaczysz:**
```
📚 Wczytuję EPUB: książka.epub
   Tytuł: Harry Potter i Kamień Filozoficzny
   Autor: J.K. Rowling

🔍 Analizuję rozdziały...
   ✅ Znaleziono rozdziałów do przetworzenia: 17
   Pominięto elementów: 5

Pominięte elementy:
   - cover: Pominięto (okładka, wstęp, metadata lub zbyt krótkie)
   - copyright: Pominięto (okładka, wstęp, metadata lub zbyt krótkie)
   - toc: Pominięto (okładka, wstęp, metadata lub zbyt krótkie)
   ...
```

---

### 7. Wznowienie po przerwaniu

```bash
# Pierwsze uruchomienie (przerwane)
python epub_to_audiobook.py duża_książka.epub
# (Ctrl+C po 3 rozdziałach)

# Wznowienie od miejsca przerwania
python epub_to_audiobook.py duża_książka.epub --resume
```

**Jak to działa:**
- Automatyczne checkpointy po każdym rozdziale
- Zapisywane w `.checkpoint.json`
- Usuwane automatycznie po ukończeniu

---

### 8. Niestandardowy katalog wyjściowy

```bash
python epub_to_audiobook.py książka.epub --output ~/Audiobooki/Moja_Książka
```

---

### 9. Batch processing (wiele książek)

```bash
#!/bin/bash
# convert_all.sh

for book in *.epub; do
    echo "Konwertuję: $book"
    python epub_to_audiobook.py "$book" --speaker narrator.wav
    echo "---"
done
```

---

## 🎯 Scenariusze zaawansowane

### A. Optymalizacja dla GPU

```bash
# Sprawdź czy GPU jest wykrywane
python -c "import torch; print('GPU:', torch.cuda.is_available())"

# Jeśli True - skrypt automatycznie użyje GPU
python epub_to_audiobook.py książka.epub
```

---

### B. Debugowanie problemu z jakością

```bash
# Testuj różne chunk sizes
for size in 200 500 1000 2000 3000 5000; do
    python epub_to_audiobook.py test.epub \
        --chunk-size $size \
        --output test_$size
done

# Porównaj jakość audio
```

---

### C. Produkcyjny pipeline

```bash
#!/bin/bash
# production_convert.sh

BOOK="$1"
SPEAKER="${2:-narrator.wav}"

echo "🎙️  Produkcyjny pipeline dla: $BOOK"

# 1. Konwertuj z optymalnymi ustawieniami
python epub_to_audiobook.py "$BOOK" \
    --speaker "$SPEAKER" \
    --chunk-size 3000 \
    --crossfade 100 \
    --verbose 2>&1 | tee conversion.log

# 2. Sprawdź czy sukces
if [ $? -eq 0 ]; then
    echo "✅ Konwersja zakończona sukcesem"

    # 3. Opcjonalnie: normalizuj głośność
    # for file in ${BOOK%.epub}_audio/*.mp3; do
    #     ffmpeg-normalize "$file" -o "$file.normalized.mp3"
    # done
else
    echo "❌ Błąd podczas konwersji"
    exit 1
fi
```

---

## 💡 Tips & Tricks

### Najlepsze ustawienia dla różnych typów książek

**Powieści (fiction):**
```bash
--chunk-size 3000 --crossfade 100
```

**Podręczniki/Non-fiction:**
```bash
--chunk-size 2000 --crossfade 50
```

**Poezja:**
```bash
--chunk-size 500 --crossfade 200
```

**Biografie:**
```bash
--chunk-size 4000 --crossfade 150
```

---

### Szacowanie czasu konwersji

**CPU (typowy laptop):**
- ~20-30 sekund na fragment 3000 znaków
- Książka 300 stron (~500k znaków): ~2-3 godziny

**GPU (NVIDIA RTX):**
- ~2-3 sekundy na fragment 3000 znaków
- Książka 300 stron: ~15-20 minut

---

## 🐛 Rozwiązywanie problemów

### Problem: Audio jest ucięte

```bash
# Użyj mniejszych fragmentów
python epub_to_audiobook.py książka.epub --chunk-size 200
```

### Problem: Zbyt długi czas generowania

```bash
# Zmniejsz chunk size lub użyj GPU
python epub_to_audiobook.py książka.epub --chunk-size 1500
```

### Problem: Nie znaleziono rozdziałów

```bash
# Użyj verbose aby zobaczyć co zostało pominięte
python epub_to_audiobook.py książka.epub --verbose
```
