#!/bin/bash
# Przykładowe użycie konwertera EPUB -> Audiobook

echo "🎙️  EtA - EPUB to Audiobook Converter"
echo ""

# Sprawdź czy istnieje plik EPUB
if [ ! -f "example.epub" ]; then
    echo "❌ Błąd: Brak pliku example.epub"
    echo "   Umieść swój plik EPUB w katalogu projektu"
    echo ""
    echo "Użycie:"
    echo "  python epub_to_audiobook.py twoja_książka.epub"
    exit 1
fi

# Uruchom konwersję z domyślnymi ustawieniami
echo "🚀 Rozpoczynam konwersję..."
echo ""

python epub_to_audiobook.py example.epub \
    --chunk-size 3000 \
    --crossfade 100 \
    --verbose

echo ""
echo "✅ Gotowe! Sprawdź katalog z plikami audio."
