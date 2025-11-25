#!/bin/bash
# Skrypt do inicjalizacji nowego repozytorium Git dla projektu EtA

echo "🚀 Inicjalizacja repozytorium Git dla EtA"
echo ""

# Sprawdź czy jesteśmy już w repo git
if [ -d ".git" ]; then
    echo "⚠️  To już jest repozytorium Git"
    echo "   Pomiń ten krok jeśli chcesz użyć istniejącego repo"
    read -p "   Czy chcesz kontynuować? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Inicjalizuj Git
echo "📦 Inicjalizacja Git..."
git init

# Dodaj wszystkie pliki
echo "➕ Dodawanie plików..."
git add .

# Pierwszy commit
echo "💾 Tworzenie pierwszego commita..."
git commit -m "Initial commit: EtA v1.0.0

- EPUB to audiobook converter with XTTS v2
- Smart content filtering
- Voice cloning support
- Checkpoint system
- GPU acceleration"

echo ""
echo "✅ Repozytorium Git zainicjalizowane!"
echo ""
echo "📌 Następne kroki:"
echo "   1. Utwórz nowe repo na GitHub/GitLab"
echo "   2. Dodaj remote:"
echo "      git remote add origin <your-repo-url>"
echo "   3. Wypchnij kod:"
echo "      git branch -M main"
echo "      git push -u origin main"
echo ""
