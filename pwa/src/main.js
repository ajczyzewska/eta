import { Player } from './player';
import { setupMediaSession, updateMediaMetadata } from './mediaSession';
// In production, /audiobooks/ is served by nginx independently of /player/.
// In dev, Vite serves public/ under the configured base (/player/), so the
// mock file at public/audiobooks/manifest.json lives at /player/audiobooks/manifest.json.
const MANIFEST_URL = import.meta.env.DEV
    ? `${import.meta.env.BASE_URL}audiobooks/manifest.json`
    : '/audiobooks/manifest.json';
// ── Helpers ──────────────────────────────────────────────────────────────────
function formatTime(seconds) {
    const s = Math.floor(seconds);
    const m = Math.floor(s / 60);
    const h = Math.floor(m / 60);
    if (h > 0) {
        return `${h}:${String(m % 60).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`;
    }
    return `${m}:${String(s % 60).padStart(2, '0')}`;
}
function bookLabel(book) {
    if (book.series != null && book.tom != null) {
        return `${book.series} T${book.tom}`;
    }
    return book.title;
}
async function fetchManifest() {
    const resp = await fetch(MANIFEST_URL, { cache: 'no-cache' });
    if (!resp.ok)
        throw new Error(`HTTP ${resp.status}`);
    return (await resp.json());
}
// ── UI initialisation ─────────────────────────────────────────────────────────
function initUI(manifest) {
    const books = manifest.books;
    const player = new Player();
    // ── DOM refs ──────────────────────────────────────────────────
    const loading = document.getElementById('loading');
    const app = document.getElementById('app');
    const bookTitleEl = document.getElementById('book-title');
    const chapterTitleEl = document.getElementById('chapter-title');
    const progressFill = document.getElementById('progress-fill');
    const progressSeek = document.getElementById('progress-seek');
    const currentTimeEl = document.getElementById('current-time');
    const durationEl = document.getElementById('duration');
    const btnPlayPause = document.getElementById('btn-play-pause');
    const btnSeekBack = document.getElementById('btn-seek-back');
    const btnSeekFwd = document.getElementById('btn-seek-fwd');
    const btnPrevChapter = document.getElementById('btn-prev-chapter');
    const btnNextChapter = document.getElementById('btn-next-chapter');
    const spinner = document.getElementById('spinner');
    const booksList = document.getElementById('books-list');
    const chaptersList = document.getElementById('chapters-list');
    loading.classList.add('hidden');
    app.classList.remove('hidden');
    // ── UI state ──────────────────────────────────────────────────
    let selectedBook = null;
    let isSeeking = false;
    let lastChapterIndex = -1;
    let lastBookSlug = null;
    // ── Render helpers ────────────────────────────────────────────
    function renderBooks() {
        booksList.innerHTML = '';
        books.forEach(book => {
            const li = document.createElement('li');
            li.textContent = bookLabel(book);
            if (selectedBook?.slug === book.slug)
                li.classList.add('active');
            li.addEventListener('click', () => { selectBook(book); });
            booksList.appendChild(li);
        });
    }
    function renderChapters(activeIndex) {
        if (!selectedBook) {
            chaptersList.innerHTML = '';
            return;
        }
        chaptersList.innerHTML = '';
        selectedBook.chapters.forEach((chapter, i) => {
            const li = document.createElement('li');
            li.textContent = chapter.label;
            if (i === activeIndex)
                li.classList.add('active');
            li.addEventListener('click', () => {
                player.jumpToChapter(i);
                player.play();
            });
            chaptersList.appendChild(li);
        });
        scrollActiveChapter();
    }
    function updateChapterHighlight(index) {
        const items = chaptersList.querySelectorAll('li');
        items.forEach((li, i) => { li.classList.toggle('active', i === index); });
        scrollActiveChapter();
    }
    function scrollActiveChapter() {
        chaptersList.querySelector('.active')
            ?.scrollIntoView({ block: 'nearest' });
    }
    // ── Book selection ────────────────────────────────────────────
    function selectBook(book, chapterIndex = 0, resumeTime = 0) {
        selectedBook = book;
        // Sync tracking vars before loadBook triggers emit(), so onStateChange
        // doesn't re-fire updateChapterHighlight redundantly.
        lastChapterIndex = chapterIndex;
        lastBookSlug = book.slug;
        renderBooks();
        renderChapters(chapterIndex);
        player.loadBook(book, chapterIndex, resumeTime);
    }
    // ── Media Session ─────────────────────────────────────────────
    setupMediaSession({
        play: () => { player.play(); },
        pause: () => { player.pause(); },
        nextTrack: () => { player.playNext(); },
        prevTrack: () => { player.playPrev(); },
        seekForward: () => { player.seekBy(30); },
        seekBackward: () => { player.seekBy(-30); },
    });
    // ── Cross-book auto-advance ───────────────────────────────
    player.onEndOfBook = () => {
        if (!selectedBook)
            return;
        const currentIndex = books.findIndex(b => b.slug === selectedBook.slug);
        if (currentIndex !== -1 && currentIndex < books.length - 1) {
            selectBook(books[currentIndex + 1], 0, 0);
            player.play();
        }
    };
    // ── Player state → UI ─────────────────────────────────────────
    player.onStateChange = (state) => {
        if (state.book) {
            bookTitleEl.textContent = bookLabel(state.book);
            const chapter = state.book.chapters[state.chapterIndex];
            chapterTitleEl.textContent = chapter?.label ?? '';
            // Update chapter highlight + media metadata only when chapter changes
            const chapterChanged = state.chapterIndex !== lastChapterIndex ||
                state.book.slug !== lastBookSlug;
            if (chapterChanged) {
                lastChapterIndex = state.chapterIndex;
                lastBookSlug = state.book.slug;
                updateChapterHighlight(state.chapterIndex);
                if (chapter)
                    updateMediaMetadata(state.book, chapter);
            }
        }
        else {
            bookTitleEl.textContent = '—';
            chapterTitleEl.textContent = '';
        }
        // Progress bar — skip updates while user is dragging the seek slider
        if (!isSeeking) {
            const fraction = state.duration > 0 ? state.currentTime / state.duration : 0;
            progressFill.style.width = `${fraction * 100}%`;
            progressSeek.value = String(fraction * 100);
        }
        currentTimeEl.textContent = formatTime(state.currentTime);
        durationEl.textContent = formatTime(state.duration);
        btnPlayPause.textContent = state.isPlaying ? '⏸ PAUSE' : '▶ PLAY';
        if (state.error) {
            spinner.textContent = 'Błąd odtwarzania — sprawdź połączenie';
            spinner.classList.remove('hidden');
        }
        else {
            spinner.textContent = 'Ładowanie audio...';
            spinner.classList.toggle('hidden', !state.isBuffering);
        }
    };
    // ── Controls ──────────────────────────────────────────────────
    btnPlayPause.addEventListener('click', () => { player.togglePlay(); });
    btnSeekBack.addEventListener('click', () => { player.seekBy(-30); });
    btnSeekFwd.addEventListener('click', () => { player.seekBy(30); });
    btnPrevChapter.addEventListener('click', () => { player.playPrev(); });
    btnNextChapter.addEventListener('click', () => { player.playNext(); });
    // Seek slider: visual update on drag, actual seek on release.
    // pointercancel handles OS-level interruptions (e.g., permission prompts on Android)
    // that would otherwise leave isSeeking stuck at true, freezing the progress bar.
    progressSeek.addEventListener('pointerdown', () => { isSeeking = true; });
    progressSeek.addEventListener('pointerup', () => {
        isSeeking = false;
        player.seekTo(Number(progressSeek.value) / 100);
    });
    progressSeek.addEventListener('pointercancel', () => { isSeeking = false; });
    progressSeek.addEventListener('input', () => {
        if (isSeeking) {
            progressFill.style.width = `${progressSeek.value}%`;
        }
    });
    // ── Resume / initial load ─────────────────────────────────────
    if (books.length === 0) {
        bookTitleEl.textContent = 'Brak dostępnych książek';
        return;
    }
    const resume = player.loadResume();
    if (resume) {
        const book = books.find(b => b.slug === resume.slug);
        if (book) {
            // Validate stored chapterIndex against actual chapter count — the book
            // may have been updated (chapters added/removed) since the resume was saved.
            const safeIndex = resume.chapterIndex < book.chapters.length ? resume.chapterIndex : 0;
            const safeTime = safeIndex === resume.chapterIndex && resume.currentTime >= 0
                ? resume.currentTime
                : 0;
            selectBook(book, safeIndex, safeTime);
            return;
        }
    }
    // No resume — load first book without auto-playing (browser policy requires
    // a user gesture for the first play)
    selectBook(books[0]);
}
// ── Bootstrap ─────────────────────────────────────────────────────────────────
async function main() {
    try {
        const manifest = await fetchManifest();
        initUI(manifest);
    }
    catch (err) {
        const loading = document.getElementById('loading');
        if (loading) {
            loading.textContent = `Błąd: ${err instanceof Error ? err.message : 'nieznany błąd'}`;
        }
    }
}
void main();
