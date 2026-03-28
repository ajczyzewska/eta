// ── Types ────────────────────────────────────────────────────────────────────
// ── Constants ────────────────────────────────────────────────────────────────
const RESUME_KEY = 'audiobook-resume';
const AUDIOBOOKS_BASE = '/audiobooks';
// ── Player ───────────────────────────────────────────────────────────────────
export class Player {
    constructor() {
        this._book = null;
        this._chapterIndex = 0;
        this._isBuffering = false;
        this._hasError = false;
        this.audio = new Audio();
        this.audio.preload = 'auto';
        this.setupListeners();
    }
    setupListeners() {
        this.audio.addEventListener('ended', () => { this.playNext(); });
        this.audio.addEventListener('timeupdate', () => {
            this.saveResume();
            this.emit();
        });
        this.audio.addEventListener('play', () => { this.emit(); });
        this.audio.addEventListener('pause', () => { this.emit(); });
        this.audio.addEventListener('waiting', () => {
            this._isBuffering = true;
            this.emit();
        });
        this.audio.addEventListener('canplaythrough', () => {
            this._isBuffering = false;
            this.emit();
        });
        this.audio.addEventListener('durationchange', () => { this.emit(); });
        this.audio.addEventListener('error', () => {
            this._isBuffering = false;
            this._hasError = true;
            this.emit();
        });
    }
    emit() {
        this.onStateChange?.(this.getState());
    }
    getState() {
        return {
            book: this._book,
            chapterIndex: this._chapterIndex,
            isPlaying: !this.audio.paused,
            isBuffering: this._isBuffering,
            error: this._hasError,
            currentTime: this.audio.currentTime,
            duration: isFinite(this.audio.duration) ? this.audio.duration : 0,
        };
    }
    loadBook(book, chapterIndex = 0, resumeTime = 0) {
        this._book = book;
        this._chapterIndex = Math.max(0, Math.min(chapterIndex, book.chapters.length - 1));
        this.loadChapter(resumeTime);
    }
    loadChapter(resumeTime = 0) {
        if (!this._book)
            return;
        const chapter = this._book.chapters[this._chapterIndex];
        if (!chapter)
            return;
        this.audio.src = `${AUDIOBOOKS_BASE}/${this._book.slug}/${chapter.filename}`;
        this.audio.currentTime = resumeTime;
        this._isBuffering = false;
        this._hasError = false;
        this.emit();
    }
    /**
     * Jump to a chapter by index. Does not auto-play; caller decides.
     */
    jumpToChapter(index) {
        if (!this._book)
            return;
        this._chapterIndex = Math.max(0, Math.min(index, this._book.chapters.length - 1));
        this.loadChapter(0);
    }
    play() {
        void this.audio.play();
    }
    pause() {
        this.audio.pause();
    }
    togglePlay() {
        if (this.audio.paused) {
            this.play();
        }
        else {
            this.pause();
        }
    }
    seekBy(seconds) {
        const dur = isFinite(this.audio.duration) ? this.audio.duration : 0;
        this.audio.currentTime = Math.max(0, Math.min(this.audio.currentTime + seconds, dur));
    }
    seekTo(fraction) {
        if (isFinite(this.audio.duration)) {
            this.audio.currentTime = Math.max(0, Math.min(fraction, 1)) * this.audio.duration;
        }
    }
    playNext() {
        if (!this._book)
            return;
        if (this._chapterIndex < this._book.chapters.length - 1) {
            this._chapterIndex++;
            this.loadChapter(0);
            void this.audio.play();
        }
        else {
            this.onEndOfBook?.();
        }
    }
    playPrev() {
        if (!this._book)
            return;
        // Restart current chapter if more than 5 s in; otherwise go to previous
        if (this.audio.currentTime > 5) {
            this.audio.currentTime = 0;
        }
        else if (this._chapterIndex > 0) {
            this._chapterIndex--;
            this.loadChapter(0);
            void this.audio.play();
        }
        else {
            this.audio.currentTime = 0;
        }
    }
    saveResume() {
        if (!this._book)
            return;
        const state = {
            slug: this._book.slug,
            chapterIndex: this._chapterIndex,
            currentTime: this.audio.currentTime,
        };
        try {
            localStorage.setItem(RESUME_KEY, JSON.stringify(state));
        }
        catch {
            // Storage unavailable (e.g., private browsing, storage quota)
        }
    }
    loadResume() {
        try {
            const raw = localStorage.getItem(RESUME_KEY);
            if (!raw)
                return null;
            const parsed = JSON.parse(raw);
            if (typeof parsed === 'object' &&
                parsed !== null &&
                typeof parsed['slug'] === 'string' &&
                typeof parsed['chapterIndex'] === 'number' &&
                typeof parsed['currentTime'] === 'number') {
                return parsed;
            }
            return null;
        }
        catch {
            return null;
        }
    }
}
