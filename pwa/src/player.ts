// ── Types ────────────────────────────────────────────────────────────────────

export interface Chapter {
  index: number;
  filename: string;
  label: string;
}

export interface Book {
  slug: string;
  title: string;
  author: string;
  series?: string;
  tom?: number;
  generated_at: string;
  chapters: Chapter[];
}

export interface Manifest {
  books: Book[];
}

export interface ResumeState {
  slug: string;
  chapterIndex: number;
  currentTime: number;
}

export interface PlayerState {
  book: Book | null;
  chapterIndex: number;
  isPlaying: boolean;
  isBuffering: boolean;
  /** True when the audio element encountered a load/decode/network error. */
  error: boolean;
  currentTime: number;
  duration: number;
}

// ── Constants ────────────────────────────────────────────────────────────────

const RESUME_KEY = 'audiobook-resume';
const AUDIOBOOKS_BASE = '/audiobooks';

// ── Player ───────────────────────────────────────────────────────────────────

export class Player {
  private readonly audio: HTMLAudioElement;
  private _book: Book | null = null;
  private _chapterIndex = 0;
  private _isBuffering = false;
  private _hasError = false;

  onStateChange?: (state: PlayerState) => void;
  /** Called when the last chapter of the current book ends. */
  onEndOfBook?: () => void;

  constructor() {
    this.audio = new Audio();
    this.audio.preload = 'auto';
    this.setupListeners();
  }

  private setupListeners(): void {
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

  private emit(): void {
    this.onStateChange?.(this.getState());
  }

  getState(): PlayerState {
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

  loadBook(book: Book, chapterIndex = 0, resumeTime = 0): void {
    this._book = book;
    this._chapterIndex = Math.max(0, Math.min(chapterIndex, book.chapters.length - 1));
    this.loadChapter(resumeTime);
  }

  private loadChapter(resumeTime = 0): void {
    if (!this._book) return;
    const chapter = this._book.chapters[this._chapterIndex];
    if (!chapter) return;
    this.audio.src = `${AUDIOBOOKS_BASE}/${this._book.slug}/${chapter.filename}`;
    this.audio.currentTime = resumeTime;
    this._isBuffering = false;
    this._hasError = false;
    this.emit();
  }

  /**
   * Jump to a chapter by index. Does not auto-play; caller decides.
   */
  jumpToChapter(index: number): void {
    if (!this._book) return;
    this._chapterIndex = Math.max(0, Math.min(index, this._book.chapters.length - 1));
    this.loadChapter(0);
  }

  play(): void {
    void this.audio.play();
  }

  pause(): void {
    this.audio.pause();
  }

  togglePlay(): void {
    if (this.audio.paused) {
      this.play();
    } else {
      this.pause();
    }
  }

  seekBy(seconds: number): void {
    const dur = isFinite(this.audio.duration) ? this.audio.duration : 0;
    this.audio.currentTime = Math.max(0, Math.min(this.audio.currentTime + seconds, dur));
  }

  seekTo(fraction: number): void {
    if (isFinite(this.audio.duration)) {
      this.audio.currentTime = Math.max(0, Math.min(fraction, 1)) * this.audio.duration;
    }
  }

  playNext(): void {
    if (!this._book) return;
    if (this._chapterIndex < this._book.chapters.length - 1) {
      this._chapterIndex++;
      this.loadChapter(0);
      void this.audio.play();
    } else {
      this.onEndOfBook?.();
    }
  }

  playPrev(): void {
    if (!this._book) return;
    // Restart current chapter if more than 5 s in; otherwise go to previous
    if (this.audio.currentTime > 5) {
      this.audio.currentTime = 0;
    } else if (this._chapterIndex > 0) {
      this._chapterIndex--;
      this.loadChapter(0);
      void this.audio.play();
    } else {
      this.audio.currentTime = 0;
    }
  }

  private saveResume(): void {
    if (!this._book) return;
    const state: ResumeState = {
      slug: this._book.slug,
      chapterIndex: this._chapterIndex,
      currentTime: this.audio.currentTime,
    };
    try {
      localStorage.setItem(RESUME_KEY, JSON.stringify(state));
    } catch {
      // Storage unavailable (e.g., private browsing, storage quota)
    }
  }

  loadResume(): ResumeState | null {
    try {
      const raw = localStorage.getItem(RESUME_KEY);
      if (!raw) return null;
      const parsed = JSON.parse(raw) as unknown;
      if (
        typeof parsed === 'object' &&
        parsed !== null &&
        typeof (parsed as Record<string, unknown>)['slug'] === 'string' &&
        typeof (parsed as Record<string, unknown>)['chapterIndex'] === 'number' &&
        typeof (parsed as Record<string, unknown>)['currentTime'] === 'number'
      ) {
        return parsed as ResumeState;
      }
      return null;
    } catch {
      return null;
    }
  }
}
