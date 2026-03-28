import type { Book, Chapter } from './player';

export interface MediaHandlers {
  play: () => void;
  pause: () => void;
  nextTrack: () => void;
  prevTrack: () => void;
  seekForward: () => void;
  seekBackward: () => void;
}

/**
 * Register Media Session action handlers once at startup.
 * The same handler references remain active for the lifetime of the page.
 */
export function setupMediaSession(handlers: MediaHandlers): void {
  if (!('mediaSession' in navigator)) return;

  navigator.mediaSession.setActionHandler('play', handlers.play);
  navigator.mediaSession.setActionHandler('pause', handlers.pause);
  navigator.mediaSession.setActionHandler('nexttrack', handlers.nextTrack);
  navigator.mediaSession.setActionHandler('previoustrack', handlers.prevTrack);
  navigator.mediaSession.setActionHandler('seekforward', handlers.seekForward);
  navigator.mediaSession.setActionHandler('seekbackward', handlers.seekBackward);
}

/**
 * Update the Now Playing metadata shown on the lock screen / notification shade.
 * Call this whenever the active chapter changes.
 */
export function updateMediaMetadata(book: Book, chapter: Chapter): void {
  if (!('mediaSession' in navigator)) return;

  navigator.mediaSession.metadata = new MediaMetadata({
    title: chapter.label,
    artist: book.author,
    album:
      book.series != null
        ? `${book.series} — Tom ${book.tom ?? ''}`
        : book.title,
  });
}
