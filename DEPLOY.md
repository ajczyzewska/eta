# mydevil Server Setup Guide

One-time setup to get the audiobook player running on mydevil.net. After this, generating and deploying a new book is a single command.

---

## Prerequisites

- SSH access to your mydevil account
- Node.js 18+ on your local machine (for PWA build)
- An SSH key added to mydevil panel (Settings → SSH Keys)

---

## Step 1 — Create server directories

SSH into mydevil and create the directories that nginx will serve:

```bash
ssh eta@mydevil.net
mkdir -p /srv/audiobooks
mkdir -p /srv/player
exit
```

> Replace `eta` with your actual mydevil username throughout this guide.

---

## Step 2 — Configure nginx

On mydevil, nginx config is edited via the panel or by editing the vhost config file directly. Add the contents of [nginx/audiobooks.conf](nginx/audiobooks.conf) inside your existing `server { }` block.

**Via mydevil panel:**
1. Panel → WWW → your domain → nginx config
2. Paste the location blocks from `nginx/audiobooks.conf` inside the `server { }` block
3. Save and reload nginx (panel usually does this automatically)

**Verify:**
```bash
curl -I https://yourdomain.net/audiobooks/manifest.json
# Should return 404 (file doesn't exist yet) — not 502 or connection refused
```

---

## Step 3 — Configure deploy.sh

Create a `.env` file in the project root (it's gitignored):

```bash
# .env
SSH_USER=eta
SSH_HOST=yourdomain.net
REMOTE_AUDIOBOOKS_DIR=/srv/audiobooks
```

Or just edit the defaults at the top of [deploy.sh](deploy.sh) directly.

**Test SSH connectivity:**
```bash
ssh eta@yourdomain.net echo "OK"
```

If this asks for a password, set up SSH key auth first:
```bash
ssh-copy-id eta@yourdomain.net
```

---

## Step 4 — Build and deploy the PWA

```bash
cd pwa
npm install
npm run build
rsync -av dist/ eta@yourdomain.net:/srv/player/
```

**Verify:**
```bash
curl https://yourdomain.net/player/
# Should return the HTML of the player app
```

---

## Step 5 — Test end-to-end with a dry run

```bash
bash deploy.sh generated_books/your-book-dir/audio \
  --name test-book \
  --title "Test Book" \
  --author "Author Name" \
  --dry-run
```

Check the output: it should show which files would be transferred and preview the generated `meta.json`.

---

## Step 6 — Deploy your first book

```bash
bash deploy.sh generated_books/your-book-dir/audio \
  --name rod-debickich-t5 \
  --title "Praca i nadzieja" \
  --author "Stanisław Modrzejewski" \
  --series "Ród Dębickich" \
  --tom 5
```

**Verify:**
```bash
curl https://yourdomain.net/audiobooks/manifest.json
# Should contain your book with chapters listed
```

Open `https://yourdomain.net/player/` in Chrome on the tablet — the book should appear.

---

## Step 7 — Add to home screen on the tablet (Android)

1. Open Chrome → navigate to `https://yourdomain.net/player/`
2. Tap the three-dot menu → "Add to Home screen"
3. Confirm

This makes it full-screen and gives Media Session (headphone controls) the most reliable behaviour. Your grandma should always open the app this way, not through the browser.

---

## Ongoing workflow

After generating a new book with `generatebook.sh`, deploy it with:

```bash
bash generatebook.sh your-book.epub --deploy
# or directly:
bash deploy.sh generated_books/<book-dir>/audio --name <slug> --title "..." --author "..."
```

The PWA automatically shows the new book on next open — no manual steps on the tablet side.

---

## Troubleshooting

**rsync hangs or times out**
- Check SSH key is added to mydevil panel
- Try `ssh -v eta@yourdomain.net` to debug

**manifest.json not updating**
- Re-run `deploy.sh` — it regenerates and rsyncs the manifest on every deploy
- Check cache: `curl -H "Cache-Control: no-cache" https://yourdomain.net/audiobooks/manifest.json`

**PWA shows blank page**
- Check browser console for errors
- Verify nginx is serving `/player/` correctly: `curl -I https://yourdomain.net/player/`
- Re-run `npm run build && rsync -av dist/ eta@yourdomain.net:/srv/player/`

**No sound / can't find book**
- Open `https://yourdomain.net/audiobooks/manifest.json` directly in the tablet browser to verify it contains the book
- Check that MP3 files exist: `ssh eta@yourdomain.net ls /srv/audiobooks/<slug>/`
