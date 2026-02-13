from flask import Flask, render_template, request, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
import os
from pathlib import Path
import threading
import subprocess
import signal
import json

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
LOG_FILE = os.path.join(BASE_DIR, 'ui', 'run.log')
STATE_FILE = os.path.join(BASE_DIR, 'ui', 'state.json')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, 'ui', 'static'), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, 'ui', 'templates'), exist_ok=True)

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, 'ui', 'templates'),
    static_folder=os.path.join(BASE_DIR, 'ui', 'static'),
    static_url_path='/ui/static'
)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Global process holder
process_info = {
    'process': None,
    'thread': None,
}


def save_state(state: dict):
    try:
        with open(STATE_FILE, 'w') as f:
            json.dump(state, f)
    except Exception:
        pass


def load_state() -> dict:
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, 'r') as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def tail_log(lines: int = 200):
    if not os.path.exists(LOG_FILE):
        return ''
    try:
        with open(LOG_FILE, 'rb') as f:
            f.seek(0, os.SEEK_END)
            end = f.tell()
            size = 1024
            data = b''
            while end > 0:
                seek = max(0, end - size)
                f.seek(seek)
                chunk = f.read(end - seek)
                data = chunk + data
                end = seek
                if data.count(b'\n') > lines:
                    break
                size *= 2
            return '\n'.join(data.decode(errors='ignore').splitlines()[-lines:])
    except Exception:
        return ''


def run_conversion(cmd, cwd=None):
    # Compute output directory from epub path (cmd[2]) if possible
    output_dir = None
    try:
        epub_path = cmd[2]
        book_name = Path(epub_path).stem
        output_dir = str(Path(cwd or BASE_DIR) / f"{book_name}_audio")
    except Exception:
        output_dir = None

    save_state({'running': True, 'pid': None, 'cmd': cmd, 'output_dir': output_dir})
    with open(LOG_FILE, 'w') as logf:
        try:
            proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, cwd=cwd)
            process_info['process'] = proc
            save_state({'running': True, 'pid': proc.pid, 'cmd': cmd, 'output_dir': output_dir})
            proc.wait()
            rc = proc.returncode
            save_state({'running': False, 'pid': None, 'returncode': rc, 'output_dir': output_dir})
        except Exception as e:
            logf.write(f"Error starting process: {e}\n")
            save_state({'running': False, 'pid': None, 'error': str(e), 'output_dir': output_dir})


@app.route('/')
def index():
    state = load_state()
    return render_template('index.html', state=state)


@app.route('/start', methods=['POST'])
def start():
    # Prevent starting another process
    state = load_state()
    if state.get('running'):
        return jsonify({'error': 'A conversion is already running.'}), 409

    epub_file = request.files.get('epub_file')
    speaker_file = request.files.get('speaker_file')
    optimize = request.form.get('optimize')
    chunk_size = request.form.get('chunk_size')
    crossfade = request.form.get('crossfade')
    speed = request.form.get('speed')
    resume = request.form.get('resume') == 'on'

    if not epub_file:
        return jsonify({'error': 'Please upload an EPUB file'}), 400

    epub_name = secure_filename(epub_file.filename)
    epub_path = os.path.join(app.config['UPLOAD_FOLDER'], epub_name)
    epub_file.save(epub_path)

    speaker_path = None
    if speaker_file and speaker_file.filename:
        speaker_name = secure_filename(speaker_file.filename)
        speaker_path = os.path.join(app.config['UPLOAD_FOLDER'], speaker_name)
        speaker_file.save(speaker_path)

    # Build command to run the CLI script
    cmd = ['python3', os.path.join(BASE_DIR, 'epub_to_audiobook.py'), epub_path]
    if speaker_path:
        cmd += ['--speaker', speaker_path]
    if optimize:
        cmd += ['--optimize', optimize]
    if chunk_size:
        cmd += ['--chunk-size', chunk_size]
    if crossfade:
        cmd += ['--crossfade', crossfade]
    if speed:
        cmd += ['--speed', speed]
    if resume:
        cmd += ['--resume']

    # Start background thread to run conversion
    thread = threading.Thread(target=run_conversion, args=(cmd, BASE_DIR), daemon=True)
    process_info['thread'] = thread
    thread.start()

    return redirect(url_for('index'))


@app.route('/status')
def status():
    state = load_state()
    running = state.get('running', False)
    pid = state.get('pid')
    log = tail_log(200)

    output_dir = state.get('output_dir')
    checkpoint = None
    files = []
    output_exists = False

    if output_dir:
        try:
            outp = Path(output_dir)
            output_exists = outp.exists()
            # list audio files
            if outp.exists() and outp.is_dir():
                for ext in ('.mp3', '.wav'):
                    for p in outp.glob(f'*{ext}'):
                        files.append(str(p.name))

                # read checkpoint
                cp = outp / '.checkpoint.json'
                if cp.exists():
                    try:
                        with open(cp, 'r') as f:
                            checkpoint = json.load(f)
                    except Exception:
                        checkpoint = None
        except Exception:
            pass

    return jsonify({'running': running, 'pid': pid, 'log': log, 'output_dir': output_dir, 'output_exists': output_exists, 'files': files, 'checkpoint': checkpoint})


@app.route('/stop', methods=['POST'])
def stop():
    state = load_state()
    pid = state.get('pid')
    if pid:
        try:
            os.kill(pid, signal.SIGTERM)
            save_state({'running': False, 'pid': None, 'stopped': True})
            return jsonify({'stopped': True})
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    return jsonify({'error': 'No running process'}), 400


@app.route('/open_output', methods=['POST'])
def open_output():
    state = load_state()
    output_dir = state.get('output_dir')
    if not output_dir:
        return jsonify({'error': 'No output directory known'}), 400

    try:
        import platform
        system = platform.system()
        if system == 'Darwin':
            subprocess.Popen(['open', output_dir])
        elif system == 'Windows':
            os.startfile(output_dir)
        else:
            subprocess.Popen(['xdg-open', output_dir])
        return jsonify({'opened': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
