"""
colab_launcher.py

Colab-only entrypoint. Mounts Google Drive, points engine.py at the
Drive project folder, starts app.py's FastAPI instance with Uvicorn in
a background thread, and exposes it publicly via Cloudflare Tunnel —
exactly as before, just moved out of notebook cells into one script.

Run with:
    python colab_launcher.py
"""

import os
import sys
import subprocess
import threading
import time
import re

# ---------------------------------------------------------------------------
# 1) Point engine.py at the Google Drive project folder BEFORE importing
#    it (engine.py reads DATA_DIR at import time).
# ---------------------------------------------------------------------------
try:
    from google.colab import drive
    drive.mount('/content/drive')
except ImportError:
    print('google.colab not available — are you running this outside Colab? '
          'For local/Render use, run `uvicorn app:app` instead of this script.')
    sys.exit(1)

os.environ['DATA_DIR'] = '/content/drive/MyDrive/JewelrySearchEngine'

# ---------------------------------------------------------------------------
# 2) Ensure Colab-only runtime dependencies are present.
# ---------------------------------------------------------------------------
def _ensure_installed(pip_name: str, import_name: str = None):
    import_name = import_name or pip_name
    try:
        __import__(import_name)
    except ImportError:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', pip_name], check=True)

_ensure_installed('nest_asyncio')
_ensure_installed('uvicorn')

import nest_asyncio
import uvicorn

nest_asyncio.apply()

# ---------------------------------------------------------------------------
# 3) Download cloudflared if not already present.
# ---------------------------------------------------------------------------
CLOUDFLARED_PATH = './cloudflared'
if not os.path.exists(CLOUDFLARED_PATH):
    subprocess.run([
        'wget', '-q',
        'https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64',
        '-O', CLOUDFLARED_PATH
    ], check=True)
    os.chmod(CLOUDFLARED_PATH, 0o755)

# ---------------------------------------------------------------------------
# 4) Import the FastAPI app (this triggers engine.py to load the model,
#    FAISS index, and metadata from DATA_DIR set above).
# ---------------------------------------------------------------------------
from app import app  # noqa: E402  (import after DATA_DIR is set on purpose)

# ---------------------------------------------------------------------------
# 5) Run Uvicorn in a background thread so this script stays interactive
#    inside a notebook, and start the Cloudflare Tunnel.
# ---------------------------------------------------------------------------
API_PORT = int(os.environ.get('PORT', 8000))

def _run_server():
    uvicorn.run(app, host='0.0.0.0', port=API_PORT, log_level='info')

def main():
    server_thread = threading.Thread(target=_run_server, daemon=True)
    server_thread.start()
    time.sleep(3)
    print(f'FastAPI server running locally on http://localhost:{API_PORT}')

    tunnel = subprocess.Popen(
        [CLOUDFLARED_PATH, 'tunnel', '--url', f'http://localhost:{API_PORT}'],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )

    for line in tunnel.stdout:
        print(line, end='')
        match = re.search(r'https://[a-zA-Z0-9\-]+\.trycloudflare\.com', line)
        if match:
            public_url = match.group(0)
            print('\n\n✅ Public API URL:', public_url)
            print('   Root       :', public_url + '/')
            print('   Health     :', public_url + '/health')
            print('   Stats      :', public_url + '/stats')
            print('   Categories :', public_url + '/categories')
            print('   Metals     :', public_url + '/metals')
            print('   Search     :', public_url + '/search  (POST, multipart/form-data: image, top_k, threshold)')
            break

    # Keep the process alive so the server + tunnel keep running.
    try:
        server_thread.join()
    except KeyboardInterrupt:
        tunnel.terminate()

if __name__ == '__main__':
    main()
