"""
download_data.py

Ensures the "data" folder contains the pre-built search artifacts
(embeddings.db, index.faiss, metadata.json) before app.py / engine.py
start up. Does NOT regenerate or modify any of these files — it only
fetches the existing ones from Google Drive.

Behavior:
- Creates ./data if it doesn't exist.
- If ./data/embeddings.db already exists, does nothing (assumes the
  full set is already present) and exits successfully.
- Otherwise downloads embeddings.db, index.faiss, and metadata.json
  from Google Drive with streaming + progress output.
- Verifies every expected file exists on disk after downloading.
- Exits with a non-zero status code if any download or verification
  fails, so this can be safely used as a pre-start step (e.g. in a
  Render build/start command) that fails loudly instead of silently
  leaving the service without data.

Run with:
    python download_data.py
"""

import os
import sys
from pathlib import Path

import requests

DATA_DIR = Path('data')

# (filename, Google Drive file id)
FILES = [
    ('embeddings.db', '1-M0hxdpO4VI7Fk4sxm9MUXqodhqgMreU'),
    ('index.faiss', '1W0CfgJeS4BTMJawl1l7UsvIIt7foh51j'),
    ('metadata.json', '1psuF90DJcY-jQlqyk0oILExBegLMVRHI'),
]

GOOGLE_DRIVE_URL = 'https://drive.google.com/uc?export=download'
CHUNK_SIZE = 1 << 20  # 1 MB


def _get_confirm_token(response: requests.Response) -> str:
    """
    Google Drive serves an HTML "can't scan this file for viruses"
    interstitial for large files instead of the raw bytes, unless a
    confirm token from that page's cookies is replayed on a second
    request. Small files skip this entirely, so returning None is
    expected and fine for them.
    """
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None


def download_file(file_id: str, destination: Path) -> bool:
    """Streams a single file from Google Drive to `destination`,
    printing progress as it goes. Returns True on success, False on
    any failure (destination is left removed on failure so a partial
    file is never mistaken for a complete one)."""
    session = requests.Session()

    try:
        response = session.get(GOOGLE_DRIVE_URL, params={'id': file_id}, stream=True, timeout=30)
        response.raise_for_status()

        token = _get_confirm_token(response)
        if token:
            response = session.get(
                GOOGLE_DRIVE_URL,
                params={'id': file_id, 'confirm': token},
                stream=True,
                timeout=30,
            )
            response.raise_for_status()

        total_size = int(response.headers.get('Content-Length', 0))
        downloaded = 0

        print(f'Downloading {destination.name} ...')
        with open(destination, 'wb') as f:
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                if not chunk:
                    continue
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    pct = downloaded / total_size * 100
                    print(f'\r  {destination.name}: {downloaded / (1 << 20):.1f} MB '
                          f'/ {total_size / (1 << 20):.1f} MB ({pct:.1f}%)', end='')
                else:
                    print(f'\r  {destination.name}: {downloaded / (1 << 20):.1f} MB downloaded', end='')
        print()  # newline after the progress line

        return True

    except Exception as e:
        print(f'  ERROR downloading {destination.name}: {e}')
        if destination.exists():
            destination.unlink()
        return False


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    marker_file = DATA_DIR / 'embeddings.db'
    if marker_file.exists():
        print(f'{marker_file} already exists — skipping download.')
        sys.exit(0)

    print(f'{marker_file} not found — downloading data files into "{DATA_DIR}/" ...')

    all_ok = True
    for filename, file_id in FILES:
        destination = DATA_DIR / filename
        ok = download_file(file_id, destination)
        all_ok = all_ok and ok

    print('\nVerifying downloaded files ...')
    missing = []
    for filename, _ in FILES:
        path = DATA_DIR / filename
        if path.exists() and path.stat().st_size > 0:
            print(f'  OK   {path} ({path.stat().st_size / (1 << 20):.2f} MB)')
        else:
            print(f'  MISSING or empty: {path}')
            missing.append(filename)

    if not all_ok or missing:
        print('\nFAILED: one or more data files could not be downloaded/verified:',
              missing if missing else '(see errors above)')
        sys.exit(1)

    print('\nAll data files downloaded and verified successfully.')
    sys.exit(0)


if __name__ == '__main__':
    main()
