#!/usr/bin/env bash
set -euo pipefail

# Run from the project directory with Git Bash: bash setup_git_bash.sh
cd "$(dirname "$0")"
PYTHON_BIN="${PYTHON_BIN:-python}"

if ! "$PYTHON_BIN" -c 'import sys; sys.exit(sys.version_info[:2] != (3, 9))'; then
    echo "Python 3.9 is required. Set PYTHON_BIN to the Python 3.9 executable." >&2
    exit 1
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "FFmpeg is required on PATH before running setup." >&2
    exit 1
fi

if [[ ! -d .venv ]]; then
    "$PYTHON_BIN" -m venv .venv
fi
if [[ -x .venv/Scripts/python.exe ]]; then
    VENV_PY=.venv/Scripts/python.exe
else
    VENV_PY=.venv/bin/python
fi

"$VENV_PY" -m pip install --upgrade pip==24.0 setuptools==70.3.0 wheel
"$VENV_PY" -m pip install --no-build-isolation -r requirements.txt
"$VENV_PY" -m pip check

# These public models are cached outside the repository for normal use.
"$VENV_PY" -c 'import whisper; whisper.load_model("base.en"); print("Whisper model ready")'
"$VENV_PY" -c 'import torch; from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding; PretrainedSpeakerEmbedding("speechbrain/spkrec-ecapa-voxceleb", device=torch.device("cpu")); print("Speaker model ready")'

echo "Setup complete. Run: $VENV_PY main.py"
