FROM python:3.9-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=5

WORKDIR /app

# FFmpeg decodes uploaded audio; libsndfile and libgomp are used by the audio stack.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libsndfile1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN python -m pip install --upgrade pip==24.0 setuptools==70.3.0 wheel \
    && python -m pip install torch==2.0.1+cpu torchaudio==2.0.2+cpu --index-url https://download.pytorch.org/whl/cpu
RUN python -m pip install --no-build-isolation -r requirements.txt \
    && python -m pip install gunicorn==21.2.0 \
    && python -m pip check

COPY main.py ./
COPY utils/Models/ utils/Models/
COPY templates/ templates/
COPY static/theme.css static/theme.css
COPY static/audioVisualize/audio.png static/audioVisualize/audio.png
COPY TestAudios/conv0.wav TestAudios/conv0.wav
COPY TestAudios/conv1.wav TestAudios/conv1.wav
COPY TestAudios/conv3.wav TestAudios/conv3.wav
COPY utils/audioFiles/long_audio_files/2_Speakers_4mins.wav utils/audioFiles/long_audio_files/2_Speakers_4mins.wav
COPY utils/audioFiles/long_audio_files/4_Speaker_5mins.wav utils/audioFiles/long_audio_files/4_Speaker_5mins.wav
COPY utils/audioFiles/long_audio_files/5_Speaker_3mins.wav utils/audioFiles/long_audio_files/5_Speaker_3mins.wav

EXPOSE 7860

# One worker keeps the legacy model's shared intermediate files serialized.
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "1", "--threads", "1", "--timeout", "600", "--access-logfile", "-", "main:app"]
