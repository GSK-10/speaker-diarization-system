# Speaker Diarization and Conversation Sentiment

This project turns a spoken conversation into a transcript grouped by estimated speaker, with sentiment labels for each speech segment, each speaker, and the conversation overall. It is a local research/demo application built around **English audio with 2–5 speakers**.

The work concentrates on joining several tasks in one usable flow: accepting an audio file or browser recording, transcribing speech, assigning speaker labels, and showing simple sentiment results in a Flask interface. Speaker labels are estimates (`SPEAKER-1`, `SPEAKER-2`, and so on), not names or verified identities. The sentiment labels are rules derived from VADER scores; they are not an emotion recognition model.

## What it uses

| Part | Technology | Role |
| --- | --- | --- |
| Web interface | Flask, Jinja templates, HTML, CSS, Bootstrap | Upload, browser recording, sample downloads, and results |
| Transcription | OpenAI Whisper `base.en` | English speech to text with timestamps |
| Speaker features | SpeechBrain ECAPA-TDNN through `pyannote.audio` | Voice embeddings for speech segments |
| Speaker grouping | scikit-learn agglomerative clustering | Groups segments into the requested number of speakers |
| Sentiment | VADER Sentiment | Scores transcript text and maps scores to display labels |
| Audio conversion | FFmpeg, SciPy, torchaudio | Decodes uploads and prepares WAV audio |
| Local packaging | Python 3.9 virtual environment or Docker | Reproducible ways to start the app |

Model weights download on first use, so the first start needs internet access and can take longer. Processing runs on CPU in the current code.

## Project layout

Paths below use Linux-style `/` separators; the same layout works on Windows.

```text
speaker-diarization/
├── main.py                         # Flask routes and processing flow
├── requirements.txt                # Python dependencies
├── setup_git_bash.sh               # Python 3.9 setup on the host
├── Dockerfile                      # Linux container image
├── .dockerignore                   # Keeps local and historical files out of the image
├── templates/                      # Upload, recording, and results pages
├── static/
│   ├── theme.css                   # Shared color variables and responsive layout
│   ├── audioVisualize/             # Existing interface image
│   ├── uploads/                    # Generated uploads; ignored by Git
│   └── generalAudioAndOutput/      # Generated intermediate audio; ignored by Git
├── utils/
│   ├── Models/                      # Diarization, recording, and sentiment code
│   └── audioFiles/long_audio_files/ # Long downloadable samples
├── TestAudios/                    # Short samples and reference transcripts
├── test_accuracy_commented.ipynb # Evaluation notebook
└── research-notebooks/           # Earlier notebooks, outputs, and notes
```

The Docker image copies the interface and all six samples offered on the upload page. The original page structure remains in the HTML templates; `static/theme.css` supplies the dark palette, angular controls, and responsive layout. The research notebooks remain in the project directory for reference and are excluded from Docker. `conv2.wav` and `conv4.wav` remain for the accuracy notebook, but are not offered in the app or copied into the image.

### Change the colors

Edit the variables at the top of `static/theme.css`. `--color-page-start` and `--color-page-end` set the background, `--color-primary` and `--color-primary-soft` set the purple gradient, `--color-surface` sets the cards, and `--color-speaker-1` through `--color-speaker-5` set the transcript label colors. The templates use these variables, so one edit updates matching colors across all pages.

The project background and published paper are available at the [SSRG research article](https://www.internationaljournalssrg.org/IJEEE/paper-details?Id=1043).

## Use the app

| Action | Where | What to do |
| --- | --- | --- |
| Try a short example | Upload page → **Try a sample** | Choose one of three short conversations, download it, upload it, and select **2** speakers. |
| Try a long example | Upload page → **Try a sample** | Choose a 2-, 4-, or 5-speaker recording, download it, and select its listed speaker count. Expect a longer CPU run. |
| Analyze your file | Upload page | Upload WAV, MP3, WebM, M4A, or MP4 audio, choose **2–5** speakers, then submit. The limit is **100 MB**. |
| Record in the browser | **Record audio** page | Allow microphone access, choose a duration and speaker count, then record. The browser sends the clip to the upload flow. |
| Read results | Results page | Review the transcript, estimated speaker labels, timestamps, and sentiment summaries; play the uploaded clip if the browser supports its format. |
| Keep results | Results page → **Download results (.txt)** | Save the transcript and overall and speaker-wise sentiment before leaving the page. Results are not saved as a text file on the server. |

Choose a speaker count no greater than the number of speech segments Whisper finds. Very short, silent, or unclear clips can fail to produce enough segments. `base.en` is an English model.

## Run locally with Git Bash

This path has been checked on Windows with **Python 3.9** and **FFmpeg** available on `PATH`. Docker offers a Linux environment below.

1. Install Python 3.9 and FFmpeg. In Git Bash, confirm both are visible:

   ```bash
   python --version
   ffmpeg -version
   ```

2. From the project root, create the virtual environment, install the pinned requirements, check dependencies, and cache the two models:

   ```bash
   bash setup_git_bash.sh
   ```

   If `python` points to another version, set `PYTHON_BIN` to a Python 3.9 executable before running the script.

   `setup_git_bash.sh` is an optional convenience script added during the local setup repair. Git itself does not require it, and Docker does not use it.

3. Start Flask:

   ```bash
   .venv/Scripts/python.exe main.py
   ```

4. Open <http://127.0.0.1:5000>. Stop the server with `Ctrl+C`.

On Linux, the setup script uses `.venv/bin/python`; start the app with `.venv/bin/python main.py`. The host setup has been verified on Windows, while the Docker image provides the intended Linux run path.

## Run in Docker Desktop

Start Docker Desktop with its **Linux engine** running. From the project root:

```bash
docker build -t speaker-diarization .
docker run --rm --name speaker-diarization -p 7860:7860 speaker-diarization
```

Open <http://localhost:7860>. Docker installs Python 3.9, FFmpeg, CPU-only PyTorch wheels, and the pinned Python packages, then starts one Gunicorn worker. A single worker matters because the current diarization code shares intermediate filenames and model state. The first container start needs internet access to download the model weights. Stop it with `Ctrl+C`.

The built image was about 3.5 GB on the tested Docker Desktop setup because PyTorch and audio processing libraries are large. Container-local uploads, generated results, and model caches are temporary unless you attach storage. Browser recording uses your browser's microphone; the older server-side microphone route cannot access a microphone inside a normal container.

## Current scope and limitations

- The short sample has been processed successfully both in the Windows Python 3.9 environment and through the local Linux Docker container. The container returned HTTP 200 with speaker labels; the long sample download also returned HTTP 200. Long-sample processing time has not been measured.
- Diarization relies on a selected speaker count and Whisper's speech segments. Labels and transcript accuracy vary with the audio.
- The current app stores uploaded clips under Flask's public `static/` path and writes intermediate audio to fixed names. The results download is generated from the page and is not retained as a text file, but uploaded and intermediate audio can remain until removed. Run it locally for now; those paths need redesign before exposing it as a public service.
- This repository does not include downloaded model weights. Setup or first container start downloads them.
- Public hosting is intentionally out of scope for this version. A working local Docker image is the immediate deployment target; a later host would need storage, privacy, concurrency, resource, and sample-audio redistribution review.

## Repository status

The local Git repository is already initialized, so there is no need to run `git init` again. An `origin` remote is configured. Review the included audio and historical files before publishing, particularly their size and redistribution rights. Pushing and public hosting are left to the repository owner.
