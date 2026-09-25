print("Started")

### import necessary modules

# whisper for speech recognition and speech-to-text-transcription
import whisper #https://openai.com/research/whisper

# Used for time calculations
import datetime

# Potentially used for external command execution
import subprocess

# Imports the PyTorch library (used by the speaker embedding model)
import torch

# Imports the speaker verification pipeline from PyAnnote for speaker identification -> For embeddings as well
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding

# Imports core PyAnnote audio functionalities
from pyannote.audio import Audio

# Imports core PyAnnote functionalities for audio segments
from pyannote.core import Segment

# Imports the Python wave module for audio file manipulation.
import wave

# Provides utilities for context managers, likely used for file handling
import contextlib
import io

# Imports the AgglomerativeClustering class from scikit-learn for speaker clustering.
from sklearn.cluster import AgglomerativeClustering

# Imports the NumPy library for numerical computations.
import numpy as np

# Used for reading and writing WAV audio files
from scipy.io import wavfile

# Imports tqdm for progress bars (used in terminal output for debugging)
from tqdm import tqdm

# importing categorize_sentiment() function from utils.Models.sentiment_analysis module for sentiment analysis.
from utils.Models.sentiment_analysis import categorize_sentiment

# variable defines Whisper model size to be used -> selected "base.en" for english base model
model_size = 'base.en' #['tiny.en', 'tiny', 'base.en', 'base', 'small.en', 'small', 'medium.en', 'medium', 'large-v1', 'large-v2', 'large-v3', 'large']

# store audio later
audio = ""

# defining duration variable
duration = 0

# debugging and logging purpose
print("Loading Embedding Model")

# loading the pretrained speaker embedding model based on the ECAPA-TDNN architecture and pre-trained on the VoxCeleb dataset
embedding_model = PretrainedSpeakerEmbedding("speechbrain/spkrec-ecapa-voxceleb",device=torch.device("cpu"))

# converts seconds to a datetime.timedelta object.
def time(secs):
    return f"{int(secs // 60):02d}:{secs % 60:04.1f}"

# generating a speaker embedding for a given segment of audio for speaker identification
def segment_embedding(segment):
    global audio, duration, embedding_model, path

    # getting start of the timestamp
    start = segment["start"]

    # Whisper overshoots the end timestamp in the last segment
    # set the overshooted timestamp to max duration of the audio recording
    # ensures that the end time does not exceed the duration of the audio recording.
    end = min(duration, segment["end"])

    # Creates a Segment object named clip, with the start and end times determined in the previous steps.
    clip = Segment(start, end)

    # extract the waveform and sample rate of the audio segment from audio object by using crop method
    # crop takes audio_file path and segment object as input
    waveform, sample_rate = audio.crop(path, clip)

    # Reshape the waveform array to a single dimension
    # to match the input shape expected by neural network models    
    wavform = waveform[None]
    
    # using embedding model to generates embeddings (numerical representations) for the input audio segment
    embd_mdl = embedding_model(wavform)

    # returning the audio segment embedding
    return embd_mdl


def get_diarization(pth, speaker_count = 2, output_txt_file = None):
    global audio, duration, embedding_model, path

    # assigns uploaded audio file path to path variable
    path = pth

    # Checks if the file extension of the given path is not 'wav'. If it's not, convert the audio file to WAV format using FFmpeg. 
    # Ensures that the input audio file is in the WAV format, which is commonly used for audio processing.
    if not path.lower().endswith('.wav'):
        subprocess.run(
            ['ffmpeg', '-y', '-i', path, '-ac', '1', '-ar', '16000', './static/generalAudioAndOutput/audio.wav'],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        )
        path = './static/generalAudioAndOutput/audio.wav'
    
    # Reads the WAV file using wavfile.read()
    fs, data = wavfile.read(path)
    
    # calcutate the length of the audio file by dividing the length of the audio by sample_rate (fs)
    audio_length = len(data)/fs

    # debugging and logging purpose
    print("Audio Length:", audio_length)


    # Attempts to write the single-channel audio data to a new WAV file. If the audio is already single-channel, it writes the data directly. 
    # This step ensures that the audio data is single-channel, which is often expected by downstream processing steps.
    try:
        print("Converting Multi Dimensional wav to Single Dimensional wav")
        wavfile.write('./static/generalAudioAndOutput/audio.wav', fs, data[:, 0])
    except:
        print("Already Got Single Dimensional wav")
        wavfile.write('./static/generalAudioAndOutput/audio.wav', fs, data[:])

    
    # plt.show()


    # setting the path to the temporary audio file
    path = './static/generalAudioAndOutput/audio.wav'

    # Load a speech-to-text model using the whisper.load_model() function, to transcribe the audio segments later in the code.
    model = whisper.load_model(model_size)#Speech to Text Model

    # Transcribes the audio file using the loaded speech-to-text model
    result = model.transcribe(path)

    # Retrieve the transcribed segments from the result dictionary and assign them to the raw_segments variable.
    raw_segments = result["segments"]

    # print("All Segments:", type(raw_segments))

    # Filters out segments whose start times exceed the duration of the audio. 
    # This ensures that only valid segments are considered for further processing.
    segments = []
    for segment in raw_segments:
        if segment["start"] <= audio_length:
            segments.append(segment)
        else:
            pass
            # print("Segment beyond Audio Duration. Ignoring")

    if len(segments) < speaker_count:
        raise ValueError(f"Found {len(segments)} speech segments; choose at most {len(segments)} speakers or use a longer recording.")

    # Opens the WAV file using the wave.open() function and calculates the duration of the audio in seconds.
    wav_file = wave.open(path,'r')
    with contextlib.closing(wav_file) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)

    # Instantiate an Audio object to handle audio processing tasks.
    audio = Audio()

    # Initialize an array named embeddings with zeros. This array will store the embeddings (features) of the audio segments.
    embeddings = np.zeros(shape=(len(segments), 192))

    # Iterate over the segments using the tqdm library, which provides a progress bar for the loop. (for terminal debugging and monitoring purposes) 
    # s_no represents the segment number, and segment represents each segment.
    for s_no, segment in tqdm(list(enumerate(segments))):
        # calls the segment_embedding function to generate embeddings for the current segment
        embd = segment_embedding(segment)
        # print(embd)
        
        # if speech probability is less, then there is no activity
        if segment["no_speech_prob"] < 0.5: 
            embeddings[s_no] = embd
        else:
            pass
            
    # perform agglomerative clustering on the generated embeddings using expected number of speakers
    clustering = AgglomerativeClustering(speaker_count).fit(embeddings)

    # Retrieves the cluster labels assigned to each segment from the clustering result.
    # labels has same length as segments and each label corresponds to each segment
    labels = clustering.labels_

    # Assigns a speaker label to each segment based on the clustering result.  
    for i in tqdm(range(len(segments))):
        try:
            # If the probability of non-speech in the segment is less than 0.5,
            # assign a speaker label of the form "SPEAKER-N" where N is the cluster label plus 1. 
            if segments[i]["no_speech_prob"] < 0.50:
                segments[i]["speaker"] = 'SPEAKER-' + str(labels[i] + 1)
                
                # debugging and logging purpose
            # If the probability of non-speech is higher, it sets the segment's text and speaker to empty strings.
            else:
                segments[i]["text"] = " "
                segments[i]["speaker"] = " "
        except Exception as e:
            print("Error in labeling.", e)


    # dictionary of emotions
    emotions = {
        "Happy": "\U0001F604",     # 😀
        "Fear": "\U0001F628",      # 😔
        "Neutral": "\U0001F610",   # 😐
        "Angry": "\U0001F620",     # 😠
        "Surprise": "\U0001F632"   # 😲
    }

    # writing into final diarization result variable
    # it will be used by html pages to render the results to user
    output_text = ""
    # Keep transcript text in memory for the web request. A caller can still
    # request a file explicitly for the standalone analysis workflow.
    f = open(output_txt_file, "w", encoding="utf-8") if output_txt_file else io.StringIO()
    for (i, segment) in enumerate(segments):
        if i == 0 or segments[i - 1]["speaker"] != segment["speaker"]:
            f.write("\n" + segment["speaker"] + ' ' + str(time(segment["start"])) + " ---> " + str(time(segment["end"])) + '\n')
            # write to result variable: speaker label + time stamps
            output_text += "\n" + segment["speaker"] + ' [from: ' + str(time(segment["start"])) + " | to: " + str(time(segment["end"])) + ']\n\n'

        f.write(segment["text"][1:] + ' ')

        if len(segment["text"][1:])>2:
            # computing sentiment category for each text segment only if text length is more 
            sent_cat, sent_score = categorize_sentiment(segment["text"][1:])
            # writing sentiment category and emotion (emoji) to end of the text segment
            output_text += segment["text"][1:] + '\n' + "[" + sent_cat + " " + emotions[sent_cat] + "]" + '\n\n'
        else:
            # otherwise simply write the text segment without any sentiment categorization as it is meaning less
            output_text += segment["text"][1:] + '\n\n'

    # close the file
    f.close()

    # return the diarization output along with sentiment analysis on each speech segment (only individual speech segments)
    return output_text

if __name__ == "__main__":
    # print(whisper.available_models())
    pth = 'test.wav'
    output = get_diarization(pth)
    print("OP:", output)
