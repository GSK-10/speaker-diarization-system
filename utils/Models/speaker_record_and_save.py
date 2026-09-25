from utils.Models.speaker_diarization import get_diarization

# module for recording audio
import pvrecorder

# wav file handling utilities
import wave
import struct
import numpy as np 
import io
import time
from scipy.io import wavfile

import numpy as np
from scipy.io import wavfile
from tqdm import tqdm
import json
from scipy.io import wavfile

# load data

DEBUG = 1

# Instantiates a PvRecorder object from the pvrecorder module
PvRecorder = pvrecorder.PvRecorder

# Iterate over available audio devices and assign the index of the first device to mic_index.
# This is done to be able to record audio from the very first available device
mic_index = None
for index, device in enumerate(PvRecorder.get_available_devices()):
    mic_index = index
    print("Left -->", end=" ")
    print(index, device)
    break

# setting audio path
audio_path = "audio.wav"

sr_rec= 16000
frame_length = 512

audio_raw = []
audio_raw_positive = []

# to ignore the start and end portions of the audio which are likely to have noise, silence or distortion
ignoreWindow = 40

audio = []


def record_audio(duration = 10): # default recording time is 10seconds

    # store the raw audio data and the index of the microphone device to be used for recording.
    global audio_raw, mic_index
    if mic_index is None:
        raise RuntimeError("No microphone was found on the server computer.")

    # Initialize an empty list audio_raw to store the raw audio data.
    audio_raw = []
    
    # Create a PvRecorder object with the specified microphone device index (mic_index) and frame length (frame_length)
    recorder = PvRecorder(device_index=mic_index, frame_length=frame_length)
    # Start recording audio.
    recorder.start()

    if DEBUG:
        print("##############")
        print("Recording Started.")
        print("##############")

    ### Loop to read audio frames from the microphone recorder. 
    ## iterate for the duration specified by duration in frames calculated based on the sampling rate (sr_rec) and frame length (frame_length). 
    # Each frame read from the recorder is appended to the audio_raw list.
    try:
        for i in tqdm(range(int(int(duration*sr_rec)//frame_length)+1)):
            frame = recorder.read()
            audio_raw.extend(frame)
            
    except KeyboardInterrupt:
        recorder.stop()

    finally:
        # Ensures that the recorder object is deleted (cleanup) after recording is completed. 
        # It prints a message indicating that the recording is done.
        recorder.delete()
        print("Recording Done")

    # By the end of this function the audio is stored in audio_raw list
    # It can be further saved into a desired path by using save_audio() function

def save_audio():
    # Noise reduction is only needed for live recording. Importing it while
    # starting the upload page can spend a long time compiling audio helpers.
    import noisereduce as nr
    global audio_raw_positive, audio, audio_raw, audio_path, ignoreWindow
    
    print("Saving Audio...", len(audio_raw), audio_path)
    
    # audio =  audio_raw

    # Process the raw audio data by removing a window of samples specified by ignoreWindow. Each sample is scaled down by a factor of 1.
    audio =  [ int(l/1) for l in audio_raw[ignoreWindow:-ignoreWindow]]

    print("Left Audio Samples:",len(audio), " | Max Amp", max(audio), " | Mean Amp:", sum(audio)/len(audio))
    
    # Opens a WAV file at the specified audio_path for writing.  
    with wave.open(audio_path, 'w') as f1:
        # set the parameters for the WAV file (number of channels, sample width, sample rate, frame length, compression type, and compression name)
        f1.setparams((1, 2, sr_rec, frame_length, "NONE", "NONE"))
        # f1.setparams((1, 1, sr_rec, frame_length, "NONE", "NONE"))
        # writes the audio frames to the file.
        f1.writeframes(struct.pack("h" * len(audio), *audio))
    
    # Read the audio file using wavfile.read()
    rate, data = wavfile.read(audio_path)
    # perform noise reduction using the reduce_noise function from the noisereduce module, 
    reduced_noise = nr.reduce_noise(y=data, sr=rate) 
    # write the processed audio data back to the same WAV file.
    wavfile.write(audio_path, rate, reduced_noise)

if __name__ == "__main__":
    record_audio(duration=10)
    save_audio()
    output = get_diarization(audio_path, speaker_count = 2)
    print("OP:", output)
