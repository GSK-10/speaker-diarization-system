# pip install flask
# pip install opencv-contrib-python
# pip install tensorflow
# pip install numpy

print("Starting..")

# flask framework
# flask: used to create a web application instance.
# render_template: render HTML templates with dynamic data.
# request: access to information about the current HTTP request, such as headers, form data, and cookies.
# redirect: redirect users to a different URL within the application.
# flash: allows you to display temporary messages (flash messages) to users, often used for conveying success or error messages.
from flask import Flask,render_template,request,json,jsonify,session,redirect,send_file,url_for,flash,abort

# functionalities for interacting with the operating system - accessing file paths, environment variables, and creating directories.
import os
from pathlib import Path
import secrets
import traceback
from threading import Lock

# Used to secure filenames before storing them directly on the filesystem, sanitizing filenames to prevent malicious code injection.
from werkzeug.utils import secure_filename

# display web content within a native desktop window (obsolete)
# import webview


# encoding and decoding data in Base64 format, binary data handling (audio) in text strings format
import base64
import io

# reading and writing WAV audio files (audio processing tasks)
from scipy.io import wavfile

# module to record and save audio file for diarization followed by sentiment analysis
import utils.Models.speaker_record_and_save as speaker_record_and_save

# function to perform actual diarization
from utils.Models.speaker_diarization import get_diarization

# functionality for sentiment analysis
from utils.Models.sentiment_analysis import categorize_sentiment

# import the necessary packages
import os

# Function that integrates speaker diarization and sentiment analysis into a single function
# This function will be called in first_page() and record() functions
diarization_lock = Lock()

def predict(filename, speaker_count):
    try:
        speakerwise_sentiment = dict() # store sentiment analysis results for each speaker.

        # sending the filename and speaker count as parameters
        # The legacy model shares intermediate filenames and module state.
        # Run one analysis at a time until those are made request-local.
        with diarization_lock:
            diarization_output_complete = get_diarization(filename, speaker_count = speaker_count)
        
        # debugging / logging purpose
        print("Diarization output ready")

        ######################################################## Sentiment analysis block ########################################################

        ################### Goals ########################

        # 1. Computing sentiment of overall conversation
        # 2. Computing sentiment of each speaker

        ##################################################

        # accumulate all sentences from the diarization output. Used to compute sentiment of overall conversation
        raw_text = ""
        
        # keep track of the current speaker -> used in core logic of the for loop
        last_speaker = ""

        ## This loop is intended to do two things
        # 1. Accumulate raw text from all the speaker and store in raw_text variable. Overall sentiment of the conversation in calculated on this
        # 2. Accumulate speaker wise sentences as {speaker: [list of speech segments]} in speakerwise_sentiment dictionary 
        for sentence in diarization_output_complete.split("\n"):
            sentence = sentence.strip()
            # The model places a sentiment label on its own line after each
            # transcript segment. It is display text, not spoken text.
            if sentence.startswith("[") and sentence.endswith("]"):
                continue
            # accounting only sentences with length more than 1
            if len(sentence) > 1:
                # See output for clarity
                # if "SPEAKER-" pattern is not there in a sentence, then it is a speech segment
                if "SPEAKER-" not in sentence:
                    
                    # Now sentence is a transcribed speech segment along with its sentiment in the end
                    # Ex: This is betadevil speaker --> Neutral 
                    
                    # Now we need to remove the sentiment label and make it null (because we need a raw speech segment) 
                    sentence = sentence.replace("--> Surprise", "")
                    sentence = sentence.replace("--> Fear", "")
                    sentence = sentence.replace("--> Neutral", "")
                    sentence = sentence.replace("--> Happy", "")
                    sentence = sentence.replace("--> Angry", "")
                    sentence = sentence.replace("|", "")

                    # The example sentence now becomes
                    # Processed Ex: This is betadevil speaker

                    # Append the processed / sentiment-label free sentence to raw_text variable
                    raw_text += sentence + " "

                    ## Accumulating the speech segment to its corresponding speaker to compute speakerwise sentiment eventually
                    # if last_speaker is already in the dict, then simply append this sentence to the list of sentences for present speaker 
                    if last_speaker in speakerwise_sentiment.keys():
                        speakerwise_sentiment[last_speaker].append(sentence)
                    # otherwise create a new key with speaker label is created and a list is created
                    else:
                        speakerwise_sentiment[last_speaker] = [sentence,]

                # otherwise it is a speaker label + start time stamp + end time stamp
                else:
                    # here we update the last speaker
                    last_speaker = sentence.split(" ")[0]
                    
                    # logging / debugging purpose
                    print("Last Speaker:", last_speaker)
                    
                    # same sub if-else block like in above if block
                    # this is done to take the time stamps also
                    if last_speaker in speakerwise_sentiment.keys():
                        speakerwise_sentiment[last_speaker].append(sentence)
                    else:
                        speakerwise_sentiment[last_speaker] = [sentence,]
                    

        # debugging / logging purpose
        print("Collected transcript for sentiment analysis")

        # categorize_sentiment function is called on the raw_text to get the overall sentiment category and sentiment scores
        # to be displayed on the top in the UI
        sentiment_category_complete, sentiment_scores_complete = categorize_sentiment(raw_text)

        # appending scores to the sentiment category complete
        sentiment_category_complete += f" | Weightage - Neutral: {sentiment_scores_complete['neu']}, Negative: {sentiment_scores_complete['neg']}, Positive: {sentiment_scores_complete['pos']}" 

        # logging and debugging purpose
        print("Speaker sentiment ready")
    
    # Handle any exceptions in diarization
    except Exception as e:
        print("ERROR in Diarization:", ascii(str(e)))
        traceback.print_exc()
        sentiment_category_complete = "Retry"
        diarization_output_complete = "Silent or \n Unclear Audio"
 
    ### Actual Sentiment Calculation for each speaker for their overall speech ###   
    ## Iterating through each speaker 
    for speaker in speakerwise_sentiment.keys():
        # Variable to concat all the sentences of each speaker
        text = ""
        # iterating through each sentence assigned to the speaker 
        for sentence in speakerwise_sentiment[speaker]:
            if "SPEAKER-" not in sentence: # it means we are interested only in the actual transcribed text
                text += sentence # append speech transcribed segment to text
        sent_cat, sent_score = categorize_sentiment(text) # finding sentiment assigned to each speaker using sentiment analysis module
        speakerwise_sentiment[speaker].append(sent_cat) # appending the sentiment of the speaker at the end of the list
        # Note: Even in the data page, we should access the last sentence to get the sentiment
    

    diarization_output_complete = diarization_output_complete.replace("| |", "")
    
    # returning the results
    return diarization_output_complete, sentiment_category_complete, speakerwise_sentiment


def format_results_text(transcript, overall_sentiment, speakerwise_sentiment):
    lines = [
        'Speaker Diarization Results', '',
        'Overall sentiment', overall_sentiment, '',
        'Transcript', transcript.strip(), '',
        'Speaker-wise sentiment',
    ]
    for speaker, details in speakerwise_sentiment.items():
        lines.extend(['', f'{speaker}: {details[-1] if details else "Unknown"}'])
        lines.extend(details[:-1])
    return '\n'.join(lines).rstrip() + '\n'


# model_warmup()


### Creating a flask application
app=Flask(__name__)

### Setting a secret key for the Flask application
app.secret_key = os.environ.get("FLASK_SECRET_KEY") or secrets.token_hex(32)

### Set up the upload folder for file uploads
app.config['UPLOAD_FOLDER'] = './static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('./static/generalAudioAndOutput', exist_ok=True)

PROJECT_ROOT = Path(__file__).resolve().parent
SAMPLE_FILES = {
    'short': PROJECT_ROOT / 'TestAudios' / 'conv0.wav',
    'long': PROJECT_ROOT / 'utils' / 'audioFiles' / 'long_audio_files' / '4_Speaker_5mins.wav',
    'conv0': PROJECT_ROOT / 'TestAudios' / 'conv0.wav',
    'conv1': PROJECT_ROOT / 'TestAudios' / 'conv1.wav',
    'conv3': PROJECT_ROOT / 'TestAudios' / 'conv3.wav',
    'long-2': PROJECT_ROOT / 'utils' / 'audioFiles' / 'long_audio_files' / '2_Speakers_4mins.wav',
    'long-4': PROJECT_ROOT / 'utils' / 'audioFiles' / 'long_audio_files' / '4_Speaker_5mins.wav',
    'long-5': PROJECT_ROOT / 'utils' / 'audioFiles' / 'long_audio_files' / '5_Speaker_3mins.wav',
}
SAMPLE_GROUPS = (
    ('Short conversations · 2 speakers', (
        ('conv0', 'Conversation 1 · 27 seconds'),
        ('conv1', 'Conversation 2 · 19 seconds'),
        ('conv3', 'Conversation 3 · 24 seconds'),
    )),
    ('Long conversations', (
        ('long-2', '2 speakers · 4 minutes 19 seconds'),
        ('long-4', '4 speakers · 5 minutes 3 seconds'),
        ('long-5', '5 speakers · 2 minutes 31 seconds'),
    )),
)


@app.route('/samples/<sample_name>')
def download_sample(sample_name):
    sample = SAMPLE_FILES.get(sample_name)
    if sample is None or not sample.is_file():
        abort(404)
    return send_file(sample, as_attachment=True, download_name=sample.name)


@app.post('/download-results')
def download_results():
    result_text = request.form.get('result_text', '')
    if not result_text:
        abort(400)
    return send_file(
        io.BytesIO(result_text.encode('utf-8')),
        mimetype='text/plain; charset=utf-8',
        as_attachment=True,
        download_name='speaker-diarization-results.txt',
    )

### Set up a set of allowed file extensions for uploaded files
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'webm', 'm4a', 'mp4'}

### Function to check if a filename has allowed extension 
def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


### flask route handler for first page function (decorator) ###
### Route for homepage of the application ###
# When a user visits the root URL (/), the first_page() function is called, and it returns an appropriate page as per request
# Handles GET and POST requests associated with '/' route
@app.route('/',methods=["post","get"])
def first_page():
    # if request is POST, diarization is done and data_page.html is rendered with the results
    if request.method=="POST":
        global image_name, image_data

        # debugging and logging purpose
        print("POST received")

        # retrieves the uploaded file from form
        file = request.files.get('file')

        # if filename is null, file is not provided
        if file is None or file.filename == '':
            # using flash to show a warning msg
            flash('Select an audio file to upload.')
            # redirect again to the same page
            return redirect(request.url)

        # otherwise file is uploaded. Now we check if uploaded file has allowed extension or not using the function (.mp3 / .wav)
        if file and allowed_file(file.filename):

            # using secure_filename to safely extract the filename from 
            filename = secrets.token_hex(8) + '_' + secure_filename(file.filename)
            
            # debugging and logging purpose
            print("Filename:", filename)
            
            # os.path.join is used to concat two paths
            # file.save is used to save the uploaded file in static/uploads folder
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            # debugging and logging purpose
            print(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            # modifiying the filename to its relative path in static/uploads
            filename = app.config['UPLOAD_FOLDER'] + "/" + filename

            # debugging and logging purpose
            print(filename)

            # extracting number of speakers from the input field -> this will be used for clustering
            try:
                speaker_count = int(request.form['speaker_count'])
            except (KeyError, ValueError):
                flash('Choose the number of speakers.')
                return redirect(request.url)
            if speaker_count not in range(2, 6):
                flash('Choose between 2 and 5 speakers.')
                return redirect(request.url)

            # debugging and logging purpose
            print("Checking the file path for audio: ", filename)
            
            # using predict function to get speaker diarization results, overall sentiment of the conversation, speakerwise sentiment 
            diarization_output, sentiment_category, speakerwise_sentiment = predict(filename, speaker_count)
            if sentiment_category == 'Retry':
                flash('Audio processing failed. Try a clearer recording or fewer speakers; the terminal has details.')
                return redirect(request.url)

            ## plotting audio spectogram (obsolete) -> working only one time
            # fs, data = wavfile.read(filename)
            # import matplotlib.pyplot as plt
            # print("Saving Audio as png")
            # plt.plot(data[0:])
            # plt.ylabel("Amplitude")
            # plt.xlabel("Time")
            # plt.savefig("./static/audioVisualize/audio.png")
            # del plt
            # print("Saving Audio as png Done")

            ## render an HTML template named "data_page.html"
            return render_template("data_page.html", filename=filename, result = sentiment_category, solution = diarization_output.split("\n"), speakerwise_sentiment = speakerwise_sentiment, img_src="./static/audioVisualize/audio.png", download_text=format_results_text(diarization_output, sentiment_category, speakerwise_sentiment))
        else:
            flash('Allowed audio files are WAV, MP3, WebM, M4A, and MP4.')
            ## if allowed type file is not used or the file is null
            # redirect to the same page again
            return redirect(request.url)
    
    # otherwise if request is of type GET -> form_page.html (upload audio recording) is rendered
    else:
        return render_template("form_page.html", sample_groups=SAMPLE_GROUPS)


### flask route handler for record page function (decorator) ###
### Route for live record page of the application ###
# When a user visits the root URL (/record), the record_page() function is called, and it returns an appropriate page as per request
# Handles GET and POST requests associated with '/' route
@app.route('/record',methods=["post","get"])
def record_page():
    global audio_path
    # when the request is a POST request
    if request.method=="POST":
        global image_name, image_data

        # Debugging and logging purpose
        print("Record received")

        # Get the record duration from the input field
        try:
            duration = int(request.form['duration'])
            speaker_count = int(request.form['speaker_count'])
        except (KeyError, ValueError):
            flash('Choose a duration and the number of speakers.')
            return redirect(request.url)
        if duration not in (10, 15, 20) or speaker_count not in range(2, 6):
            flash('Choose a listed duration and between 2 and 5 speakers.')
            return redirect(request.url)
        
        # calling the record_audio funtion from speaker_record_and_save module to record audio using the user's microphone.
        try:
            speaker_record_and_save.record_audio(duration=duration)
        except Exception as exc:
            print('Recording error:', ascii(str(exc)))
            flash('Could not access this computer’s microphone. Use browser recording or upload an audio file.')
            return redirect(request.url)
        
        # Setting the audio path of the processed recording
        speaker_record_and_save.audio_path = "./static/uploads/recordedAudio.wav"
        
        # Saving the audio file
        try:
            speaker_record_and_save.save_audio()
        except Exception as exc:
            print('Recording save error:', ascii(str(exc)))
            flash('Could not save the recording. Try again or upload an audio file.')
            return redirect(request.url)

        # setting the file to the location of the saved audio file
        filename = speaker_record_and_save.audio_path

        # debugging purpose
        print("Checking the file path for audio: ", filename)
        
        # using predict function to get speaker diarization results, overall sentiment of the conversation, speakerwise sentiment
        diarization_output, sentiment_category, speakerwise_sentiment = predict(filename, speaker_count)
        if sentiment_category == 'Retry':
            flash('Audio processing failed. Try a clearer recording or fewer speakers; the terminal has details.')
            return redirect(request.url)

        ## plotting audio spectogram (obsolete) -> working only one time    
        # fs, data = wavfile.read(speaker_record_and_save.audio_path) ###
        # import matplotlib.pyplot as plt
        # print("Saving Audio as png")
        # plt.plot(data[0:])
        # plt.ylabel("Amplitude")
        # plt.xlabel("Time")
        # plt.savefig("./static/audioVisualize/audio.png")
        # del plt
        # print("Saving Audio as png done")

        ## render an HTML template named "data_page.html"
        return render_template("data_page.html", filename=filename, result = sentiment_category, solution = diarization_output.split("\n"), speakerwise_sentiment = speakerwise_sentiment, img_src="./static/audioVisualize/audio.png", download_text=format_results_text(diarization_output, sentiment_category, speakerwise_sentiment))
    else:
        # otherwise if request is of type GET -> form_page_record.html (upload audio recording) is rendered
        return render_template("form_page_record.html")

# starts the built-in development server
# enable Flask's debug mode for detailed error reporting, automatic page reloading for reflecting changes
if __name__ == "__main__":
    app.run(debug=os.getenv("FLASK_DEBUG") == "1", use_reloader=False)

