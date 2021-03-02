from flask_ngrok import run_with_ngrok
from flask import Flask, render_template, redirect, url_for, request
from google.cloud import translate_v2 as gtranslate
# Import modules
from google.cloud import speech as speech
from scipy.io.wavfile import read as wav_read
from src.transformer import *
import os
import io
import pickle

# Load model
f = open('SRC.pkl', 'rb')
SRC= pickle.load(f)
f.close()

f = open('TRG.pkl', 'rb')
TRG=pickle.load(f)
f.close()
# opt['n_layers'] = 8
global model
model = Transformer(len(SRC.vocab), len(TRG.vocab), opt['d_model'], opt['n_layers'], opt['heads'], opt['dropout'])
model = model.to(opt['device'])
model.load_state_dict(torch.load('./model/trans8layers_ep30.pth'))

# Define some function to translate text

def set_key(key_name):
	os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = key_name

def google_trans(text):
	client = gtranslate.Client()
	r = client.translate(text, target_language='vi')

	return r['translatedText']

def speech_recog(filename):
	# Instantiate a client
	speech_client = speech.SpeechClient()

	# Load audio file
	with io.open(os.path.join('./upload', filename), 'rb') as audio_file:
		content = audio_file.read()
		# sample = speech_client.sample(
		# 	content,
		# 	source_uri=None,
		# 	encoding='LINEAR16'
		# 	)

	config = {
		'config': {
			'language_code': 'en-US',
			# 'sample_rate_hertz': 8000,
			'encoding': speech.RecognitionConfig.AudioEncoding['LINEAR16']
		},

		'audio': {
			'content': content
		}
	}

	# Detect speech
	# print('[DBG] Den day roi')
	response = speech_client.recognize(config)
	# print(response)
	for t in response.results:
		a = t.alternatives[0].transcript
	# print(a)
	return a

# Build Flask app
app = Flask(__name__)
run_with_ngrok(app)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/text', methods=["POST", "GET"])
def text(transformer="", google="",origin=""):
	if request.method == "POST":
		origin = request.form['nm']
  
		translated = translate_sentence(origin, model, SRC, TRG, opt['device'], opt['k'], opt['max_strlen'])
		return render_template('text.html', origin=origin, google=google_trans(origin), transformer=translated)
	else:
		return render_template('text.html')

@app.route('/record')
def record():
	return render_template('record.html')

@app.route('/audio', methods=["POST", "GET"])
def audio(google="", transformer=""):
	if request.method == "POST":
		file = request.files['filename']
		filename = file.filename
		file.save(os.path.join('./upload', filename))
		text = speech_recog(filename=filename)
		google = google_trans(text)
		transformer = translate_sentence(text, model, SRC, TRG, opt['device'], opt['k'], opt['max_strlen'])
		return render_template('audio.html', google=google, transformer=transformer)
	else:
		return render_template('audio.html')

if __name__ == '__main__':
	key_name = 'NLP1-290e070a9ba9.json'
	set_key(key_name)
	app.run()