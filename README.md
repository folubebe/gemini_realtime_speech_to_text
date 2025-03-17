# Speech Translator

A real-time speech translation tool using Google's Gemini API.

## Features
- Continuous audio recording and processing
- Real-time transcription of speech
- Translation between languages (configurable)
- Saves transcriptions to a file

## Requirements
- Python 3.x
- PyAudio
- NumPy
- Google Generative AI Python SDK 


## Installation

1. Clone the repository:
git clone https://github.com/folubebe/gemini_realtime_speech_to_text.git
cd gemini_realtime_speech_to_text

2. Install dependencies:
pip install -r requirements.txt

3. Create an API key from https://aistudio.google.com/apikey

Set the API key as an environment variable:
set GOOGLE_API_KEY=your-api-key-here # Replace with your actual API key.

Alternatively, create a .env file in the project root with:
GOOGLE_API_KEY=your-api-key-here # Replace with your actual API key.

## Usage

Run the script:
python speech_to_text.py

Follow the prompts to select your input device. The script will start recording audio in 5-second chunks, process them, and display translations in real-time.

You can modify the following parameters in the script:
- `CHUNK_DURATION_SEC = 5`: Duration in seconds for each audio processing chunk
- `TARGET_LANGUAGE = "English"`: The desired output language for translation
- `SOURCE_LANGUAGE = "auto"`: Source language detection (set to specific language to override auto-detection)

Press Ctrl+C to stop the recording. Transcriptions will be saved to `translation_output.txt`.