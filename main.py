import logging
import os
import time
import soundfile as sf
from fastapi import FastAPI, File, UploadFile, Form
from huggingface_hub import snapshot_download
from sensevoice.onnx.sense_voice_ort_session import SenseVoiceInferenceSession
from sensevoice.utils.frontend import WavFrontend
from sensevoice.utils.fsmn_vad import FSMNVad
import subprocess
from opencc import OpenCC
import csv
import re

# Constants
languages = {"auto": 0, "zh": 3, "en": 4, "yue": 7, "ja": 11, "ko": 12, "nospeech": 13}
DOWNLOAD_MODEL_PATH = os.path.join(os.path.dirname(__file__), "resource")

# Setup logging
formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
logging.basicConfig(format=formatter, level=logging.INFO)

# Load the model and resources once at startup
if not os.path.exists(DOWNLOAD_MODEL_PATH):
    logging.info("Downloading model from HuggingFace Hub")
    snapshot_download(repo_id="lovemefan/SenseVoice-onnx", local_dir=DOWNLOAD_MODEL_PATH)

front = WavFrontend(os.path.join(DOWNLOAD_MODEL_PATH, "am.mvn"))

# default setting for loading model
device_default=-1
num_threads_default=4
use_int8_default = False

model = SenseVoiceInferenceSession(
    os.path.join(DOWNLOAD_MODEL_PATH, "embedding.npy"),
    os.path.join(DOWNLOAD_MODEL_PATH, "sense-voice-encoder.onnx"),
    os.path.join(DOWNLOAD_MODEL_PATH, "chn_jpn_yue_eng_ko_spectok.bpe.model"),
    device_default,
    num_threads_default,
)

# Load VAD model
vad = FSMNVad(DOWNLOAD_MODEL_PATH)

# Make a 'audio' directory to store audio
if not os.path.exists('audio'):
    os.makedirs('audio')
    print("The 'audio' directory is created!")

# Make a 'transcript' directory to store transcript txt files
if not os.path.exists('transcript'):
    os.makedirs('transcript')
    print("The 'transcript' directory is created!")

def get_wav_info(wav_file):
    with sf.SoundFile(wav_file) as f:
        num_channels = f.channels
        sample_rate = f.samplerate
    return num_channels, sample_rate

# modify the emoji as you like, try some old days emoji symbol text, otherwise would be cleansed in terminal
emoji_dict = {
    "<|nospeech|><|Event_UNK|>": "",
    "<|zh|>": "",
    "<|en|>": "",
    "<|yue|>": "",
    "<|ja|>": "",
    "<|ko|>": "",
    "<|nospeech|>": "",
    "<|HAPPY|>": "(ᵔ ᵕ ᵔ)",
    "<|SAD|>": "☹",
    "<|ANGRY|>": "(⩺_⩹)",
    "<|NEUTRAL|>": "",
    "<|BGM|>": "♫",
    "<|Speech|>": "",
    "<|Applause|>": "(^^)//",
    "<|Laughter|>": "( ≧ᗜ≦)",
    "<|FEARFUL|>": "ヾ(｡ꏿ﹏ꏿ)ﾉﾞ",
    "<|DISGUSTED|>": "(¬_¬”)",
    "<|SURPRISED|>": "(⚆ᗝ⚆)",
    "<|Cry|>": "( ´〒^〒`)",
    "<|EMO_UNKNOWN|>": "",
    "<|Sneeze|>": "(> ་། <)",
    "<|Breath|>": "(꒪ܠ꒪)༄༄",
    "<|Cough|>": "( >д<)､;'.",
    "<|Sing|>": "ヾ('O`)ﾉ🎙️ ♪",
    "<|Speech_Noise|>": "",
    "<|withitn|>": "",
    "<|woitn|>": "",
    "<|GBG|>": "",
    "<|Event_UNK|>": ""
}

empty_dict ={
    "<|nospeech|><|Event_UNK|>": "",
    "<|zh|>": "",
    "<|en|>": "",
    "<|yue|>": "",
    "<|ja|>": "",
    "<|ko|>": "",
    "<|nospeech|>": "",
    "<|HAPPY|>": "",
    "<|SAD|>": "",
    "<|ANGRY|>": "",
    "<|NEUTRAL|>": "",
    "<|BGM|>": "",
    "<|Speech|>": "",
    "<|Applause|>": "",
    "<|Laughter|>": "",
    "<|FEARFUL|>": "",
    "<|DISGUSTED|>": "",
    "<|SURPRISED|>": "",
    "<|Cry|>": "",
    "<|EMO_UNKNOWN|>": "",
    "<|Sneeze|>": "",
    "<|Breath|>": "",
    "<|Cough|>": "",
    "<|Sing|>": "",
    "<|Speech_Noise|>": "",
    "<|withitn|>": "",
    "<|woitn|>": "",
    "<|GBG|>": "",
    "<|Event_UNK|>": ""
}

row_num = None

# check if the authorization key is in csv_file, for simplicity, only row[0] (the first column) is checked
def check_key_in_csv(key, filename):
    with open(filename, newline='') as csvfile:
        csv_reader = csv.reader(csvfile)
        line_count = 0
        for row in csv_reader:
            if (row[0] == key) and (line_count!=0):
                global row_num
                row_num= line_count
                return True
            line_count += 1
    return False

# FastAPI app
app = FastAPI()

@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    SENSE_VOICE_KEY: str= Form(None),
    device: int = Form(-1),
    num_threads: int = Form(4),
    language: str = Form('auto'),
    use_int8: bool = Form(False),
    use_itn: bool = Form(True),
    replace_tag: str= Form(None),
    s2t: bool = Form(False),
    timecode: bool = Form(True),  
):
    keys_filename = 'keys.csv'
    is_key_present = check_key_in_csv(SENSE_VOICE_KEY, keys_filename)
    if not is_key_present:
        return "Your keys is invalid or missing!"
    # if the requested model parameters is not same as default, reload the models
    if (device!=device_default) or (num_threads!=num_threads_default) or (use_int8!=use_int8_default):
        logging.info(f"Reloading model with device={device}, num_threads={num_threads},--use_int8={use_int8}")
        global model
        model = SenseVoiceInferenceSession(
            os.path.join(DOWNLOAD_MODEL_PATH, "embedding.npy"),
            os.path.join(DOWNLOAD_MODEL_PATH, 
                         "sense-voice-encoder-int8.onnx"
                         if use_int8
                         else "sense-voice-encoder.onnx",
                         ),
            os.path.join(DOWNLOAD_MODEL_PATH, "chn_jpn_yue_eng_ko_spectok.bpe.model"),
            device,
            num_threads,
        )

    # Save the uploaded file
    audio_file_path = f"audio/{file.filename}"
    audio_name, _ = os.path.splitext(file.filename)
    with open(audio_file_path, "wb") as f:
        f.write(await file.read())

    base_name, file_extension = os.path.splitext(audio_file_path)
    # if wav file is uploaded, it will check if it is a single channel, 16000Hz
    if file_extension == '.wav':
        channels, rate = get_wav_info(audio_file_path)
        print(f"It is a wave file")
        print(f"Number of channels: {channels}")
        print(f"Sampling frequency: {rate} Hz")
        if (channels==1) and(rate==16000):
            wav_file_path = audio_file_path
            print(f"No need to be converted")
    else:    
        # Convert the audio file using ffmpeg
        # base_name, _ = os.path.splitext(audio_file_path)
        wav_file_path = base_name + '.converted.wav'
        subprocess.run(["ffmpeg", "-y", "-i", audio_file_path, "-f", "wav", "-acodec", "pcm_s16le", "-ac", "1", "-ar", "16000", wav_file_path])

    # Read audio file
    waveform, _sample_rate = sf.read(wav_file_path, dtype="float32", always_2d=True)
    logging.info(f"Audio {file.filename} is {len(waveform) / _sample_rate} seconds, {waveform.shape[1]} channel")
    start = time.time()
    # Process each audio channel
    results = []
    for channel_id, channel_data in enumerate(waveform.T):
        segments = vad.segments_offline(channel_data)
        for part in segments:
            audio_feats = front.get_features(channel_data[part[0] * 16 : part[1] * 16])
            asr_result = model(
                audio_feats[None, ...],
                language=languages[language],
                use_itn=use_itn,
            )
            result = f"[{part[0] / 1000}s - {part[1] / 1000}s] {asr_result}"
            logging.info(result)
            results.append(result)
        vad.vad.all_reset_detection()

    decoding_time = time.time() - start
    logging.info(f"Decoder audio takes {decoding_time} seconds")
    logging.info(f"The RTF is {decoding_time/(waveform.shape[1] * len(waveform) / _sample_rate)}.")
    response=''.join([str(item) for item in results])
    print(f"The raw response: {response}")

    # write the transcript text file
    transcript_raw_path = f"transcript/{audio_name}.raw.txt"
    f = open(transcript_raw_path, "w")
    f.write(response)
    f.close()
    print(f"{transcript_raw_path} is written")

    # replace tag, either emoji or empty if in arguments
    if replace_tag=='emoji':
        for key, value in emoji_dict.items():
            response = response.replace(key, value)
    elif replace_tag=='empty':
        for key, value in empty_dict.items():
            response = response.replace(key, value)

    # convert simplified chinese to traditional chinese
    if s2t:
        cc = OpenCC('s2t')
        response = cc.convert(response)
    
    # remove time codes
    cleaned_response = re.sub(r'\[\d+\.\d+s - \d+\.\d+s\]', '', response).strip()
    cleaned_response = cleaned_response.strip(',')
    print(f"The cleaned response: {cleaned_response}")
    # write the transcript file
    transcript_path = f"transcript/{audio_name}.txt"
    f = open(transcript_path, "w")
    f.write(cleaned_response)
    f.close()
    print(f"{transcript_path} is written")
    return (response if timecode else cleaned_response)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9528)