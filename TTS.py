from transformers import BarkModel
import torch
from IPython.display import Audio
import scipy
import json
from pydub import AudioSegment
import os



model = BarkModel.from_pretrained("suno/bark-small")
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = model.to(device)

from transformers import AutoProcessor

processor = AutoProcessor.from_pretrained("suno/bark")

def textToSpeech(text_prompt, filename, voice_preset, i):
    if voice_preset == "EN_Speaker(Male)":
        voice_preset = "v2/en_speaker_6"
    elif voice_preset == "EN_Speaker(Female)":
        voice_preset = "v2/en_speaker_9"
    elif voice_preset == "CN_Speaker(Male)":
        voice_preset = "v2/zh_speaker_8"
    else:
        voice_preset = "v2/zh_speaker_9"
    inputs = processor(text_prompt, voice_preset=voice_preset)

    speech_output = model.generate(**inputs.to(device))
    sampling_rate = model.generation_config.sample_rate
    scipy.io.wavfile.write(f"{filename}/bark_out{i}.wav", rate=sampling_rate, data=speech_output[0].cpu().numpy())

def mergeAll(filename):
    merge_voice = AudioSegment.from_file(f"{filename}/bark_out0.wav", format="wav")
    for i in range(1, 8):
        if f'bark_out{i}.wav' in os.listdir(filename):
            merge_voice+= AudioSegment.from_file(f"{filename}/bark_out{i}.wav", format="wav")
        else:
            break
    merge_voice.export(f"{filename}/bark_out.wav", format="wav")
