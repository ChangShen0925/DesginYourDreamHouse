
from diffusers import AudioLDM2Pipeline
import torch
import scipy

import random
from ImageEffects import *
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeAudioClip
from PIL import Image
import cv2
import numpy as np

effect_list = [AppearByPixels, X_Ray, swap_waterfall, zoom_swap, bur_bright_swap, open_door_swap, os.getcwd() + "/VideoEffects/snowVideo.mp4", os.getcwd() + "/VideoEffects/smokeVideo.mp4", os.getcwd() + "/VideoEffects/cloudVideo.mp4"]

def text2audio(prompt, filename):
    repo_id = "cvssp/audioldm2"
    pipe = AudioLDM2Pipeline.from_pretrained(repo_id, torch_dtype=torch.float32)
    pipe = pipe.to("cuda")

    audio = pipe(prompt, num_inference_steps=20, audio_length_in_s=30.0).audios[0]

    scipy.io.wavfile.write(f"{filename}/techno.wav", rate=16000, data=audio)


def JustImage(select_list, image_list, filename, description):
    selected_images = []
    for i in range(len(select_list)):
        if select_list[i] and image_list[i] is not None:
            selected_images.append(image_list[i])
            
    width, height = selected_images[0].size
    combined_width = width * len(selected_images)
    combined_height = height
    combined_image = Image.new("RGB", (combined_width, combined_height))

    for i in range(len(selected_images)):
        combined_image.paste(selected_images[i], (i * width, 0))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    video = cv2.VideoWriter(f'{filename}/output.mp4', fourcc, 100.0, (width, height))

    upper = width * (len(selected_images) - 1)
    combined_image = cv2.cvtColor(np.array(combined_image), cv2.COLOR_RGB2BGR)

    if description:
        stop_time = 700
    else:
        stop_time = 0
    for i in range(0, upper, 2):
        frame = combined_image[:height, i:i+width]
        if i%width == 0:
            video = stopFrame(video, stop_time, frame)
        video.write(frame)
    
    video.release()

def AddImageEffects(select_list, image_list, filename, description):
    selected_images = []
    for i in range(len(select_list)):
        if select_list[i] and image_list[i] is not None:
            selected_images.append(cv2.cvtColor(np.array(image_list[i]), cv2.COLOR_RGB2BGR))
            
    height, width, _ = selected_images[0].shape

    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    video = cv2.VideoWriter(f'{filename}/output_with_effects.mp4', fourcc, 100.0, (width, height))

    if description:
        stop_time = 1200
    else:
        stop_time = 100
    random_elements = random.sample(effect_list, len(selected_images))
    video = stopFrame(video, stop_time, selected_images[0])
    for i in range(len(selected_images) - 1):
        if type(random_elements[i]) == str:
            video = video_background(selected_images[i], selected_images[i + 1], video, random_elements[i])
        else:
            video = random_elements[i](selected_images[i], selected_images[i + 1], video)
        video = stopFrame(video, stop_time, selected_images[i + 1])
    
    video.release()



def audio2video(video, filename, description):

    audio = AudioFileClip(f"{filename}/techno.wav")

    

    num_audio_repeats = int(video.duration / audio.duration) + 1

    audio = audio.volumex(0.5)  
    audio_clips = [audio] * num_audio_repeats
    


    if description:
        audio2 = AudioFileClip(f"{filename}/bark_out.wav")
        final_audio = CompositeAudioClip([audio, audio2])
    else:
        final_audio = CompositeAudioClip(audio_clips)

    final_audio = final_audio.subclip(0, video.duration)
    video = video.set_audio(final_audio)

    video.write_videofile(f"{filename}/output_with_audio.mp4", codec="libx264", audio_codec="aac")

    video.close()
    audio.close()

def speech2video(video, filename):

    audio = AudioFileClip(f"{filename}/bark_out.wav")

    audio = audio.volumex(5.0)  


    video = video.set_audio(audio)

    video.write_videofile(f"{filename}/output_with_speech.mp4", codec="libx264", audio_codec="aac")

    video.close()
    audio.close()



def stopFrame(video, stopTime, frame):
    for _ in range(stopTime):
        video.write(frame)
    return video