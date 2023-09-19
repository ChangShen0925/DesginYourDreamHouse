import os 
os.system('pip install -r requirement.txt')
from diffusers import DiffusionPipeline
import torch
import qrcode
import cv2
import numpy as np
from OutsideDesign import image2videoOutside
from InsideDesign import InsideDesign
from moviepy.editor import VideoFileClip
import gradio as gr


HOUSE = None
PROMPT = ''
ROOM = {"living room": None, "kitchen": None, "dinning room": None, "bathroom": None, "bedroom": [None, None, None, None]}


import os

import torch


from chatGPT import enhance_your_sentence, enhance_your_sentence2, enhance_your_sentence3
from Additional_elements import *

from diffusers.pipeline_utils import DiffusionPipeline


os.makedirs(os.getcwd(), exist_ok=True)


def clear():
    return None

def generateQRcode(audio, effect, request: gr.Request):
    url = str(request.headers['origin']) + '/file=' + os.getcwd()
    if audio:
        url+='/output_with_audio.mp4'
    elif effect:
        url+='/output_with_effects.mp4'
    else:
        url+='/output.mp4'
    qr = qrcode.QRCode(
    version=1,
    error_correction=qrcode.constants.ERROR_CORRECT_L,
    box_size=10,
    border=4,
    )
    qr.add_data(url)
    qr.make(fit=True)
    qr_code = qr.make_image(fill_color="black", back_color="white")
    qr_code.save('qrcode.png')
    return Image.open('qrcode.png'), url


def viewAllimages():
    return HOUSE, ROOM["dinning room"], ROOM["kitchen"], ROOM["living room"], ROOM["bathroom"], ROOM["bedroom"][0], ROOM["bedroom"][1], ROOM["bedroom"][2], ROOM["bedroom"][3]

def generateVideo(cb1, cb2, cb3, cb4, cb5, cb6, cb7, cb8, cb9, audio, effect):
    if audio and effect:
        AddImageEffects([cb1, cb2, cb3, cb4, cb5, cb6, cb7, cb8, cb9], [HOUSE, ROOM["dinning room"], ROOM["kitchen"], ROOM["living room"], ROOM["bathroom"], ROOM["bedroom"][0], ROOM["bedroom"][1], ROOM["bedroom"][2], ROOM["bedroom"][3]])
        audio2video(VideoFileClip('output_with_effects.mp4'))
        return os.getcwd() + "/output_with_audio.mp4"
    elif audio:
        JustImage([cb1, cb2, cb3, cb4, cb5, cb6, cb7, cb8, cb9], [HOUSE, ROOM["dinning room"], ROOM["kitchen"], ROOM["living room"], ROOM["bathroom"], ROOM["bedroom"][0], ROOM["bedroom"][1], ROOM["bedroom"][2], ROOM["bedroom"][3]])
        audio2video(VideoFileClip('output.mp4'))
        return os.getcwd() + "/output_with_audio.mp4"
    elif effect:
        AddImageEffects([cb1, cb2, cb3, cb4, cb5, cb6, cb7, cb8, cb9], [HOUSE, ROOM["dinning room"], ROOM["kitchen"], ROOM["living room"], ROOM["bathroom"], ROOM["bedroom"][0], ROOM["bedroom"][1], ROOM["bedroom"][2], ROOM["bedroom"][3]])
        return os.getcwd() + "/output_with_effects.mp4"
    else:
        JustImage([cb1, cb2, cb3, cb4, cb5, cb6, cb7, cb8, cb9], [HOUSE, ROOM["dinning room"], ROOM["kitchen"], ROOM["living room"], ROOM["bathroom"], ROOM["bedroom"][0], ROOM["bedroom"][1], ROOM["bedroom"][2], ROOM["bedroom"][3]])
        return os.getcwd() + "/output.mp4"
    

def viewCurrentImage(cata):
    if cata == 'bedroom':
        return ROOM[cata][0], gr.Image.update(visible=True), gr.Image.update(visible=True), gr.Image.update(visible=True)
    else:
        return ROOM[cata], gr.Image.update(visible=False), gr.Image.update(visible=False), gr.Image.update(visible=False)
    
def change_options(choice):
    if choice == "Other, please specify":
        return gr.Textbox.update(visible=True)
    else:
        return gr.Textbox.update(visible=False)


def viewExample(Style):
    if len(Style) == 0:
        return None, None, None, None, None
    elif 'Other' in Style:
        return Image.open(os.getcwd() + '/HouseStyle/Other/Other1.png'), Image.open(os.getcwd() + '/HouseStyle/Other/Other2.png'), Image.open(os.getcwd() + '/HouseStyle/Other/Other3.png'), Image.open(os.getcwd() + '/HouseStyle/Other/Other4.png'), Image.open(os.getcwd() + '/HouseStyle/Other/Other5.png')
    else:
        return Image.open(os.getcwd() + '/HouseStyle/' + Style + '/' + Style +'1.png'), Image.open(os.getcwd() + '/HouseStyle/' + Style + '/' + Style +'2.png'), Image.open(os.getcwd() + '/HouseStyle/' + Style + '/' + Style +'3.png'), Image.open(os.getcwd() + '/HouseStyle/' + Style + '/' + Style +'4.png'), Image.open(os.getcwd() + '/HouseStyle/' + Style + '/' + Style +'5.png')

def generator(Prompt):
    pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
    pipe.to("cuda")
    images = pipe(prompt=Prompt, num_inference_steps=20).images[0]
    torch.cuda.empty_cache()
    return images

    


def generateImage(Prompt, style):
    global HOUSE, PROMPT
    PROMPT = Prompt
    
    frontview = generator("The front view of A house in style of " + style +  "under the blue sky which " + Prompt)
    HOUSE = frontview
    torch.cuda.empty_cache()
    return frontview


def generate_audio(prompt):
    text2audio(prompt)
    torch.cuda.empty_cache()
    return os.getcwd()+'/techno.wav'



def generate_room_inside(cata, other):
    global ROOM, PROMPT
   
    if cata == '':
        return "Please at least select the room!", None
    else:
        result = generator("The panorama of the room of " + cata +  " which is "+ other + "in a house of " + PROMPT)
        if cata != 'bedroom':
            ROOM[cata] = result
            return "Do you like the image?", result, gr.Image.update(visible=False), gr.Image.update(visible=False), gr.Image.update(visible=False)
        else:
            if ROOM[cata][0] == None:
                ROOM[cata][0] = result
                return "Do you like the image?", result, gr.Image.update(visible=True), gr.Image.update(visible=True), gr.Image.update(visible=True)
            elif ROOM[cata][1] == None:
                ROOM[cata][1] = result
                return "Do you like the image?", gr.Image.update(visible=True), result, gr.Image.update(visible=True), gr.Image.update(visible=True)
            elif ROOM[cata][2] == None:
                ROOM[cata][2] = result
                return "Do you like the image?", gr.Image.update(visible=True), gr.Image.update(visible=True), result, gr.Image.update(visible=True)
            elif ROOM[cata][3] == None:
                ROOM[cata][3] = result
                return "Do you like the image?", gr.Image.update(visible=True), gr.Image.update(visible=True), gr.Image.update(visible=True), result
            else:
                ROOM[cata][0] = ROOM[cata][1] 
                ROOM[cata][1] = ROOM[cata][2] 
                ROOM[cata][2] = ROOM[cata][3] 
                ROOM[cata][3] = result
                return "Do you like the image?", ROOM[cata][0], ROOM[cata][1], ROOM[cata][2], result





with gr.Blocks() as demo:
    with gr.Tabs():
        with gr.TabItem("Design your House(Outside)"):
            with gr.Row():
                options = gr.Dropdown(["Asain", "Gothic", "Modern", "Neoclassical", "Nordic", "Other, please specify"], multiselect=False, label="Choose your House style")
                txt     = gr.Textbox(label="House Style", visible=False)
            #btn1 = gr.Button(value="View Some example!")
            with gr.Row():
                img1    =  gr.Image()
                img2    =  gr.Image()
                img3    =  gr.Image()
                img4    =  gr.Image()
                img5    =  gr.Image()

            with gr.Row():
                txt1 = gr.Textbox(label="Enter Your Description", lines=3)

            with gr.Row():
                btn2 = gr.Button(value="Clear")
                btn3 = gr.Button(value="Enhance Your Sentence?")
                btn4 = gr.Button(value="Submit")
                # video_1 = gr.Video()
            img    =  gr.Image(height = 900, width = 1600)
            #btn1.click(viewExample, inputs = [options], outputs = [img1, img2, img3, img4, img5])
            btn2.click(clear, inputs=[], outputs=[txt1])
            btn3.click(enhance_your_sentence, inputs = [options, txt, txt1], outputs = [txt1])
            btn4.click(generateImage, inputs=[txt1, options], outputs=[img])
            options.change(change_options, inputs = [options], outputs = [txt])
            options.change(viewExample, inputs = [options], outputs = [img1, img2, img3, img4, img5])

        with gr.TabItem("Design your House(Inside)"):
            with gr.Row():
                room_options = gr.Dropdown([ "dinning room", "kitchen", "living room", "bathroom", "bedroom"], multiselect=False, label="Choose your design")
                I_txt = gr.Textbox(label="Enter Your Description")
            with gr.Row():
                btn1 = gr.Button(value="Clear")
                btn2 = gr.Button(value="Enhance Your Sentence?")
                btn3 = gr.Button(value="Submit")
            
            I_txt_2 = gr.Textbox()

            with gr.Row():
                I_im_1 = gr.Image()
                I_im_2 = gr.Image(visible = False)
            with gr.Row():
                I_im_3 = gr.Image(visible = False)
                I_im_4 = gr.Image(visible = False)

            
            btn1.click(clear, inputs=[], outputs=[I_txt])
            btn2.click(enhance_your_sentence2, inputs = [room_options, I_txt], outputs = [I_txt])
            btn3.click(generate_room_inside, inputs=[room_options, I_txt], outputs=[I_txt_2, I_im_1, I_im_2, I_im_3, I_im_4])
            room_options.change(viewCurrentImage, inputs = [room_options], outputs = [I_im_1, I_im_2, I_im_3, I_im_4])

        with gr.TabItem("Design your background music and image effects"):
            btn_view_images = gr.Button(value="View all designs!")
            with gr.Row():
                cb_house   = gr.Checkbox(value = 'True', label = "House")
                cb_dinning = gr.Checkbox(value = 'True', label = "Dinning Room")
                cb_kitchen = gr.Checkbox(value = 'True', label = "Kitchen")
            
            with gr.Row():
                img_house   = gr.Image()
                img_dinning = gr.Image()
                img_kitchen = gr.Image()
            
            with gr.Row():
                cb_living   = gr.Checkbox(value = 'True', label = "Living Roon")
                cb_bath     = gr.Checkbox(value = 'True', label = "Bathroom")
                cb_bedroom1 = gr.Checkbox(value = 'True', label = "Bedroom One")
            
            with gr.Row():
                img_living   = gr.Image()
                img_bath     = gr.Image()
                img_bedroom1 = gr.Image()

            with gr.Row():
                cb_bedroom2 = gr.Checkbox(value = 'True', label = "Bedroom Two")
                cb_bedroom3 = gr.Checkbox(value = 'True', label = "Bedroom Three")
                cb_bedroom4 = gr.Checkbox(value = 'True', label = "Bedroom Four")
            
            with gr.Row():
                img_bedroom2 = gr.Image()
                img_bedroom3 = gr.Image()
                img_bedroom4 = gr.Image()


            with gr.Row():
                txt   = gr.Textbox(label="Enter Your Description")
                audio = gr.Audio(source="microphone")
            with gr.Row():
                btn1 = gr.Button(value="Clear")
                btn2 = gr.Button(value="Enhance Your Sentence?")
                btn3 = gr.Button(value="Submit")

            btn1.click(clear, inputs=[], outputs=[txt])
            btn2.click(enhance_your_sentence3, inputs = [txt], outputs = [txt])
            btn3.click(generate_audio, inputs=[txt], outputs=[audio])
            btn_view_images.click(viewAllimages, inputs = [], outputs = [img_house, img_dinning, img_kitchen, img_living, img_bath, img_bedroom1, img_bedroom2, img_bedroom3, img_bedroom4])

            with gr.Row():
                cb_audio  = gr.Checkbox(value = 'True', label = "audio ")
                cb_effect = gr.Checkbox(value = 'True', label = "effect")
            
            final_video = gr.Video()
            btn_video = gr.Button(value="Generate Your Video!")
            btn_video.click(generateVideo, inputs = [cb_house, cb_dinning, cb_kitchen, cb_living, cb_bath, cb_bedroom1, cb_bedroom2, cb_bedroom3, cb_bedroom4, cb_audio, cb_effect], outputs = [final_video])
        
        with gr.TabItem("QR Code"):
            with gr.Row():
                QR_btn = gr.Button(value = 'Generate Your QR code!')
            
            QR_img  = gr.Image(height = 256, width = 1280, container = False)
            QR_link = gr.Textbox(label = 'Copy your link', show_copy_button = True)
            QR_btn.click(generateQRcode, inputs = [cb_audio, cb_effect], outputs = [QR_img, QR_link])


    
if __name__ == "__main__":
    demo.queue().launch(share=True, debug=True)