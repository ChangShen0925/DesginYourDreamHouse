import os 
os.system('pip install -r requirement.txt')
from diffusers import DiffusionPipeline
import torch
import qrcode
import cv2
import numpy as np
import webbrowser
from moviepy.editor import VideoFileClip
from IPython.display import HTML, display
import gradio as gr
import random
import time





import os

import torch


from chatGPT import enhance_your_sentence, enhance_your_sentence2, enhance_your_sentence3
from Additional_elements import *

from diffusers.pipeline_utils import DiffusionPipeline



os.makedirs(os.getcwd(), exist_ok=True)


def clear():
    return None

def newWebsite(media, URL):

    url = ''
    if media == 'Twitter':
        url+=f"https://twitter.com/intent/tweet?url={URL}"
    elif media == "Facebook":
        url+=f"https://www.facebook.com/sharer/sharer.php?u={URL}"
    elif media == "Weibo":
        url+=f"http://service.weibo.com/share/share.php?url={URL}"
    else:
        url+=f"https://reddit.com/submit?url={URL}"

    return gr.Button.update(value = 'share', scale = 1, link = url)



def generateQRcode(audio, effect, filename, request: gr.Request):
    url = str(request.headers['origin']) + '/file=' + os.getcwd()
    if audio:
        url+=f'/{filename}/output_with_audio.mp4'
    elif effect:
        url+=f'/{filename}/output_with_effects.mp4'
    else:
        url+=f'/{filename}/output.mp4'
    qr = qrcode.QRCode(
    version=1,
    error_correction=qrcode.constants.ERROR_CORRECT_L,
    box_size=10,
    border=4,
    )
    qr.add_data(url)
    qr.make(fit=True)
    qr_code = qr.make_image(fill_color="black", back_color="white")
    qr_code.save(f'{filename}/qrcode.png')

    return Image.open(f'{filename}/qrcode.png'), url, url


def viewAllimages(filename):
    output = [Image.open(f'{filename}/House.png'), None, None, None, None, None, None, None, None,]
    for i in os.listdir(filename):
        if ('.png' in i and 'House' not in i) and 'qrcode.png' != i:
            for j in range(len(output)):
                if output[j] is None:
                    output[j] = Image.open(f'{filename}/{i}')
                    break

    return output[0], output[1], output[2], output[3], output[4], output[5], output[6], output[7], output[8]

def generateVideo(cb1, cb2, cb3, cb4, cb5, cb6, cb7, cb8, cb9, audio, effect, filename):
    if audio and effect:
        AddImageEffects([cb1, cb2, cb3, cb4, cb5, cb6, cb7, cb8, cb9], viewAllimages(filename), filename)
        audio2video(VideoFileClip(f"{filename}/output_with_effects.mp4"), filename)
        return os.getcwd() + f"/{filename}/output_with_audio.mp4"
    elif audio:
        JustImage([cb1, cb2, cb3, cb4, cb5, cb6, cb7, cb8, cb9], viewAllimages(filename), filename)
        audio2video(VideoFileClip(f"{filename}/output.mp4"), filename)
        return os.getcwd() + f"/{filename}/output_with_audio.mp4"
    elif effect:
        AddImageEffects([cb1, cb2, cb3, cb4, cb5, cb6, cb7, cb8, cb9], viewAllimages(filename), filename)
        return os.getcwd() + f"/{filename}/output_with_effects.mp4"
    else:
        JustImage([cb1, cb2, cb3, cb4, cb5, cb6, cb7, cb8, cb9], viewAllimages(filename), filename)
        return os.getcwd() + f"/{filename}/output.mp4"
    

def viewCurrentImage(cata, filename):
    if cata == 'bedroom':
        for i in os.listdir(filename):
            if 'bedroom1.png' == i:
                return Image.open(f'{filename}/{i}'), gr.Image.update(visible=True), gr.Image.update(visible=True), gr.Image.update(visible=True)
        return None, gr.Image.update(visible=True), gr.Image.update(visible=True), gr.Image.update(visible=True)
    else:
        for i in os.listdir(filename):
            if f'{cata}.png' == i:
                return Image.open(f'{filename}/{i}'), gr.Image.update(visible=False), gr.Image.update(visible=False), gr.Image.update(visible=False)
        return None, gr.Image.update(visible=False), gr.Image.update(visible=False), gr.Image.update(visible=False)
       
    
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
        if Style == "Asian":
            Style = "Asain"
        return Image.open(os.getcwd() + '/HouseStyle/' + Style + '/' + Style +'1.png'), Image.open(os.getcwd() + '/HouseStyle/' + Style + '/' + Style +'2.png'), Image.open(os.getcwd() + '/HouseStyle/' + Style + '/' + Style +'3.png'), Image.open(os.getcwd() + '/HouseStyle/' + Style + '/' + Style +'4.png'), Image.open(os.getcwd() + '/HouseStyle/' + Style + '/' + Style +'5.png')

def generator(Prompt):
    pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
    pipe.to("cuda")
    images = pipe(prompt=Prompt, num_inference_steps=20).images[0]
    torch.cuda.empty_cache()
    return images

    


def generateImage(Prompt, style):
    PROMPT = Prompt
    if len(style) == 0:
        style = ''
    frontview = generator("The front view of A house in style of " + style +  "under the blue sky which " + Prompt)
 
    torch.cuda.empty_cache()
    timestamp = time.time()

    my_list = '1234567890qwertyuiopasdfghjklzxcvbnm'
    fileName = ''
    for i in random.sample(my_list, 12):
        fileName+=i
    fileName+=str(int(timestamp))
    os.system(f"mkdir {fileName}")
    frontview.save(f'{fileName}/House.png')
    return frontview, fileName, Prompt


def generate_audio(prompt, filename):
    text2audio(prompt, filename)
    torch.cuda.empty_cache()
    return os.getcwd()+f'/{filename}/techno.wav'



def generate_room_inside(cata, other, filename, PROMPT):
   
    imgList =  os.listdir(filename)

    if cata == '':
        return "Please at least select the room!", None
    else:
        result = generator("The panorama of the room of " + cata +  " which is "+ other + "in a house of " + PROMPT)
        if cata != 'bedroom':
            result.save(f'{filename}/{cata}.png')
            return "Do you like the image?", result, gr.Image.update(visible=False), gr.Image.update(visible=False), gr.Image.update(visible=False)
        else:
            if 'bedroom1.png' not in imgList:
                result.save(f'{filename}/bedroom1.png')
                return "Do you like the image?", result, gr.Image.update(visible=True), gr.Image.update(visible=True), gr.Image.update(visible=True)
            elif 'bedroom2.png' not in imgList:
                result.save(f'{filename}/bedroom2.png')
                return "Do you like the image?", gr.Image.update(visible=True), result, gr.Image.update(visible=True), gr.Image.update(visible=True)
            elif 'bedroom3.png' not in imgList:
                result.save(f'{filename}/bedroom3.png')
                return "Do you like the image?", gr.Image.update(visible=True), gr.Image.update(visible=True), result, gr.Image.update(visible=True)
            elif 'bedroom4.png' not in imgList:
                result.save(f'{filename}/bedroom4.png')
                return "Do you like the image?", gr.Image.update(visible=True), gr.Image.update(visible=True), gr.Image.update(visible=True), result
            else:
                Image.open(f'{filename}/bedroom2.png').save(f'{filename}/bedroom1.png')
                Image.open(f'{filename}/bedroom3.png').save(f'{filename}/bedroom2.png')
                Image.open(f'{filename}/bedroom4.png').save(f'{filename}/bedroom3.png')
                result.save(f'{filename}/bedroom4.png')
       
                return "Do you like the image?", Image.open(f'{filename}/bedroom1.png'), Image.open(f'{filename}/bedroom2.png'), Image.open(f'{filename}/bedroom3.png'), Image.open(f'{filename}/bedroom4.png')





with gr.Blocks(theme='Taithrah/Minimal') as demo:
    fileName = gr.Textbox(visible=False)
    prompt = gr.Textbox(visible=False)
    url = gr.Textbox(visible=False)
    with gr.Tabs():
        gr.Markdown("""
        <html>
        <head>
        <style>
            .centered-text {
            text-align: center;
            font-weight: bold;
            }
        </style>
        </head>
        <body>

        <p style="font-size: 50px; text-align: center; color: lightblue;">Design Your Dream House!</p>
        </body>
        </html>
        """)
        with gr.TabItem("Design your House(Outside)"):
            with gr.Row():
                options = gr.Dropdown(choices = ["Asian", "Gothic", "Modern", "Neoclassical", "Nordic", "Other, please specify"], value = "Asian", multiselect=False, label="Choose your House style")
                txt     = gr.Textbox(label="House Style", visible=False)
            #btn1 = gr.Button(value="View Some example!")
            with gr.Row():
                img1    =  gr.Image(value = Image.open(os.getcwd() + '/HouseStyle/Asain/Asain1.png'), label = "Preview 1")
                img2    =  gr.Image(value = Image.open(os.getcwd() + '/HouseStyle/Asain/Asain2.png'), label = "Preview 2")
                img3    =  gr.Image(value = Image.open(os.getcwd() + '/HouseStyle/Asain/Asain3.png'), label = "Preview 3")
                img4    =  gr.Image(value = Image.open(os.getcwd() + '/HouseStyle/Asain/Asain4.png'), label = "Preview 4")
                img5    =  gr.Image(value = Image.open(os.getcwd() + '/HouseStyle/Asain/Asain5.png'), label = "Preview 5")

            with gr.Row():
                txt1 = gr.Textbox(label="Enter Your Description", lines=3)

            with gr.Row():
                btn2 = gr.Button(variant="primary", value="Clear")
                btn3 = gr.Button(variant="primary", value="Enhance Your Sentence?")
                btn4 = gr.Button(variant="primary", value="Submit")
                # video_1 = gr.Video()
            img    =  gr.Image(height = 900, width = 1600)
            #btn1.click(viewExample, inputs = [options], outputs = [img1, img2, img3, img4, img5])
            btn2.click(clear, inputs=[], outputs=[txt1])
            btn3.click(enhance_your_sentence, inputs = [options, txt, txt1], outputs = [txt1])
            btn4.click(generateImage, inputs=[txt1, options], outputs=[img, fileName, prompt])
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
            btn3.click(generate_room_inside, inputs=[room_options, I_txt, fileName, prompt], outputs=[I_txt_2, I_im_1, I_im_2, I_im_3, I_im_4])
            room_options.change(viewCurrentImage, inputs = [room_options, fileName], outputs = [I_im_1, I_im_2, I_im_3, I_im_4])

        with gr.TabItem("Design your background music and image effects"):
            btn_view_images = gr.Button(value="View all designs!")
            with gr.Row():
                cb_house   = gr.Checkbox(value = 'True')
                cb_dinning = gr.Checkbox(value = 'True')
                cb_kitchen = gr.Checkbox(value = 'True')
            
            with gr.Row():
                img_house   = gr.Image()
                img_dinning = gr.Image()
                img_kitchen = gr.Image()
            
            with gr.Row():
                cb_living   = gr.Checkbox(value = 'True')
                cb_bath     = gr.Checkbox(value = 'True')
                cb_bedroom1 = gr.Checkbox(value = 'True')
            
            with gr.Row():
                img_living   = gr.Image()
                img_bath     = gr.Image()
                img_bedroom1 = gr.Image()

            with gr.Row():
                cb_bedroom2 = gr.Checkbox(value = 'True')
                cb_bedroom3 = gr.Checkbox(value = 'True')
                cb_bedroom4 = gr.Checkbox(value = 'True')
            
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
            btn3.click(generate_audio, inputs=[txt, fileName], outputs=[audio])
            btn_view_images.click(viewAllimages, inputs = [fileName], outputs = [img_house, img_dinning, img_kitchen, img_living, img_bath, img_bedroom1, img_bedroom2, img_bedroom3, img_bedroom4])

            with gr.Row():
                cb_audio  = gr.Checkbox(value = 'True', label = "audio ")
                cb_effect = gr.Checkbox(value = 'True', label = "effect")
            
            final_video = gr.Video()
            btn_video = gr.Button(value="Generate Your Video!")
            btn_video.click(generateVideo, inputs = [cb_house, cb_dinning, cb_kitchen, cb_living, cb_bath, cb_bedroom1, cb_bedroom2, cb_bedroom3, cb_bedroom4, cb_audio, cb_effect, fileName], outputs = [final_video])
        
        with gr.TabItem("QR Code"):
            with gr.Row():
                QR_btn = gr.Button(value = 'Generate Your QR code!')
            
            QR_img  = gr.Image(height = 256, width = 1280, container = False)
            with gr.Row():
                QR_link   = gr.Textbox(label = 'Copy your link', show_copy_button = True, scale = 10)
                QR_share  = gr.Dropdown([ "Twitter", "Facebook", "Weibo", "Reddit"], multiselect=False, label="Share", scale = 1)
                share_btn = gr.Button(value = 'share', scale = 1, link = "https://www.google.com/")

            QR_btn.click(generateQRcode, inputs = [cb_audio, cb_effect, fileName], outputs = [QR_img, QR_link, url])
            QR_img.change(newWebsite, inputs = [QR_share, url], outputs = [share_btn])
            QR_share.change(newWebsite, inputs = [QR_share, url], outputs = [share_btn])

    
if __name__ == "__main__":
    demo.queue().launch(share=True, debug=True)