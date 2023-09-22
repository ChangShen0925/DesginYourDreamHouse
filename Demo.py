import os 
os.system('pip install -r requirement.txt')
from diffusers import DiffusionPipeline
import torch
import qrcode
import cv2
import numpy as np
import webbrowser
from moviepy.editor import VideoFileClip
import gradio as gr
import random
import time
import json




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
    if filename == '':
        return None, None, None
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
    if filename == '':
        return None, None, None, None, None, None, None, None
    output = [Image.open(f'{filename}/House.png'), None, None, None, None, None, None, None]
    for i in os.listdir(filename):
        if ('.png' in i and 'House' not in i) and 'qrcode.png' != i:
            for j in range(len(output)):
                if output[j] is None:
                    output[j] = Image.open(f'{filename}/{i}')
                    break

    return output[0], output[1], output[2], output[3], output[4], output[5], output[6], output[7]

def generateVideo(cb1, cb2, cb3, cb4, cb5, cb6, cb7, cb8, audio, effect, filename):
    if filename == '':
        return None
    if audio and effect:
        AddImageEffects([cb1, cb2, cb3, cb4, cb5, cb6, cb7, cb8], viewAllimages(filename), filename)
        audio2video(VideoFileClip(f"{filename}/output_with_effects.mp4"), filename)
        return os.getcwd() + f"/{filename}/output_with_audio.mp4"
    elif audio:
        JustImage([cb1, cb2, cb3, cb4, cb5, cb6, cb7, cb8], viewAllimages(filename), filename)
        audio2video(VideoFileClip(f"{filename}/output.mp4"), filename)
        return os.getcwd() + f"/{filename}/output_with_audio.mp4"
    elif effect:
        AddImageEffects([cb1, cb2, cb3, cb4, cb5, cb6, cb7, cb8], viewAllimages(filename), filename)
        return os.getcwd() + f"/{filename}/output_with_effects.mp4"
    else:
        JustImage([cb1, cb2, cb3, cb4, cb5, cb6, cb7, cb8], viewAllimages(filename), filename)
        return os.getcwd() + f"/{filename}/output.mp4"
    

def viewCurrentImage(cata, filename):
    if filename == '' and cata != 'bedroom':
        return None, gr.Image.update(visible=False), gr.Image.update(visible=False)
    elif filename == '':
        return None, gr.Image.update(visible=True), gr.Image.update(visible=True)
    if cata == 'bedroom':
        for i in os.listdir(filename):
            if 'bedroom1.png' == i:
                return Image.open(f'{filename}/{i}'), gr.Image.update(visible=True), gr.Image.update(visible=True)
        return None, gr.Image.update(visible=True), gr.Image.update(visible=True)
    else:
        for i in os.listdir(filename):
            if f'{cata}.png' == i:
                return Image.open(f'{filename}/{i}'), gr.Image.update(visible=False), gr.Image.update(visible=False)
        return None, gr.Image.update(visible=False), gr.Image.update(visible=False)
       
    
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

    


def generateImage(dic, Prompt, style):
    room_dic = json.loads(dic)
    room_dic['house'] = "The front view of A house in style of " + style +  "under the blue sky which " + Prompt
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
    return frontview, fileName, json.dumps(room_dic)


def generate_audio(prompt, filename):
    if filename == '':
        return None
    text2audio(prompt, filename)
    torch.cuda.empty_cache()
    return os.getcwd()+f'/{filename}/techno.wav'



def generate_room_inside(cata, other, filename, PROMPT):
    room_dic = json.loads(PROMPT)
    if filename == '':
        return None, None, None

    imgList =  os.listdir(filename)
    result = generator("The panorama of the room of " + cata +  " which is "+ other + "in a house of " + room_dic['house'])
    if cata != 'bedroom':
        result.save(f'{filename}/{cata}.png')
        return result, gr.Image.update(visible=False), gr.Image.update(visible=False)
    else:
        if 'bedroom1.png' not in imgList:
            result.save(f'{filename}/bedroom1.png')
            return result, gr.Image.update(visible=True), gr.Image.update(visible=True)
        elif 'bedroom2.png' not in imgList:
            result.save(f'{filename}/bedroom2.png')
            return gr.Image.update(visible=True), result, gr.Image.update(visible=True)
        elif 'bedroom3.png' not in imgList:
            result.save(f'{filename}/bedroom3.png')
            return gr.Image.update(visible=True), gr.Image.update(visible=True), result
        else:
            Image.open(f'{filename}/bedroom2.png').save(f'{filename}/bedroom1.png')
            Image.open(f'{filename}/bedroom3.png').save(f'{filename}/bedroom2.png')
            result.save(f'{filename}/bedroom3.png')
    
            return Image.open(f'{filename}/bedroom1.png'), Image.open(f'{filename}/bedroom2.png'), Image.open(f'{filename}/bedroom3.png')




css="""
    .select button.selected{
        background-color: #4CAF50;
        font-size: 17px !important;
        width: 374px;
        margin: 0px 1px;
    }
    .hover button:hover{
        background-color: #FFFFFF;
        font-size: 17px !important;
        color: black;
        transition: all 1s ease;
        width: 374px;
        margin: 0px 1px;
    }
    .slient button{
        background-color: #8aedd3;
        font-size: 17px !important;
        color: black;
        width: 374px;
        margin: 0px 1px;
    }
    """

with gr.Blocks(theme='Taithrah/Minimal', css = css) as demo:
    fileName = gr.Textbox(visible=False)
    prompt = gr.Textbox(value = '{"house": null, "living room": null, "kitchen": null, "dining room": null, "bathroom": null, "bedroom": [null, null, null, null]}',visible=False)
    url = gr.Textbox(visible=False)
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

        <p style="font-size: 50px; text-align: center; color: blue;">Design Your Dream House!</p>
        </body>
        </html>
        """)
    with gr.Tabs(elem_classes=["hover", "select", "slient"]):
        with gr.TabItem("Step one: Design your house exterior look!"):
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
            img    =  gr.Image(height = 512, width = 1536)
            #btn1.click(viewExample, inputs = [options], outputs = [img1, img2, img3, img4, img5])
            btn2.click(clear, inputs=[], outputs=[txt1])
            btn3.click(enhance_your_sentence, inputs = [options, txt, txt1], outputs = [txt1])
            btn4.click(generateImage, inputs=[prompt, txt1, options], outputs=[img, fileName, prompt])
            options.change(change_options, inputs = [options], outputs = [txt])
            options.change(viewExample, inputs = [options], outputs = [img1, img2, img3, img4, img5])

        with gr.TabItem("Step two: Plan your house's interior style") as tab2:
            room_options = gr.Dropdown([ "dinning room", "kitchen", "living room", "bathroom", "bedroom"],  value = "dinning room", multiselect=False, label="Choose your design")
            I_txt = gr.Textbox(label="Enter Your Description", line = 3)
            with gr.Row():
                btn1 = gr.Button(variant="primary", value="Clear")
                btn2 = gr.Button(variant="primary", value="Enhance Your Sentence?")
                btn3 = gr.Button(variant="primary", value="Submit")
            

            with gr.Row():
                I_im_1 = gr.Image()
                I_im_2 = gr.Image(visible = False)
                I_im_3 = gr.Image(visible = False)


            
            btn1.click(clear, inputs=[], outputs=[I_txt])
            btn2.click(enhance_your_sentence2, inputs = [room_options, I_txt], outputs = [I_txt])
            btn3.click(generate_room_inside, inputs=[room_options, I_txt, fileName, prompt], outputs=[I_im_1, I_im_2, I_im_3])
            room_options.change(viewCurrentImage, inputs = [room_options, fileName], outputs = [I_im_1, I_im_2, I_im_3])
            tab2.select(viewCurrentImage, inputs = [room_options, fileName], outputs = [I_im_1, I_im_2, I_im_3])

        with gr.TabItem("Step three: Create your audio-visual effects") as tab3:
            gr.Markdown("""
                        <div>
                            <p style="font-size: 25px; color: blue;">View All your current Designs!</p>
                        </div>
                        """)
            with gr.Row():
                cb_house   = gr.Checkbox(label = "Chosen", value = 'True')
                cb_dinning = gr.Checkbox(label = "Chosen", value = 'True')
                cb_kitchen = gr.Checkbox(label = "Chosen", value = 'True')
                cb_living  = gr.Checkbox(label = "Chosen", value = 'True')
            
            with gr.Row():
                img_house   = gr.Image()
                img_dinning = gr.Image()
                img_kitchen = gr.Image()
                img_living   = gr.Image()
            
            with gr.Row():
                
                cb_bath     = gr.Checkbox(label = "Chosen", value = 'True')
                cb_bedroom1 = gr.Checkbox(label = "Chosen", value = 'True')
                cb_bedroom2 = gr.Checkbox(label = "Chosen", value = 'True')
                cb_bedroom3 = gr.Checkbox(label = "Chosen", value = 'True')
            
            with gr.Row():
                
                img_bath     = gr.Image()
                img_bedroom1 = gr.Image()
                img_bedroom2 = gr.Image()
                img_bedroom3 = gr.Image()
                


            gr.Markdown("""
                        <div>
                            <p style="font-size: 25px; color: blue;">Design Your own Music!</p>
                        </div>
                        """)
            with gr.Row():
                txt   = gr.Textbox(label="Enter Your Description")
                audio = gr.Audio(source="microphone")
            with gr.Row():
                btn1 = gr.Button(variant="primary", value="Clear")
                btn2 = gr.Button(variant="primary", value="Enhance Your Sentence?")
                btn3 = gr.Button(variant="primary", value="Submit")

            btn1.click(clear, inputs=[], outputs=[txt])
            btn2.click(enhance_your_sentence3, inputs = [txt], outputs = [txt])
            btn3.click(generate_audio, inputs=[txt, fileName], outputs=[audio])
            tab3.select(viewAllimages, inputs = [fileName], outputs = [img_house, img_dinning, img_kitchen, img_living, img_bath, img_bedroom1, img_bedroom2, img_bedroom3])

            gr.Markdown("""
                        <div>
                            <p style="font-size: 25px; color: blue;">Generate the Video!</p>
                        </div>
                        """)
            
            with gr.Row():
                cb_audio  = gr.Checkbox(value = 'True', label = "audio ")
                cb_effect = gr.Checkbox(value = 'True', label = "effect")
            
            final_video = gr.Video(height = 512, width = 1536)
            btn_video = gr.Button(value="Generate Your Video!")
            btn_video.click(generateVideo, inputs = [cb_house, cb_dinning, cb_kitchen, cb_living, cb_bath, cb_bedroom1, cb_bedroom2, cb_bedroom3, cb_audio, cb_effect, fileName], outputs = [final_video])
        
        with gr.TabItem("Step four: Generate your unique QR code now"):
            with gr.Row():
                QR_btn = gr.Button(value = 'Generate Your QR code!')
            
            QR_img  = gr.Image(height = 512, width = 1536, container = False)
            with gr.Row():
                QR_link   = gr.Textbox(label = 'Copy your link', show_copy_button = True, scale = 10)
                QR_share  = gr.Dropdown([ "Twitter", "Facebook", "Weibo", "Reddit"], multiselect=False, label="Share", scale = 1)
                share_btn = gr.Button(value = 'share', scale = 1, link = "https://www.google.com/")

            QR_btn.click(generateQRcode, inputs = [cb_audio, cb_effect, fileName], outputs = [QR_img, QR_link, url])
            QR_img.change(newWebsite, inputs = [QR_share, url], outputs = [share_btn])
            QR_share.change(newWebsite, inputs = [QR_share, url], outputs = [share_btn])

    
if __name__ == "__main__":
    demo.queue().launch(share=True, debug=True)