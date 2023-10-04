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
from TTS import *
from transformers import pipeline
import numpy as np
import pyshorteners
import os

import torch


from chatGPT import *
from Additional_elements import *

from diffusers.pipeline_utils import DiffusionPipeline



os.makedirs(os.getcwd(), exist_ok=True)

transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-base.en")

def transcribe(audio):
    try:
        sr, y = audio
        y = y.astype(np.float32)
        y /= np.max(np.abs(y))

        return transcriber({"sampling_rate": sr, "raw": y})["text"]
    except:
        return None

def clear():
    return None

def hideColumns():
    return gr.Column.update(visible = False), gr.Column.update(visible = False), gr.Textbox.update(value = "00")
def showSharePage(boolean):
    if boolean[0] == "0":
        return gr.Column.update(visible = True), gr.Column.update(visible = False), gr.Textbox.update(value = "10")
    else:
        return gr.Column.update(visible = False), gr.Column.update(visible = False), gr.Textbox.update(value = "00")

def showQRcodePage(boolean):
    if boolean[1] == "0":
        return gr.Column.update(visible = True), gr.Column.update(visible = False), gr.Textbox.update("01")
    else:
        return gr.Column.update(visible = False), gr.Column.update(visible = False), gr.Textbox.update("00")
    

def generateDescription(cb_house, cb_dinning, cb_kitchen, cb_living, cb_bath, cb_bedroom1, cb_bedroom2, cb_bedroom3, prompts, filename, voice):
    room_dic = json.loads(prompts)
    if filename == '':
        return None, None
    chosen_list = [cb_house, cb_dinning, cb_kitchen, cb_living, cb_bath, cb_bedroom1, cb_bedroom2, cb_bedroom3]
    skip_list = []
    for i in range(len(chosen_list)):
        if not chosen_list[i]:
            skip_list.append(i)
    description = [enhance_your_sentence4('This is the exterior look of the house, ' + room_dic['house'])]
    index = 1
    for i in os.listdir(filename):
        if ('.png' in i and 'House' not in i) and 'qrcode.png' != i:
            if index not in skip_list:
                if ('1' in i or '2' in i) or '3' in i:
                    description.append(enhance_your_sentence4('And then we can hou go the next room.' + room_dic['bedroom'][int(i[-5]) - 1]))
                else:
                    description.append(enhance_your_sentence4('And then we can hou go the next room.' + room_dic[i[0:-4]]))
            index+=1
    
    description[-1]+="This is the end of the house"

    
    introduction = ''
    for i in range(len(description)):
        introduction+=description[i] + ' '
        textToSpeech(description[i], filename, voice, i)
        
    mergeAll(filename)
    return introduction, os.getcwd()+f'/{filename}/bark_out.wav'

def generateQRcode(audio, effect, description, filename, request: gr.Request):
    if filename == '':
        return None, None, None, None, None
    if 'techno.wav' not in os.listdir(filename):
        audio = False
    if 'bark_out.wav' not in os.listdir(filename):
        description = False
    url = str(request.headers['origin']) + '/file=' + os.getcwd()
    if audio:
        url+=f'/{filename}/output_with_audio.mp4'
    elif description:
        url+=f'/{filename}/output_with_speech.mp4'
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

    

    twitter_url  = f"https://twitter.com/intent/tweet?url={url}"
    facebook_url = f"https://www.facebook.com/sharer/sharer.php?u={url}"
    weibo_url    = f"http://service.weibo.com/share/share.php?url={url}"
    reddit_url   = f"https://reddit.com/submit?url={url}"

    return Image.open(f'{filename}/qrcode.png'), gr.HTML.update(f"""
                              <html>
                              <head>
                              <style>
                              .center {{
                                display: block;
                                margin-left: auto;
                                margin-right: auto;
                              }}
                              </style>
                              </head>
                              <body>
                              <a href="{facebook_url}" target="_blank">
                                <img src="https://cdn3.iconfinder.com/data/icons/picons-social/57/46-facebook-1024.png" class="center" style="width: 35px; height: 35px">
                              </a>
                              </body>
                              </html>
                              """), gr.HTML.update(f"""
                              <html>
                              <head>
                              <style>
                              .center {{
                                display: block;
                                margin-left: auto;
                                margin-right: auto;
                              }}
                              </style>
                              </head>
                              <body>
                              <a href="{reddit_url}" target="_blank">
                                <img src="https://static-00.iconduck.com/assets.00/reddit-logo-icon-2048x2048-vtzhwa71.png" class="center" style="width: 35px; height: 35px">
                              </a>
                              </body>
                              </html>
                              """), gr.HTML.update(f"""
                              <html>
                              <head>
                              <style>
                              .center {{
                                display: block;
                                margin-left: auto;
                                margin-right: auto;
                              }}
                              </style>
                              </head>
                              <body>
                              <a href="{weibo_url}" target="_blank">
                                <img src="https://cdn.freebiesupply.com/images/large/2x/weibo-logo-black-transparent.png" class="center" style="width: 35px; height: 35px">
                              </a>
                              </body>
                              </html>
                              """), gr.HTML.update(f"""
                              <html>
                              <head>
                              <style>
                              .center {{
                                display: block;
                                margin-left: auto;
                                margin-right: auto;
                              }}
                              </style>
                              </head>
                              <body>
                              <a href="{twitter_url}" target="_blank">
                                <img src="https://cdn3.iconfinder.com/data/icons/picons-social/57/43-twitter-1024.png" class="center" style="width: 35px; height: 35px">
                              </a>
                              </body>
                              </html>
                              """), gr.Textbox.update(pyshorteners.Shortener().tinyurl.short(url))




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

def generateVideo(cb1, cb2, cb3, cb4, cb5, cb6, cb7, cb8, audio, effect, description, filename):
    if filename == '':
        return None
    if 'techno.wav' not in os.listdir(filename):
        audio = False
    if 'bark_out.wav' not in os.listdir(filename):
        description = False
    if effect:
        AddImageEffects([cb1, cb2, cb3, cb4, cb5, cb6, cb7, cb8], viewAllimages(filename), filename, description)
        if description:
            speech2video(VideoFileClip(f"{filename}/output_with_effects.mp4"), filename)
            if audio:
                audio2video(VideoFileClip(f"{filename}/output_with_speech.mp4"), filename, description)
                return os.getcwd() + f"/{filename}/output_with_audio.mp4"
            else:
                return os.getcwd() + f"/{filename}/output_with_speech.mp4"
        else:
            if audio:
                audio2video(VideoFileClip(f"{filename}/output_with_effects.mp4"), filename, description)
                return os.getcwd() + f"/{filename}/output_with_audio.mp4"
            else:
                return os.getcwd() + f"/{filename}/output_with_effects.mp4"
            
    else:
        JustImage([cb1, cb2, cb3, cb4, cb5, cb6, cb7, cb8], viewAllimages(filename), filename, description)
        if description:
            speech2video(VideoFileClip(f"{filename}/output.mp4"), filename)
            if audio:
                audio2video(VideoFileClip(f"{filename}/output_with_speech.mp4"), filename, description)
                return os.getcwd() + f"/{filename}/output_with_audio.mp4"
            else:
                return os.getcwd() + f"/{filename}/output_with_speech.mp4"
        else:
            if audio:
                audio2video(VideoFileClip(f"{filename}/output.mp4"), filename, description)
                return os.getcwd() + f"/{filename}/output_with_audio.mp4"
            else:
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
    room_dic['house'] = "The front view of A house in style of " + style +  "which " + Prompt
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
        return None, None, None, None

    imgList =  os.listdir(filename)
    result = generator("The panorama of the room of " + cata +  " which is "+ other + "in a house of " + room_dic['house'])
    if cata != 'bedroom':
        room_dic[cata] = "The panorama of the room of " + cata +  " which is "+ other + "in a house of " + room_dic['house']
        result.save(f'{filename}/{cata}.png')
        return result, gr.Image.update(visible=False), gr.Image.update(visible=False), json.dumps(room_dic)
    else:
        if 'bedroom1.png' not in imgList:
            room_dic[cata][0] = "The panorama of the room of " + cata +  " which is "+ other + "in a house of " + room_dic['house']
            result.save(f'{filename}/bedroom1.png')
            return result, gr.Image.update(visible=True), gr.Image.update(visible=True), json.dumps(room_dic)
        elif 'bedroom2.png' not in imgList:
            room_dic[cata][1] = "The panorama of the room of " + cata +  " which is "+ other + "in a house of " + room_dic['house']
            result.save(f'{filename}/bedroom2.png')
            return gr.Image.update(visible=True), result, gr.Image.update(visible=True), json.dumps(room_dic)
        elif 'bedroom3.png' not in imgList:
            room_dic[cata][2] = "The panorama of the room of " + cata +  " which is "+ other + "in a house of " + room_dic['house']
            result.save(f'{filename}/bedroom3.png')
            return gr.Image.update(visible=True), gr.Image.update(visible=True), result, json.dumps(room_dic)
        else:
            room_dic[cata][0] = room_dic[cata][1]
            Image.open(f'{filename}/bedroom2.png').save(f'{filename}/bedroom1.png')
            room_dic[cata][1] = room_dic[cata][2]
            Image.open(f'{filename}/bedroom3.png').save(f'{filename}/bedroom2.png')
            room_dic[cata][2] = "The panorama of the room of " + cata +  " which is "+ other + "in a house of " + room_dic['house']
            result.save(f'{filename}/bedroom3.png')
    
            return Image.open(f'{filename}/bedroom1.png'), Image.open(f'{filename}/bedroom2.png'), Image.open(f'{filename}/bedroom3.png'), json.dumps(room_dic)




css="""
    #image_background {background-color: #000000}
    #button_close {position: relative; left:450px; font-size: 30px; color: black !important; max-width: 1px !important; max-height: 30px !important; min-width: 1px !important; min-height: 30px !important;}

    #poppage {
      position: relative;
      height: 250px;
      bottom: 530px;
      left: 300px;
      max-width: 500px;
      margin: auto;
      }

    #button_submit {background: #FF7712; color: #FFFFFF !important; max-width: 25%; min-width: 25%;} 
    #button_enhance {border-color: #FF7712; color: #FF7712; background: #FFFFFF; max-width: 25%; min-width: 25%;} 
    #button_clear {border-color: #FF7712; color: #FF7712; background: #FFFFFF; max-width: 12.5%; min-width: 12.5%;} 

    #audio_button_submit {position: relative; background: #FF7712; color: #FFFFFF !important; top: 30px; max-width: 25%; min-width: 25%;}
    #audio_button_enhance {position: relative; border-color: #FF7712; color: #FF7712; background: #FFFFFF; top: 30px; max-width: 25%; min-width: 25%;} 
    #audio_button_clear {position: relative; border-color: #FF7712; color: #FF7712; background: #FFFFFF; top: 30px; max-width: 12.5%; min-width: 12.5%;} 
    #audio_id {height: 34px;}
    #checkbox_id {max-width: 16.7%; min-width: 16.7%;}
    .select button.selected{
        background-color: #FF7712;
        width: 25%;
        font-size: 17px !important;
        color: white;

    }
 
    .slient button{
        background-color: ##F4F6FA;
        width: 25%;
        font-size: 17px !important;
        color: black;
    }
    .select2 button.selected{
        background-color: #FF7712;
        font-size: 17px !important;
        width: 100%
        color: white;

    }

    .slient2 button{
        background-color: ##F4F6FA;
        font-size: 17px !important;
        width: 100%;
        color: black;
    }
    """

with gr.Blocks(css = css) as demo:
    
    gr.HTML("""
            <html>
            <head>
            <style>
            .container {
                position: relative;
            }

            .top-left-image {
                position: absolute;
                top: 11px;
                left: 11px;
            }
            </style>
            </head>
            <body>
            <div class="container">
            <img src="https://raw.githubusercontent.com/ChangShen0925/DesignYourDreamHouse/main/Rectangle.png" alt="Your Image" style="width: 1512px; height: 47px;">
            <img class="top-left-image" src="https://raw.githubusercontent.com/ChangShen0925/DesignYourDreamHouse/main/logo.png" alt="Your Image" style="width: 120px; height: 20px;">
            </div>
            </body>
            </html>
            """)
    fileName = gr.Textbox(visible=False)
    prompt = gr.Textbox(value = '{"house": null, "living room": null, "kitchen": null, "dining room": null, "bathroom": null, "bedroom": [null, null, null, null]}',visible=False)
    url = gr.Textbox(visible=False)
    gr.Markdown("""
        <html>
        <head>
        </head>
        <body>
        <div style="width: 100%; text-align: center; color: black; font-size: 42px; font-weight: 800; line-height: 59.09px; word-wrap: break-word">Design your Dream House</div>
        </body>
        </html>
        """)
    with gr.Tabs(elem_classes=['select', 'slient']):
        with gr.TabItem("Exterior"):
            with gr.Row():
                options = gr.Dropdown(choices = ["Asian", "Gothic", "Modern", "Neoclassical", "Nordic", "Other, please specify"], value = "Asian", multiselect=False, label="Choose your House style")
                txt     = gr.Textbox(label="House Style", visible=False)
            #btn1 = gr.Button(value="View Some example!")
            with gr.Column(elem_id = "image_background"):
                gr.Markdown("""
                                <html>
                                <head>
                                </head>
                                <body>
                                <div style="width: 100%; color: black; font-size: 10px; font-family: Roboto Flex; font-weight: 800; line-height: 10px; word-wrap: break-word">Design your Dream House</div>
                                </body>
                                </html>
                                """)
                with gr.Row():
                    
                    img1    =  gr.Image(value = Image.open(os.getcwd() + '/HouseStyle/Asain/Asain1.png'), show_download_button = False, label = "Preview 1")
                    img2    =  gr.Image(value = Image.open(os.getcwd() + '/HouseStyle/Asain/Asain2.png'), show_download_button = False, label = "Preview 2")
                    img3    =  gr.Image(value = Image.open(os.getcwd() + '/HouseStyle/Asain/Asain3.png'), show_download_button = False, label = "Preview 3")
                    img4    =  gr.Image(value = Image.open(os.getcwd() + '/HouseStyle/Asain/Asain4.png'), show_download_button = False, label = "Preview 4")
                    img5    =  gr.Image(value = Image.open(os.getcwd() + '/HouseStyle/Asain/Asain5.png'), show_download_button = False, label = "Preview 5")
                gr.Markdown("""
                            <html>
                            <head>
                            </head>
                            <body>
                            <div style="width: 100%; text-align: center; color: black; font-size: 10px; font-weight: 800; line-height: 10px; word-wrap: break-word">Design your Dream House</div>
                            </body>
                            </html>
                            """)
            with gr.Column(elem_classes=['select2', 'slient2']):
                with gr.Row():
                    txt1 = gr.Textbox(label="Enter Your Description", lines=3)
                    speech_audio = gr.Audio(source="microphone")

                with gr.Row():
                    gr.Markdown("                                          ")
                    btn2 = gr.Button(value="Clear", elem_id = "button_clear")
                    btn3 = gr.Button(value="Enhance Your Sentence?", elem_id = "button_enhance")
                    btn4 = gr.Button(value="Submit", elem_id = "button_submit")
                # video_1 = gr.Video()
            img    =  gr.Image(height = 512, width = 1536, show_download_button = False)
            #btn1.click(viewExample, inputs = [options], outputs = [img1, img2, img3, img4, img5])
            btn2.click(clear, inputs=[], outputs=[txt1])
            btn3.click(enhance_your_sentence, inputs = [options, txt, txt1], outputs = [txt1])
            btn4.click(generateImage, inputs=[prompt, txt1, options], outputs=[img, fileName, prompt])
            options.change(change_options, inputs = [options], outputs = [txt])
            options.change(viewExample, inputs = [options], outputs = [img1, img2, img3, img4, img5])
            speech_audio.change(transcribe, inputs = [speech_audio], outputs = [txt1])

        with gr.TabItem("Interior") as tab2:
            room_options = gr.Dropdown([ "dinning room", "kitchen", "living room", "bathroom", "bedroom"],  value = "dinning room", multiselect=False, label="Choose your design")
            with gr.Column(elem_classes=['select2', 'slient2']):
                with gr.Row():
                    I_txt = gr.Textbox(label="Enter Your Description", line = 3)
                    I_audio = gr.Audio(source="microphone")
                with gr.Row():
                    gr.Markdown("                                          ")

                    btn1 = gr.Button(value="Clear", elem_id = "button_clear")
                    btn2 = gr.Button(value="Enhance Your Sentence?", elem_id = "button_enhance")
                    btn3 = gr.Button(variant="primary", value="Submit", elem_id = "button_submit")
            

            with gr.Row():
                I_im_1 = gr.Image(height = 512, width = 1536, show_download_button = False)
                I_im_2 = gr.Image(show_download_button = False, visible = False)
                I_im_3 = gr.Image(show_download_button = False, visible = False)


            I_audio.change(transcribe, inputs = [I_audio], outputs = [I_txt])
            btn1.click(clear, inputs=[], outputs=[I_txt])
            btn2.click(enhance_your_sentence2, inputs = [room_options, I_txt], outputs = [I_txt])
            btn3.click(generate_room_inside, inputs=[room_options, I_txt, fileName, prompt], outputs=[I_im_1, I_im_2, I_im_3, prompt])
            room_options.change(viewCurrentImage, inputs = [room_options, fileName], outputs = [I_im_1, I_im_2, I_im_3])
            tab2.select(viewCurrentImage, inputs = [room_options, fileName], outputs = [I_im_1, I_im_2, I_im_3])

        with gr.TabItem("Audio") as tab3:
            gr.Markdown("""
                        <div>
                            <p style="font-size: 17px; color: black;"><b>Select your designs!</b></p>
                        </div>
                        """)
            with gr.Row():
                cb_house   = gr.Checkbox(label = "Chosen", value = 'True')
                cb_dinning = gr.Checkbox(label = "Chosen", value = 'True')
                cb_kitchen = gr.Checkbox(label = "Chosen", value = 'True')
                cb_living  = gr.Checkbox(label = "Chosen", value = 'True')
            
            with gr.Row():
                img_house   = gr.Image(show_download_button = False)
                img_dinning = gr.Image(show_download_button = False)
                img_kitchen = gr.Image(show_download_button = False)
                img_living   = gr.Image(show_download_button = False)
            
            with gr.Row():
                
                cb_bath     = gr.Checkbox(label = "Chosen", value = 'True')
                cb_bedroom1 = gr.Checkbox(label = "Chosen", value = 'True')
                cb_bedroom2 = gr.Checkbox(label = "Chosen", value = 'True')
                cb_bedroom3 = gr.Checkbox(label = "Chosen", value = 'True')
            
            with gr.Row():
                
                img_bath     = gr.Image(show_download_button = False)
                img_bedroom1 = gr.Image(show_download_button = False)
                img_bedroom2 = gr.Image(show_download_button = False)
                img_bedroom3 = gr.Image(show_download_button = False)
                
                        
            gr.Markdown("""
                        <div>
                            <p style="font-size: 17px; color: black;"><b>Design Your own Music!</b></p>
                        </div>
                        """)
            with gr.Row():
                txt   = gr.Textbox(label="Enter Your Description")
            with gr.Column(elem_classes=['select2', 'slient2']):
                with gr.Row():
                    audio = gr.Audio(source="microphone", container = False, elem_id = "audio_id", show_download_button = False)
                    btn1 = gr.Button(value="Clear", elem_id = "audio_button_clear")
                    btn2 = gr.Button(value="Enhance Your Sentence?", elem_id = "audio_button_enhance")
                    btn3 = gr.Button(value="Submit", elem_id = "audio_button_submit")
            
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
                                <div style="width: 100%; color: white; font-size: 10px; font-family: Roboto Flex; font-weight: 800; line-height: 10px; word-wrap: break-word">Design your Dream House</div>
                                </body>
                                </html>
                                """)
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
                                <div style="width: 100%; color: white; font-size: 10px; font-family: Roboto Flex; font-weight: 800; line-height: 10px; word-wrap: break-word">Design your Dream House</div>
                                </body>
                                </html>
                                """)
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
                                <div style="width: 100%; color: white; font-size: 10px; font-family: Roboto Flex; font-weight: 800; line-height: 10px; word-wrap: break-word">Design your Dream House</div>
                                </body>
                                </html>
                                """)

            btn1.click(clear, inputs=[], outputs=[txt])
            btn2.click(enhance_your_sentence3, inputs = [txt], outputs = [txt])
            btn3.click(generate_audio, inputs=[txt, fileName], outputs=[audio])
            tab3.select(viewAllimages, inputs = [fileName], outputs = [img_house, img_dinning, img_kitchen, img_living, img_bath, img_bedroom1, img_bedroom2, img_bedroom3])

            
        
        with gr.TabItem("Generate"):
            gr.Markdown("""
                        <div>
                            <p style="font-size: 17px; color: black;"><b>Generate your house introduction</b></p>
                        </div>
                        """)
            
            intro_txt   = gr.Textbox(label="Enter Your Description")
            intro_drop  = gr.Dropdown(["EN_Speaker(Male)", "EN_Speaker(Female)", "CN_Speaker(Male)", "CN_Speaker(Female)"], value = "EN_Speaker(Male)", multiselect=False, label="Share")
            with gr.Column(elem_classes=['select2', 'slient2']):
                with gr.Row():
                    intro_audio = gr.Audio(source="microphone", container = False, elem_id = "audio_id", show_download_button = False)
                    intro_clear = gr.Button(value="Clear", elem_id = "audio_button_clear")
                    intro_btn = gr.Button(value="Generate", elem_id = "audio_button_submit")
            intro_btn.click(generateDescription, inputs = [cb_house, cb_dinning, cb_kitchen, cb_living, cb_bath, cb_bedroom1, cb_bedroom2, cb_bedroom3, prompt, fileName, intro_drop], outputs = [intro_txt, intro_audio])
            intro_clear.click(clear, inputs = [], outputs = [intro_txt])
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
                                <div style="width: 100%; color: white; font-size: 10px; font-family: Roboto Flex; font-weight: 800; line-height: 10px; word-wrap: break-word">Design your Dream House</div>
                                </body>
                                </html>
                                """)
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
                                <div style="width: 100%; color: white; font-size: 10px; font-family: Roboto Flex; font-weight: 800; line-height: 10px; word-wrap: break-word">Design your Dream House</div>
                                </body>
                                </html>
                                """)
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
                                <div style="width: 100%; color: white; font-size: 10px; font-family: Roboto Flex; font-weight: 800; line-height: 10px; word-wrap: break-word">Design your Dream House</div>
                                </body>
                                </html>
                                """)

            gr.Markdown("""
                        <div>
                            <p style="font-size: 17px; color: black;"><b>Select what audio to include in your video</b></p>
                        </div>
                        """)
            
            
            with gr.Row():
                cb_audio  = gr.Checkbox(value = 'True', label = "audio ", container = False, elem_id = "checkbox_id")
                cb_effect = gr.Checkbox(value = 'True', label = "effect", container = False, elem_id = "checkbox_id")
                cb_descri = gr.Checkbox(value = 'True', label = "description",container = False, elem_id = "checkbox_id")
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
                                <div style="width: 100%; color: white; font-size: 10px; font-family: Roboto Flex; font-weight: 800; line-height: 10px; word-wrap: break-word">Design your Dream House</div>
                                </body>
                                </html>
                                """)
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
                                <div style="width: 100%; color: white; font-size: 10px; font-family: Roboto Flex; font-weight: 800; line-height: 10px; word-wrap: break-word">Design your Dream House</div>
                                </body>
                                </html>
                                """)
            with gr.Column(elem_classes=['select2', 'slient2']):
                with gr.Row():
                    gr.Markdown("                                          ")
                    btn_Share = gr.Button(value="Share", elem_id = "button_clear")
                    btn_QR = gr.Button(value="Generate QR code", elem_id = "button_enhance")
                    btn_video = gr.Button(value="Generate Your Video!", elem_id = "button_submit")

            bool_test = gr.Textbox(value = "00", visible = False)
            with gr.Column(elem_classes=['select2', 'slient2']):
                final_video = gr.Video(height = 512, width = 1536)
            with gr.Column(variant = 'compact', elem_id = "poppage", visible = False) as QRcodePage:
                close_button_QR = gr.Button(value = 'X', elem_id = "button_close")
                QR_img  = gr.Image(height = 256, width = 1536, container = False, show_download_button = False)
            with gr.Column(variant = 'compact', elem_id = "poppage", visible = False) as SharePage:
                close_button_Share = gr.Button(value = 'X', elem_id = "button_close")
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

                        <p style="font-size: 20px; text-align: center; color: #FF7712;">Share your work!</p>
                        </body>
                        </html>
                        """)
                with gr.Row():
                    gr.Markdown("            ")
                    facebook = gr.HTML("""
                                <html>
                                <head>
                                <style>
                                .center {
                                    display: block;
                                    margin-left: auto;
                                    margin-right: auto;
                                }
                                </style>
                                </head>
                                <body>
                                <img src="https://cdn3.iconfinder.com/data/icons/picons-social/57/46-facebook-1024.png" class="center" style= "width: 35px; height: 35px">
            
                                </body>
                                </html>
                                """)
                    gr.Markdown("            ")
                    reddit = gr.HTML("""
                                <html>
                                <head>
                                <style>
                                .center {
                                display: block;
                                margin-left: auto;
                                margin-right: auto;
                                }
                                </style>
                                </head>
                                <body>
                                <img src="https://static-00.iconduck.com/assets.00/reddit-logo-icon-2048x2048-vtzhwa71.png" class="center" style= "width: 35px; height: 35px">
                                </body>
                                </html>
                                """)
                    gr.Markdown("            ")
                    weibo = gr.HTML("""
                                <html>
                                <head>
                                <style>
                                .center {
                                display: block;
                                margin-left: auto;
                                margin-right: auto;
                                }
                                </style>
                                </head>
                                <body>
                                <img src="https://cdn.freebiesupply.com/images/large/2x/weibo-logo-black-transparent.png" class="center" style= "width: 35px; height: 35px">
                                </body>
                                </html>
                                """)
                    gr.Markdown("            ")
                    twitter = gr.HTML("""
                                <html>
                                <head>
                                <style>
                                .center {
                                display: block;
                                margin-left: auto;
                                margin-right: auto;
                                }
                                </style>
                                </head>
                                <body>
                                <img src="https://cdn3.iconfinder.com/data/icons/picons-social/57/43-twitter-1024.png" class="center" style= "width: 35px; height: 35px">
                                </body>
                                </html>
                                """)
                    gr.Markdown("            ")
       
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

                        <p style="font-size: 15px; color: black;">&nbsp;&nbsp;Or copy the link</p>
                        </body>
                        </html>
                        """)
                copy_page = gr.Textbox(value = "", label = "", show_copy_button = True)

    
            close_button_QR.click(hideColumns, inputs = [], outputs = [QRcodePage, SharePage, bool_test])
            close_button_Share.click(hideColumns, inputs = [], outputs = [QRcodePage, SharePage, bool_test])
            btn_video.click(generateVideo, inputs = [cb_house, cb_dinning, cb_kitchen, cb_living, cb_bath, cb_bedroom1, cb_bedroom2, cb_bedroom3, cb_audio, cb_effect, cb_descri, fileName], outputs = [final_video])
            btn_Share.click(showSharePage, inputs = [bool_test], outputs = [SharePage, QRcodePage, bool_test])
            btn_QR.click(showQRcodePage, inputs = [bool_test], outputs = [QRcodePage, SharePage, bool_test])
            final_video.change(generateQRcode, inputs = [cb_audio, cb_effect, cb_descri, fileName], outputs = [QR_img, facebook, reddit, weibo, twitter, copy_page])
    

    
if __name__ == "__main__":
    demo.queue().launch(share=True, debug=True)