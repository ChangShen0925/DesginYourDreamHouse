import cv2
import copy
import numpy as np
from PIL import Image, ImageEnhance
import math
import os 

def AppearByPixels(img1, img2, video):
    height, width, _ = img1.shape
    next_frame = randomReducePixel(img1, img2)
    for i in range(1, int(height * width / 10 / 1024) * 2 - 1):
        if i%2 == 0:
            next_frame = randomReducePixel(next_frame, img2)
        video.write(next_frame)
    return video


def randomReducePixel(image1, image2):
    copy_image = copy.deepcopy(image1)
    mask = (image1 != image2).any(axis=2)  

    indices = np.argwhere(mask)

    num_pixels_to_select = 10 * 1024 
    selected_pixels_indices = np.random.choice(len(indices), num_pixels_to_select)
    selected_pixels_coordinates = indices[selected_pixels_indices]
    for i in selected_pixels_coordinates:
        copy_image[i[0], i[1]] = image2[i[0], i[1]]
    return copy_image

def X_Ray(img1, img2, video):
    height, width, _ = img1.shape
    img1_edge = cv2.cvtColor(cv2.Canny(img1,100,200), cv2.COLOR_GRAY2BGR)
    img2_edge = cv2.cvtColor(cv2.Canny(img2,100,200), cv2.COLOR_GRAY2BGR)
    for i in range(height):
        frame = up_down_swap(img1, img1_edge, i, True)
        if i%32 == 0:
            video.write(frame)
    
    for i in range(height):
        frame = up_down_swap(img1_edge, img2_edge, i, False)
        if i%32 == 0:
            video.write(frame)
    
    for i in range(height):
        frame = up_down_swap(img2_edge, img2, i, True)
        if i%32 == 0:
            video.write(frame)
    return video

def up_down_swap(img1, img2, i, up):
    copy_img = copy.deepcopy(img1)
    height, width, _ = img1.shape
    if up:
        copy_img[0 : i, 0 : width] =  img2[0 : i, 0 : width]
    else:
        copy_img[height - 1 - i: height - 1, 0 : width] = img2[height - 1 - i: height - 1 , 0 : width]
    return copy_img


def swap_waterfall(img1, img2, video):
    height, width, _ = img1.shape
    for i in range(50):
        alpha_channel = np.full((height, width, 1), 255, dtype=np.uint8)
        if i%5 == 0:
            image_4_channels = np.concatenate((img1, alpha_channel), axis=2)
            m_height = height // 2
            m_width  = width  // 2
            rate = 2 ** int(i / 5) 
            image_4_channels[m_height - rate : m_height + rate - 1, m_width - rate : m_width + rate - 1] = waterFall(img2[m_height - rate : m_height + rate - 1, m_width - rate : m_width + rate - 1])
        video.write(image_4_channels[:, :, :3])
    
    frame = img2
    for i in range(50):
        alpha_channel = np.full((height, width, 1), 255, dtype=np.uint8)
        if i%5 == 0:
            image_4_channels = np.concatenate((frame, alpha_channel), axis=2)
            m_height = height // 2
            m_width  = width  // 2
            rate = 2 ** int(9 - (i / 5))
            image_4_channels[m_height - rate : m_height + rate - 1, m_width - rate : m_width + rate - 1] = waterFall(image_4_channels[m_height - rate : m_height + rate - 1, m_width - rate : m_width + rate - 1])
        video.write(image_4_channels[:, :, :3])
    return video

def waterFall(img):
    rows, cols = img.shape[:2]

    dst = np.zeros((rows, cols, 4), dtype="uint8")

    wavelength = 60
    amplitude = 30
    phase = math.pi / 4

    centreX = 0.5
    centreY = 0.5
    radius = min(rows, cols) / 2


    icentreX = cols*centreX
    icentreY = rows*centreY
        
    for i in range(rows):
        for j in range(cols):
            dx = j - icentreX
            dy = i - icentreY
            distance = dx*dx + dy*dy
            
            if distance>radius*radius:
                x = j
                y = i
            else:
        
                distance = math.sqrt(distance)
                amount = amplitude * math.sin(distance / wavelength * 2*math.pi - phase)
                amount = amount *  (radius-distance) / radius
                amount = amount * wavelength / (distance+0.0001)
                x = j + dx * amount
                y = i + dy * amount

        
            if x<0:
                x = 0
            if x>=cols-1:
                x = cols - 2
            if y<0:
                y = 0
            if y>=rows-1:
                y = rows - 2

            p = x - int(x)
            q = y - int(y)
            

            dst[i, j, :3] = (1 - p) * (1 - q) * img[int(y), int(x), :3] + p * (1 - q) * img[int(y), int(x), :3] + (1 - p) * q * img[int(y), int(x), :3] + p * q * img[int(y), int(x), :3]

            dst[i, j, 3] = 1
    
    return dst

def zoom_swap(img1, img2, video):
    for i in range(100):
        frame = zoomIn(img1,  i, True)
        video.write(frame)
    for i in range(100):
        frame = zoomIn(img2,  i, False)
        video.write(frame)
    return video

def zoomIn(frame, i, big):
    frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    width, height = frame.size
    if big:
        scale_factor = 1.0 + (i / 100)
    else:
        scale_factor = 2.0 -  (i / 100)
    zoomed_img = frame.resize((int(width * scale_factor), int(height * scale_factor)), Image.ANTIALIAS)
    
    left = (zoomed_img.width - width) / 2
    top = (zoomed_img.height - height) / 2
    right = (zoomed_img.width + width) / 2
    bottom = (zoomed_img.height + height) / 2
    zoomed_img = zoomed_img.crop((left, top, right, bottom))
    return cv2.cvtColor(np.array(zoomed_img), cv2.COLOR_RGB2BGR)

def bur_bright_swap(img1, img2, video):
    max_radius = 20  
    for i in range(100):
        bright_frame = control_bright(img1, 1 - (i/100))
        frame = create_blur_frame(100 - i, max_radius, bright_frame)
        video.write(frame)

    for i in range(100):
        bright_frame = control_bright(img2, 0.1 + (i/100))
        frame = create_blur_frame(i, max_radius, bright_frame)
        video.write(frame)
    
    return video
    

def control_bright(frame, i):
    frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    enhancer = ImageEnhance.Brightness(frame)

    frame = enhancer.enhance(i)
    frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
    
    return frame

def apply_blur(image, radius):
    return cv2.GaussianBlur(image, (radius * 2 + 1, radius * 2 + 1), 0)

def create_blur_frame(frame_number, max_radius, input_image):
   
 
    radius = int(max_radius * abs(frame_number - 200 / 2) / (200 / 2))

    blurred_image = apply_blur(input_image, radius)

    return blurred_image

def open_door_swap(img1, img2, video):
    frame = img1
    height, width, _ = frame.shape
    for i in range(width//2):
        frame = open_door(frame, img2, i)
        
        if i%2 == 0:
            video.write(frame)
    return video

def open_door(frame, image, i):
    height, width, _ = frame.shape

    w_mid = width // 2

    for x in range(height):
        if w_mid - i > 0:
            frame[x, w_mid - i] = image[x, w_mid - i]
        if w_mid + i < width:
            frame[x, w_mid + i] = image[x, w_mid + i]
    
    return frame

def video_background(img1, img2, video, video_path):
    img_frame = video2room(img1, img2, video_path)
    for i in range(len(img_frame)):
        next_frame = img_frame[i]
        video.write(next_frame)

    return video

def video2room(background1, background2, video_path):
    # Open the video file
    
    cap = cv2.VideoCapture(video_path)
    im_height, im_width, _ = background1.shape
    if not cap.isOpened():
        print("Error: Could not open video file.")
        exit()

    frame_count = 0
    frame_list = []
    output_list = []
    while True:
        ret, frame = cap.read()

        if not ret:
            break
        frame_count += 1
        frame_list.append(frame)

    for i in range(len(frame_list)):
        frame = frame_list[i]
        if i < len(frame_list) / 2:
            background = background1
        else:
            background = background2

        cloud = cv2.resize(frame, (im_width, im_height), interpolation=cv2.INTER_AREA)
        height, width, _ = cloud.shape

    
        new_width = int(height * 9 / 10)

        cropped_image = cloud[:, :new_width]
        cropped_image = cv2.resize(cropped_image, (im_width, im_height), interpolation=cv2.INTER_AREA)
        roi = Image.fromarray(cv2.cvtColor(background, cv2.COLOR_BGR2RGB))
        img2 = Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
        
        roi.putalpha(250)
        if i < len(frame_list) / 2:
            img2.putalpha(int(i * 7))
        else:
            img2.putalpha(int((frame_count - i) * 7))
        img3 = Image.alpha_composite(roi, img2)

        img3 = cv2.cvtColor(np.array(img3), cv2.COLOR_RGBA2BGRA)
        
        output_list.append(img3[:, :, :3])
    return output_list



