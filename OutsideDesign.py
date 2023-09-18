import cv2
import math
import numpy as np
from PIL import Image, ImageEnhance

# Load the panorama image
def image2videoOutside(image, image2):
    panorama = cv2.imread(image)

    house = cv2.imread(image2)
    panorama = resize_image(panorama, house)
    # Get the dimensions of the image
    height, width, _ = panorama.shape
    h_height, h_width, _ = house.shape
    video_width = h_width

    # Create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    video = cv2.VideoWriter('output.mp4', fourcc, 100.0, (video_width, height))

    # Loop over the width of the image
    upper = width-video_width
    render_point = upper - 100

    for i in range(0, upper, 2):
        frame = panorama[:height, i:i+video_width]
        frame = change_bright(frame, i, upper, True)
        if i > render_point - 1:
            frame = change_blur(frame, i - render_point, upper - render_point, False)
            frame = zoomIn(frame,  i - render_point, True)
        # Write the frame to the video
        video.write(frame)

    for i in range(0, 400):
        frame = house
        frame = change_bright(frame, i, 400, False)
        if i < 100:
            frame = change_blur(frame, i, 100, True)
            frame = zoomIn(frame,  i, False)
        if i > 200:
            frame = zoomInDoor(frame, i - 200)
        video.write(frame)
    
    '''
    for i in range(frame.shape[1] // 2):
        frame = open_door(frame, i)
        if i%4 == 0:
            video.write(frame)
    '''
    
    # Release the VideoWriter
    video.release()
    return video, frame


def change_bright(frame, i, upper, dark):
    frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    enhancer = ImageEnhance.Brightness(frame)
    if dark:
        frame = enhancer.enhance(1.3 - i/upper)
    else:
        frame = enhancer.enhance(0.4 + i/upper)
    frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)

    return frame

def change_blur(frame, i, upper, clear):
    if clear:
        return cv2.GaussianBlur(frame, (15, 15), 10 * (1 - i/upper))
    else:
        return cv2.GaussianBlur(frame, (15, 15), 10 * i/upper)

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

def zoomInDoor(frame, i):
    frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    width, height = frame.size
    scale_factor = 1.0 + (i / 200)
    zoomed_img = frame.resize((int(width * scale_factor), int(height * scale_factor)), Image.ANTIALIAS)
    
    top = zoomed_img.height - height
    bottom = zoomed_img.height
    left = (zoomed_img.width - width) / 2
    right = (zoomed_img.width + width) / 2
    zoomed_img = zoomed_img.crop((left, top, right, bottom))
    return cv2.cvtColor(np.array(zoomed_img), cv2.COLOR_RGB2BGR)

def open_door(frame, image, i):
    height, width, _ = frame.shape

    w_mid = width // 2

    for x in range(height):
        if w_mid - i > 0:
            frame[x, w_mid - i] = image[x, w_mid - i]
        if w_mid + i < width:
            frame[x, w_mid + i] = image[x, w_mid + i]
    
    return frame

def resize_image(img1, img2):
    img1 = cv2.resize(img1, (1800, 1024), interpolation=cv2.INTER_AREA)
    rows,cols,channels = img2.shape
    img2 = cv2.resize(img2, (rows//8, cols//8), interpolation=cv2.INTER_AREA)


    rows,cols,channels = img2.shape
    rows2, cols2, _ = img1.shape

    roi  = img1[rows2 * 2//5 - rows//2 : rows2 * 2//5 + rows//2, cols2  - cols * 4 : cols2 - cols * 3]

    roi = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
    img2 = Image.fromarray(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    roi.putalpha(250)
    img2.putalpha(75)

    img3 = Image.alpha_composite(roi, img2)

    img1 = Image.fromarray(cv2.cvtColor(img1, cv2.COLOR_BGRA2RGBA))
    img1.paste(img3, (cols2 - cols * 4, rows2 * 2 // 5 - rows // 2))

    # Convert img1 back to a NumPy array
    img1 = cv2.cvtColor(np.array(img1), cv2.COLOR_RGBA2BGRA)

    return img1

#image2videoOutside("image.png", "image2.png")
    