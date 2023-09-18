from OutsideDesign import *
import cv2
import math
from PIL import Image, ImageEnhance
import copy
import random

def InsideDesign(img1, img2, img3, img4, img5, img6):
    corr = cv2.imread(img1)
    dinn = cv2.imread(img2)
    kitc = cv2.imread(img3)
    eati = cv2.imread(img4)
    bedroom =  cv2.imread(img5)
    bathroom = cv2.imread(img6)
    dinn = cv2.resize(dinn, (1800, 1024), interpolation=cv2.INTER_AREA)
    height, width, _ = corr.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    video = cv2.VideoWriter('Insideoutput.mp4', fourcc, 100.0, (height, width))
    frame = cv2.imread('frame.png')
    cv2.imwrite('frame.png', frame)
    
    video.write(frame)
    corr_copy = copy.deepcopy(corr)
    corr_copy = control_bright(corr, 0.1)
    max_radius = 20  
    
    
    for i in range(width//2):
        frame = open_door(frame, corr_copy, i)
        
        if i%2 == 0:
            video.write(frame)
    for i in range(200):
        bright_frame = control_bright(corr, 0.1 + abs(0.1 * math.sin(i * 0.01)))
        frame = create_blur_frame(i, max_radius, bright_frame)
        video.write(frame)
        
    bright_frame = control_bright(corr, 1.1)
    frame = create_blur_frame(i, max_radius, bright_frame)

    for _ in range(100):  
        video.write(bright_frame)
    

    corr_edge = cv2.cvtColor(cv2.Canny(corr,100,200), cv2.COLOR_GRAY2BGR)
    dinn_edge = cv2.cvtColor(cv2.Canny(dinn,100,200), cv2.COLOR_GRAY2BGR)
    for i in range(height):
        frame = up_down_swap(corr, corr_edge, i, True)
        if i%32 == 0:
            video.write(frame)
    
    for i in range(height):
        frame = up_down_swap(corr_edge, dinn_edge, i, False)
        if i%32 == 0:
            video.write(frame)
    
    for i in range(height):
        frame = up_down_swap(dinn_edge[0 : height, 0 : width], dinn[0 : height, 0 : width], i, True)
        if i%32 == 0:
            video.write(frame)
    
    


    for i in range(0, dinn.shape[1] - width, 2):
        frame = dinn[:height, i:i+width]
        video.write(frame)
    
    

    for i in range(50):
        alpha_channel = np.full((height, width, 1), 255, dtype=np.uint8)
        if i%5 == 0:
            image_4_channels = np.concatenate((frame, alpha_channel), axis=2)
            m_height = height // 2
            m_width  = width  // 2
            rate = 2 ** int(i / 5) 
            image_4_channels[m_height - rate : m_height + rate - 1, m_width - rate : m_width + rate - 1] = waterFall(kitc[m_height - rate : m_height + rate - 1, m_width - rate : m_width + rate - 1])
        video.write(image_4_channels[:, :, :3])
    
    frame = kitc
    for i in range(50):
        alpha_channel = np.full((height, width, 1), 255, dtype=np.uint8)
        if i%5 == 0:
            image_4_channels = np.concatenate((frame, alpha_channel), axis=2)
            m_height = height // 2
            m_width  = width  // 2
            rate = 2 ** int(9 - (i / 5))
            image_4_channels[m_height - rate : m_height + rate - 1, m_width - rate : m_width + rate - 1] = waterFall(image_4_channels[m_height - rate : m_height + rate - 1, m_width - rate : m_width + rate - 1])
        video.write(image_4_channels[:, :, :3])
    
    kitch_frame = smokeToKitchen(frame)
    for i in range(96 * 4):
        if i%4 == 0:
            next_frame = kitch_frame[int(i/4)][0]
        video.write(next_frame)

    frame = eati
    
    for i in range(96 * 4):
        if i%4 == 0:
            roi = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            smoke_frame = kitch_frame[len(kitch_frame) - 1 - int(i/4)][1]

            roi.putalpha(250)
            img3 = Image.alpha_composite(roi, smoke_frame)
            next_frame = cv2.cvtColor(np.array(img3), cv2.COLOR_RGBA2BGRA)
        video.write(next_frame[:, :, :3])

    
    next_frame = randomReducePixel(frame, bedroom)
    for i in range(210):
        if i%2 == 0:
            next_frame = randomReducePixel(next_frame, bedroom)
        video.write(next_frame)
    
    ctb = cloudToBedroom(bedroom)
    for i in range(len(ctb) * 4):
        if i%4 == 0:
            next_frame = ctb[int(i/4)][0]
        video.write(next_frame)
    
    black_canvas = np.zeros((1024, 1024, 3), dtype=np.uint8)
    for i in range(200):
        video.write(black_canvas)

    frame = toilet_light(bathroom)
    for i in range(300):
        video.write(frame)

    video.release()


def square_change(img1, img2, i):
    output = copy.deepcopy(img1)
    output[0 : i ** 2, 0 : i **2] = img2[0 : i ** 2, 0 : i **2]
    return output

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


def up_down_swap(img1, img2, i, up):
    copy_img = copy.deepcopy(img1)
    height, width, _ = img1.shape
    if up:
        copy_img[0 : i, 0 : width] =  img2[0 : i, 0 : width]
    else:
        copy_img[height - 1 - i: height - 1, 0 : width] = img2[height - 1 - i: height - 1 , 0 : width]
    return copy_img
    
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

def smokeToKitchen(background):
    # Open the video file
    video_path = 'smokeVideo.mp4'  # Replace with your video file path
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        exit()

    frame_count = 0
    frame_list = []
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        smoke = cv2.resize(frame, (1024, 1024), interpolation=cv2.INTER_AREA)

        roi = Image.fromarray(cv2.cvtColor(background, cv2.COLOR_BGR2RGB))
        img2 = Image.fromarray(cv2.cvtColor(smoke, cv2.COLOR_BGR2RGB))
        height, width = 1024, 1024
        left_bottom_x = width - width // 7
        left_bottom_y = height - height // 7

        black_rectangle = Image.new('RGB', (width, height), (0, 0, 0))

        img2.paste(black_rectangle, (left_bottom_x, left_bottom_y))
        
        roi.putalpha(250)
        img2.putalpha(frame_count * 2)

        img3 = Image.alpha_composite(roi, img2)

        img3 = cv2.cvtColor(np.array(img3), cv2.COLOR_RGBA2BGRA)
        
        frame_count += 1
        frame_list.append((img3[:, :, :3], img2))
    return frame_list

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

def cloudToBedroom(background):
    # Open the video file
    video_path = 'cloudpng.mp4'  # Replace with your video file path
    video_path2 = 'thunder.mp4'
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        exit()

    frame_count = 0
    frame_list = []
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        cloud = cv2.resize(frame, (1024, 1024), interpolation=cv2.INTER_AREA)
        height, width, _ = cloud.shape

    
        new_height = int(height * 6 / 7)

        cropped_image = cloud[:new_height, :]
        cropped_image = cv2.resize(cropped_image, (1024, 1024), interpolation=cv2.INTER_AREA)
        roi = Image.fromarray(cv2.cvtColor(background, cv2.COLOR_BGR2RGB))
        img2 = Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
        
        roi.putalpha(250)
        img2.putalpha(int(frame_count/1.7))
        img3 = Image.alpha_composite(roi, img2)

        img3 = cv2.cvtColor(np.array(img3), cv2.COLOR_RGBA2BGRA)
        
        frame_count += 1
        frame_list.append((img3[:, :, :3], img2))
    
    cap = cv2.VideoCapture(video_path2)
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        #cloud = cv2.resize(frame, (1024, 1024), interpolation=cv2.INTER_AREA)
        height, width, _ = frame.shape

    
        new_width = int(width * 6 / 7)

        cropped_image = frame[:, :new_width]
        cropped_image = cv2.resize(cropped_image, (1024, 1024), interpolation=cv2.INTER_AREA)
        roi = Image.fromarray(cv2.cvtColor(background, cv2.COLOR_BGR2RGB))
        img2 = Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
        
        roi.putalpha(250)
        img2.putalpha(int(frame_count/1.7))
        img3 = Image.alpha_composite(roi, img2)

        img3 = cv2.cvtColor(np.array(img3), cv2.COLOR_RGBA2BGRA)
        
        frame_count += 1
        frame_list.append((img3[:, :, :3], img2))
    return frame_list

def toilet_light(img):
    rows, cols = img.shape[:2]

    centerX = rows / 4
    centerY = cols / 2
    radius = min(centerX, centerY)

    strength = 200

    dst = np.zeros((rows, cols, 3), dtype="uint8")

    for i in range(rows):
        for j in range(cols):
        
            distance = math.pow((centerY-j), 2) + math.pow((centerX-i), 2)
            B =  img[i,j][0]
            G =  img[i,j][1]
            R = img[i,j][2]
            if (distance < radius * radius):
            
                result = (int)(strength*( 1.0 - math.sqrt(distance) / radius ))
                B = img[i,j][0] + result
                G = img[i,j][1] + result
                R = img[i,j][2] + result
                B = min(255, max(0, B))
                G = min(255, max(0, G))
                R = min(255, max(0, R))
                dst[i,j] = np.uint8((B, G, R))
            else:
                dst[i,j] = np.uint8((B, G, R))
    return dst

#InsideDesign("corr.png", "dinn.png", "kitchen.png", "eat.png", "bedroom.png", "bathroom.png")
    