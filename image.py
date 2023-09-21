import cv2
from PIL import Image
import numpy as np
a = cv2.imread('background.png')
frame = cv2.resize(a, (1024+256, 1024), interpolation=cv2.INTER_AREA)
Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).save('background.png')