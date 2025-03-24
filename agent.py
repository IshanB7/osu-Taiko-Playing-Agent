import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
import time
# from tensorflow.keras.preprocessing.image import load_img, img_to_array

import mss
import cv2
import pyautogui

model = load_model('osu_agent.keras')
ACTIONS = ['', 'k', 'j']
region = {'top': 252, 'left': 207, 'width': 235, 'height': 235}

while True:
    with mss.mss() as sct:
        img = sct.grab(region)
        img = np.array(img)

    image = cv2.resize(img[:, :, :3], (16, 16))
    input_img = np.expand_dims(image / 255.0, axis=0)
    press = np.argmax(model(input_img)[0])
    pyautogui.press(ACTIONS[press])