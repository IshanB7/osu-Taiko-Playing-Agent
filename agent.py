import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model

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

    img_array = image / 255.0
    img_array = np.round(img_array)

    input_img = np.expand_dims(img_array, axis=0)
    press = np.argmax(model(input_img)[0])
    pyautogui.press(ACTIONS[press])