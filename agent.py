import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model

import mss
import cv2
import pydirectinput

# pydirectinput.PAUSE = 0 
model = load_model('osu_agent.keras')
ACTIONS = ['', 'k', 'j']
region = {'top': 450, 'left': 340, 'width': 280, 'height': 280}  # Screenshot region

while True:
    with mss.mss() as sct:
        img = sct.grab(region)
        img = np.array(img)

    image = cv2.resize(img[:, :, :3], (16, 16))

    img_array = image / 255.0
    img_array = np.round(img_array)

    input_img = np.expand_dims(img_array, axis=0)
    press = np.argmax(model(input_img)[0])
    pydirectinput.press(ACTIONS[press])