import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from pynput import keyboard

import mss
import cv2
import pyautogui
import time

# model = load_model('osu_agent.keras')
model = load_model('best_agent.keras')

ACTIONS = ['', 'k', 'j']
region = {'top': 252, 'left': 207, 'width': 235, 'height': 235}

interval = 1 / 30.
next_frame_time = time.time()

capture = False
def on_press(key):
    pass

def on_release(key):
    global capture
    if key == keyboard.Key.enter:
        capture = not capture

listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

while not capture:
    time.sleep(0.1)

while capture:
    sleep_time = next_frame_time - time.time()
    if sleep_time > 0:
        time.sleep(sleep_time)
    next_frame_time += interval

    with mss.mss() as sct:
        img = sct.grab(region)
        img = np.array(img)

    image = cv2.resize(img[:, :, :3], (16, 16))

    img_array = image / 255.0
    input_img = np.expand_dims(img_array, axis=0)
    press = np.argmax(model(input_img)[0])
    pyautogui.press(ACTIONS[press], _pause=False)
    # pyautogui.press(ACTIONS[press])