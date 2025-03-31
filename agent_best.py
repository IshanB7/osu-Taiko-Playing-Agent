# import tensorflow as tf
import pydirectinput  
import numpy as np
# from tensorflow.keras.models import load_model
import mss
import time
from pynput import keyboard
pydirectinput.PAUSE = 0.1 

# model = load_model('osu_agent.keras')
ACTIONS = ['', 'k', 'j']
region = {'top': 505, 'left': 550, 'width': 20, 'height': 10}
# upper region for the large circle
region2 = {'top': 460, 'left': 550, 'width': 10, 'height': 10}

last_value = 1

def classify_color(avg_color):
    """Classify the average color as red, blue, or yellow."""
    global last_value
    b, g, r = avg_color

    if r > 150 and g < 90 and b < 90:  #  red
        return 2
    # elif r > 180 and g > 100 and b < 90:  #  yellow
    # # return 1 
    #     last_value = 1 if last_value == 2 else 2
    #     return last_value
    elif b > 120 and r < 90 and g > 120:  #  blue green
        return 1
    else:
        return 0
    

while True:
    with mss.mss() as sct:
        img = sct.grab(region)
        img = np.array(img)

        img2 = sct.grab(region2)
        img2 = np.array(img2)
    

    avg_color = img2[:, :, :3].mean(axis=(0, 1))
    # check upper region first 
    if avg_color[0] < 100 and avg_color[1] <100  and avg_color[2] < 100:
        avg_color = img[:, :, :3].mean(axis=(0, 1))

    # print(f"Average color: {avg_color}")
    press = classify_color(avg_color)
    # print(f"Press: {press}")
    if ACTIONS[press]:  
        pydirectinput.press(ACTIONS[press])
        # keyboard.KeyCode.from_char(ACTIONS[press])