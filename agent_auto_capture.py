# import tensorflow as tf
import pydirectinput  
import numpy as np
# from tensorflow.keras.models import load_model
import mss
import time
import os
import threading
from pynput import keyboard

pydirectinput.PAUSE = 0 

# model = load_model('osu_agent.keras')
region = {'top': 505, 'left': 550, 'width': 20, 'height': 10}
# upper region for the large circle
region2 = {'top': 460, 'left': 550, 'width': 10, 'height': 10}


capture_region = {'top': 450, 'left': 340, 'width': 280, 'height': 280}  # Screenshot region
ACTIONS = [False, 'k', 'j']
last_value = 1
capture = False 

os.makedirs('images', exist_ok=True)
counter = len(os.listdir('images'))  # Initialize counter based on existing files
counter_lock = threading.Lock()  

def capture_screenshot(label):
    global counter
    with mss.mss() as sct:
        img = sct.grab(capture_region)

        with counter_lock:
            file_path = f"images/image_{counter}.png"

            mss.tools.to_png(img.rgb, img.size, output=file_path)

            with open('labels.txt', 'a') as f:
                f.write(f"{label}\n")

            counter += 1

def on_key_press(key):
    global capture
    try:
        if key == keyboard.Key.esc:
            capture = False
            print("Capture stopped.")
        elif not capture:
            capture = True
            print("Capture started...")
    except Exception as e:
        print(f"Key error: {e}")
        
def classify_color(avg_color):
    """Classify the average color as red, blue, or yellow."""
    global last_value
    b, g, r = avg_color

    if r > 150 and g < 90 and b < 90:  #  red
        return 2
    
    # ignore yellow 
    # elif r > 180 and g > 100 and b < 90:  #  yellow
    # # return 1 
    #     last_value = 1 if last_value == 2 else 2
    #     return last_value
    elif b > 120 and r < 90 and g > 120:  #  blue green
        return 1
    else:
        return 0
    

def main():
    interval = 1/30
    listener = keyboard.Listener(on_press=on_key_press)
    listener.start()

    print("Waiting for key press...")
    while not capture:
        time.sleep(0.01)

    print("Capturing...")
    next_frame_time = time.perf_counter()
    
    while capture:
        frame_start = time.perf_counter()

        with mss.mss() as sct:
            img = np.array(sct.grab(region))
            img2 = np.array(sct.grab(region2))

        avg_color = img2[:, :, :3].mean(axis=(0, 1))
        if avg_color[0] < 110 and avg_color[1] < 110 and avg_color[2] < 110:
            avg_color = img[:, :, :3].mean(axis=(0, 1))

        press = classify_color(avg_color)
        if ACTIONS[press]:
            pydirectinput.press(ACTIONS[press])
            capture_screenshot(ACTIONS[press])
        else:
            capture_screenshot('_')

        # Precision sleep
        elapsed = time.perf_counter() - frame_start
        sleep_time = max(0, interval - elapsed)
        time.sleep(sleep_time)
        next_frame_time += interval

    print("Capture stopped.")

            
            
if __name__ == "__main__":
    main()