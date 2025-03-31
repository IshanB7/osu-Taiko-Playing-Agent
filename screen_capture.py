import mss
import time
import threading
from pynput import keyboard
import os

# monitor width height and location
# start: 415, 505
# width, height: 2880 - 415 = 2465, 975 - 505 = 470

region = {'top': 450, 'left': 340, 'width': 280, 'height': 280}  # Screenshot region

os.makedirs('images', exist_ok=True)
counter = len(os.listdir('images'))
counter_lock = threading.Lock()

def capture_screenshot(label):
    global counter
    with mss.mss() as sct:
        img = sct.grab(region)

        with counter_lock:
            file_path = f"images/image_{counter}.png"

            mss.tools.to_png(img.rgb, img.size, output=file_path)

            with open('labels.txt', 'a') as f:
                f.write(f"{label}\n")

            counter += 1

def on_press(key):
    try:
        capture_screenshot(key.char)
    except AttributeError:
        pass

def on_release(key):
    global capture

    if key == keyboard.Key.enter:
        capture = not capture
    
capture = False 

def listener_thread():
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()

def main():
    global capture
    interval = 1 / 30.
    listener_thread_instance = threading.Thread(target=listener_thread)
    listener_thread_instance.daemon = True
    listener_thread_instance.start()

    while not capture:
        time.sleep(0.1)

    while capture:
        capture_screenshot('-')
        time.sleep(interval)

main()