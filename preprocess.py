import cv2
import os

dir = 'images'

for filename in os.listdir(dir):
    image_path = os.path.join(dir, filename)

    image = cv2.imread(image_path)
    resized_image = cv2.resize(image, (16, 16))

    cv2.imwrite(image_path, resized_image)
    
