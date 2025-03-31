import cv2
import os

# Input video file
video_path = r"2025-03-13 02-11-00.mkv"  # Change this to the video file path
output_folder = "frames_output"

# Create output directory if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Open the video file
cap = cv2.VideoCapture(video_path)

frame_number = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Stop if the video ends

    # Save the frame as a JPEG image
    frame_filename = os.path.join(output_folder, f"frame_{frame_number:04d}.jpg")
    cv2.imwrite(frame_filename, frame)

    frame_number += 1

cap.release()
cv2.destroyAllWindows()

print(f"Frames saved in '{output_folder}'")
