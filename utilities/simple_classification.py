import cv2
import os
import shutil
import numpy as np

# A simple image classfier to put the corresponding images in the corresponding folders
parent_folder = "train_data_by_folder"
# Input and output directories
input_folder = "frames_output"  # Folder containing the images

def is_beat_on(image):
    """Check if the beat is on based on the average color of the pixels within a circle around the center pixel."""
    center_x, center_y = 158, 200  # Center coordinates
    radius =15  # Radius around the center pixel

    # Create a mask with a filled circle
    mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    cv2.circle(mask, (center_x, center_y), radius, 255, -1)

    # Extract the region of interest (ROI) using the mask
    roi = cv2.bitwise_and(image, image, mask=mask)

    # Calculate the average color of the ROI
    mean_val = cv2.mean(image, mask=mask)
    average_color = np.array(mean_val[:3])  # Exclude the alpha channel if present

    # print(f"Average color: {average_color[2]:.2f}, {average_color[1]:.2f}, {average_color[0]:.2f}")

    # non of the following will give a perfect separation of the beat on and off images
    # however, this can greatly reduce the workload for manual classification

    # Check if the average color is within the specified range
    return ( average_color[2] +  # Red channel
             average_color[1] +  # Green channel
             average_color[0] >= 325)     # Blue channel

    # Check if the average color is within the specified range
    return (145 <= average_color[2] and  # Red channel
            145 <= average_color[1] and  # Green channel
            145 <= average_color[0])     # Blue channel

    return (150 <= average_color[2] and  # Red channel
            170 <= average_color[1] and  # Green channel
            170 <= average_color[0])     # Blue channel

def is_red_beat(image):
    center_x, center_y = 56, 197  # Center coordinates
    radius = 34  # Radius around the center pixel

    # Create masks for the left and right halves of the circle
    mask_left = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    mask_right = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    
    # Draw the left half of the circle
    cv2.ellipse(mask_left, (center_x, center_y), (radius, radius), 0, 90, 270, 255, -1)
    
    # Draw the right half of the circle
    cv2.ellipse(mask_right, (center_x, center_y), (radius, radius), 0, -90, 90, 255, -1)

    # # Visualize the masks
    # cv2.imshow("Left Half Mask", mask_left)
    # cv2.imshow("Right Half Mask", mask_right)

    # # Overlay the masks on the original image
    # overlay_left = cv2.addWeighted(image, 0.7, cv2.cvtColor(mask_left, cv2.COLOR_GRAY2BGR), 0.3, 0)
    # overlay_right = cv2.addWeighted(image, 0.7, cv2.cvtColor(mask_right, cv2.COLOR_GRAY2BGR), 0.3, 0)

    # # Display the overlays
    # cv2.imshow("Overlay Left Half", overlay_left)
    # cv2.imshow("Overlay Right Half", overlay_right)
    # cv2.waitKey(0)  # Wait for a key press to close the windows
    # cv2.destroyAllWindows()  # Close all windows

    # Calculate the average color of the left half
    mean_val_left = cv2.mean(image, mask=mask_left)
    average_color_left = np.array(mean_val_left[:3])  # Exclude the alpha channel if present

    # Calculate the average color of the right half
    mean_val_right = cv2.mean(image, mask=mask_right)
    average_color_right = np.array(mean_val_right[:3])  # Exclude the alpha channel if present

    red, green, blue = 120, 100, 100
    # Check if the average color of the left half is within the specified range
    left_half_check = (red <= average_color_left[2]  and  # Red channel
                       green <= average_color_left[1]  and  # Green channel
                       blue <= average_color_left[0])     # Blue channel

    # Check if the average color of the right half is within the specified range
    right_half_check = (red <= average_color_right[2] and  # Red channel
                        green <= average_color_right[1] and  # Green channel
                        blue <= average_color_right[0])     # Blue channel

    return left_half_check, right_half_check


def is_blue_beat(image):
    center_x, center_y = 56, 197  # Center coordinates
    inner_radius = 34  # Inner radius of the annulus
    outer_radius = 48  # Outer radius of the annulus

    # Create a mask for the entire annulus
    mask_annulus = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    cv2.circle(mask_annulus, (center_x, center_y), outer_radius, 255, -1)
    cv2.circle(mask_annulus, (center_x, center_y), inner_radius, 0, -1)

    # Create masks for the left and right halves of the annulus
    mask_left = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    mask_right = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    
    # Draw the left half of the annulus
    cv2.ellipse(mask_left, (center_x, center_y), (outer_radius, outer_radius), 0, 90, 270, 255, -1)
    cv2.ellipse(mask_left, (center_x, center_y), (inner_radius, inner_radius), 0, 90, 270, 0, -1)
    
    # Draw the right half of the annulus
    cv2.ellipse(mask_right, (center_x, center_y), (outer_radius, outer_radius), 0, -90, 90, 255, -1)
    cv2.ellipse(mask_right, (center_x, center_y), (inner_radius, inner_radius), 0, -90, 90, 0, -1)

    # # Visualize the masks
    # cv2.imshow("Left Half Annulus Mask", mask_left)
    # cv2.imshow("Right Half Annulus Mask", mask_right)

    # # Overlay the masks on the original image
    # overlay_left = cv2.addWeighted(image, 0.7, cv2.cvtColor(mask_left, cv2.COLOR_GRAY2BGR), 0.3, 0)
    # overlay_right = cv2.addWeighted(image, 0.7, cv2.cvtColor(mask_right, cv2.COLOR_GRAY2BGR), 0.3, 0)

    # # Display the overlays
    # cv2.imshow("Overlay Left Half Annulus", overlay_left)
    # cv2.imshow("Overlay Right Half Annulus", overlay_right)
    # cv2.waitKey(0)  # Wait for a key press to close the windows
    # cv2.destroyAllWindows()  # Close all windows

    # Calculate the average color of the left half of the annulus
    mean_val_left = cv2.mean(image, mask=mask_left)
    average_color_left = np.array(mean_val_left[:3])  # Exclude the alpha channel if present

    # Calculate the average color of the right half of the annulus
    mean_val_right = cv2.mean(image, mask=mask_right)
    average_color_right = np.array(mean_val_right[:3])  # Exclude the alpha channel if present

    red_lb, green_lb, blue_lb = 100, 90, 90
    red_ub, green_ub, blue_ub = 115, 255, 255

    # Check if the average color of the left half is within the specified range
    left_half_check = (  # Red channel
                        green_lb <= average_color_left[1] and  # Green channel
                        blue_lb <= average_color_left[0] and      # Blue channel
                        average_color_left[2] <= red_ub)    # Red channel

    # Check if the average color of the right half is within the specified range
    right_half_check = (  # Red channel
                        green_lb <= average_color_right[1] and  # Green channel
                        blue_lb <= average_color_right[0] and      # Blue channel
                        average_color_right[2] <= red_ub)    # Red channel
    
    return left_half_check, right_half_check


def process_image_by_beat_on_off(input_folder="input", output_folder_on = "beat_on", output_folder_off = "beat_off"):
    os.makedirs(output_folder_on, exist_ok=True)
    os.makedirs(output_folder_off, exist_ok=True)
    # Process each image in the directory
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # Only process image files
            image_path = os.path.join(input_folder, filename)
            image = cv2.imread(image_path)
            
            # cv2.imshow("Image", image)
            # cv2.waitKey(0)  # Wait for a key press
            # cv2.destroyAllWindows()  # Close the window

            if image is None:
                continue  # Skip unreadable files

            # Check for color matches
            if is_beat_on(image):
                shutil.copy(image_path, os.path.join(output_folder_on, filename))
            else:
                shutil.copy(image_path, os.path.join(output_folder_off, filename))
    print(f"Processing complete. Images saved in '{output_folder_on}' and '{output_folder_off}'")


def process_image_by_color_range(output_folder1 = "red", output_folder2 = "blue"):
    os.makedirs(output_folder1, exist_ok=True)
    os.makedirs(output_folder2, exist_ok=True)
    os.makedirs("bad", exist_ok=True)

    for filename in os.listdir(input_folder):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # Only process image files
                image_path = os.path.join(input_folder, filename)
                image = cv2.imread(image_path)

                if image is None:
                    continue  # Skip unreadable files
                
                red_left, red_right = is_red_beat(image)
                blue_left, blue_right = is_blue_beat(image)
                red = red_left or red_right
                blue = blue_left or blue_right  
                # Check for red beat
                if red :
                    shutil.copy(image_path, os.path.join(output_folder1, filename))
                # Check for blue beat
                if blue:
                    shutil.copy(image_path, os.path.join(output_folder2, filename))
                if not red and not blue:
                    shutil.copy(image_path, os.path.join("bad", filename))
    print(f"Processing complete. Images saved in '{output_folder1}' and '{output_folder2}'")

# debug function to check the color of the image
def check_image():
    input_folder = "problematic"
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # Only process image files
            image_path = os.path.join(input_folder, filename)
            image = cv2.imread(image_path)
            if image is None:
                continue  # Skip unreadable files


def main():
    # First, process images by color
    process_image_by_color_range(output_folder1=os.path.join(parent_folder, "red"), output_folder2=os.path.join(parent_folder, "blue"))

    # Then, process images in the red folder to check for the beat
    process_image_by_beat_on_off(input_folder=os.path.join(parent_folder, "red"), output_folder_on=os.path.join(parent_folder, "red_on_beat"), output_folder_off=os.path.join(parent_folder, "red_off_beat"))

    # Finally, process images in the blue folder to check for the beat
    process_image_by_beat_on_off(input_folder=os.path.join(parent_folder, "blue"), output_folder_on=os.path.join(parent_folder, "blue_on_beat"), output_folder_off=os.path.join(parent_folder, "blue_off_beat"))

    # Delete the red and blue folders after processing
    shutil.rmtree(os.path.join(parent_folder, "red"))
    shutil.rmtree(os.path.join(parent_folder, "blue"))

if __name__ == "__main__":
    main()