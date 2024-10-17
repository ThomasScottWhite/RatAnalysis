import cv2
import tkinter as tk
from tkinter import filedialog
import numpy as np
import pandas as pd

# Select File
root = tk.Tk()
root.withdraw()

video_path = filedialog.askopenfilename(title="Select Video File", 
                                        filetypes=(("AVI files", "*.avi"), ("All files", "*.*")))

# Get a single frame
cap = cv2.VideoCapture(video_path)
sucess,frame = cap.read()
cap.release()

# Select points
selected_point = None

def select_point(event, x, y, flags, param):
    global selected_point
    if event == cv2.EVENT_LBUTTONDOWN:
        selected_point = (x, y)
        print(f"Point selected: {selected_point}")

image = frame #cv2.imread('frame.jpg')

cv2.namedWindow('Select Point')
cv2.setMouseCallback('Select Point', select_point)

while True:
    cv2.imshow('Select Point', image)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or selected_point is not None: 
        break

cv2.destroyAllWindows()

# Determine if the light is on
light_on_rgb = np.array([255, 243, 197])
light_off_rgb = np.array([250, 160, 135])

def is_light_on(selected_point,frame):
    x, y = selected_point
    pixel_rgb = frame[y, x]
    pixel_rgb = pixel_rgb[::-1]
    # Calculate the Euclidean distance to both the "on" and "off" reference colors
    dist_to_on = np.linalg.norm(pixel_rgb - light_on_rgb)
    dist_to_off = np.linalg.norm(pixel_rgb - light_off_rgb)

    # Return True if the light is closer to "on" color, otherwise False
    if dist_to_on < dist_to_off:
        return True
    else:
        return False  
    

if is_light_on(selected_point,frame):
    print("The light is ON.")
else:
    print("The light is OFF.")

# Create dataframe
data = []

cap = cv2.VideoCapture(video_path)
success, frame = cap.read()
frame_count = 1
while success:
    print(f"Processing frame {frame_count}")
    frame_count += 1
    light_status = is_light_on(selected_point,frame)
    data.append({"Light On": light_status})
    success, frame = cap.read()
cap.release()
df = pd.DataFrame(data)

df.to_csv(video_path[:-4] + ".csv")