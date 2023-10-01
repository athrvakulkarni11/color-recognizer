import cv2
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial import distance
import pandas as pd

# Global variables to store mouse position
x, y = -1, -1

# Mouse callback function to update x, y on mouse events
def update_mouse_pos(event, _x, _y, flags, param):
    global x, y
    if event == cv2.EVENT_MOUSEMOVE:
        x, y = _x, _y

# Load your dataset with color names and RGB values
data = pd.read_csv('color_names.csv')

# Function to find the closest color name based on RGB values
def find_closest_color(rgb_values, color_data):
    min_distance = float('inf')
    closest_color = None
    closest_rgb = None
    for index, row in color_data.iterrows():
        dataset_rgb = np.array([row['Red (8 bit)'], row['Green (8 bit)'], row['Blue (8 bit)']])
        dst = distance.euclidean(rgb_values, dataset_rgb)
        if dst < min_distance:
            min_distance = dst
            closest_color = row['Name']
            closest_rgb = dataset_rgb
    return closest_color, closest_rgb

# Initialize the camera
cap = cv2.VideoCapture(0)  # 0 for the default camera, you can change it if you have multiple cameras

# Create a window and set the mouse callback function
cv2.namedWindow('Dominant Color Detection')
cv2.setMouseCallback('Dominant Color Detection', update_mouse_pos)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Get the position of the mouse cursor
    if x != -1 and y != -1:
        # Get the color of the pixel under the cursor
        pixel_color = frame[y, x]

        # Get the closest color name and corresponding RGB value based on the pixel color
        closest_color_name, closest_rgb = find_closest_color(pixel_color, data)

        # Convert closest_rgb from NumPy array to tuple of integers
        bgr_color = (int(closest_rgb[2]), int(closest_rgb[1]), int(closest_rgb[0]))

        # Draw a crosshair at the position of the mouse cursor
        crosshair_size = 15
        frame = cv2.line(frame, (x - crosshair_size, y), (x + crosshair_size, y), bgr_color, 2)
        frame = cv2.line(frame, (x, y - crosshair_size), (x, y + crosshair_size), bgr_color, 2)

        # Display the color name and RGB value of the pixel under the cursor
        cv2.putText(frame, f"Color: {closest_color_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, bgr_color, 2)
        cv2.putText(frame, f"RGB: {closest_rgb}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, bgr_color, 2)

    # Display the frame
    cv2.imshow('Dominant Color Detection', frame)

    # Press 'q' to exit the loop and close the window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
