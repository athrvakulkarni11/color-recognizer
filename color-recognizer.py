import cv2
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial import distance
import pandas as pd


x, y = -1, -1


def update_mouse_pos(event, _x, _y, flags, param):
    global x, y
    if event == cv2.EVENT_MOUSEMOVE:
        x, y = _x, _y


data = pd.read_csv('color_names.csv')


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


cap = cv2.VideoCapture(0)  


cv2.namedWindow('Dominant Color Detection')
cv2.setMouseCallback('Dominant Color Detection', update_mouse_pos)

while True:
    
    ret, frame = cap.read()

    
    if x != -1 and y != -1:
        
        pixel_color = frame[y, x]

        
        closest_color_name, closest_rgb = find_closest_color(pixel_color, data)

        
        bgr_color = (int(closest_rgb[2]), int(closest_rgb[1]), int(closest_rgb[0]))

        
        crosshair_size = 15
        frame = cv2.line(frame, (x - crosshair_size, y), (x + crosshair_size, y), bgr_color, 2)
        frame = cv2.line(frame, (x, y - crosshair_size), (x, y + crosshair_size), bgr_color, 2)

        
        cv2.putText(frame, f"Color: {closest_color_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, bgr_color, 2)
        cv2.putText(frame, f"RGB: {closest_rgb}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, bgr_color, 2)

    
    cv2.imshow('Dominant Color Detection', frame)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
