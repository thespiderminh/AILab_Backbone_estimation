# import torch
# import cv2
# import mediapipe as mp
# import sklearn
# from matplotlib import pyplot as plt
import numpy as np
import scipy.io
# import time
# import os
# import tkinter as tk

# mp_holistic = mp.solutions.holistic
# mp_drawing = mp.solutions.drawing_utils

# root = tk.Tk()
# screen_width = root.winfo_screenwidth()
# screen_height = root.winfo_screenheight()

# def mediapipe_detection(image, model):
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     image.flags.writeable = False
#     results = model.process(image)
#     image.flags.writeable = True
#     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#     return image, results


# def draw_landmarks(image, results):
#     mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
#                               mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
#                               mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2))
    
# def get_data():
#     cap = cv2.VideoCapture("data.mp4")
#     cv2.namedWindow("frame", cv2.WND_PROP_FULLSCREEN)
#     cv2.setWindowProperty("frame", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

#     fourcc = cv2.VideoWriter_fourcc(*'XVID')
#     out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (screen_width, screen_height))
#     if not cap.isOpened():
#         print("Cannot open camera")
#     else:
#         with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
#             while cap.isOpened():

#                 # Read feed
#                 ret, frame = cap.read()
#                 if ret == True:
#                     frame, results = mediapipe_detection(frame, holistic)
#                     draw_landmarks(frame, results)

#                     frame = cv2.resize(frame,(screen_width, screen_height))
#                     out.write(frame)

#                     cv2.imshow('frame', frame)

#                     if cv2.waitKey(1) & 0xFF == ord('\x1b'): # ESC button
#                         break
#                 else:
#                     break


#     cap.release()
#     out.release()
#     cv2.destroyAllWindows()


def main():
    # get_data()
    mat = scipy.io.loadmat('data/lsp-master/joints.mat')
    print()
    print()
    print()
    print()
    print()
    print()
    print()
    print()
    print(mat['joints'])
    print()
    print()
    print()
    print()
    print()
    print()
    print()
    print()
    print(mat['joints'].transpose(2,1,0))
    print()
    print()
    print()
    print()
    print()
    print()
    print()
    print()


main()