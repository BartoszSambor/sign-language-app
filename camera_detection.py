from tensorflow import keras
import numpy as np
import pandas as pd
import cv2
import string

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import cv2
import mediapipe as mp
from keras.models import load_model
import numpy as np
import time


model = keras.models.load_model('./models/model2/best.h5')
# model = keras.models.load_model('./models/model1')

# detect sing from camera input but it is very bad
def cam_pred():
    cap = cv2.VideoCapture(0)
    while (cap.isOpened()):
        ret, frame = cap.read()
        cv2.imshow('RGB', frame)
        if cv2.waitKey(100) & 0xFF == ord('q'):  # break the loop when the 'q' key is pressed
            break

        # frame processing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = gray / 255.0
        scaled = cv2.resize(gray, (28, 28))

        cv2.imshow("scaled", cv2.resize(scaled.reshape(28, 28), (300, 300)))
        input_im = scaled.reshape(-1, 28, 28, 1)
        results = model.predict(input_im)
        pred = np.argmax(results, axis=1)
        letter = dict(enumerate(string.ascii_uppercase))
        print(pred, letter[pred[0]])
    cap.release()

def cam_pred_mediapipe():
    model = load_model('models/mediapipe/smnist.h5')

    mphands = mp.solutions.hands
    hands = mphands.Hands()

    cap = cv2.VideoCapture(0)
    _, frame = cap.read()
    h, w, c = frame.shape


    cap = cv2.VideoCapture(0)
    while (cap.isOpened()):
        ret, frame = cap.read()
        if cv2.waitKey(100) & 0xFF == ord('q'):  # break the loop when the 'q' key is pressed
            break

        result = hands.process(frame)
        hand = np.zeros((300, 300))
        hand_landmarks = result.multi_hand_landmarks
        if hand_landmarks:
            for handLMs in hand_landmarks:
                x_max = 0
                y_max = 0
                x_min = w
                y_min = h
                for lm in handLMs.landmark:
                    x, y = int(lm.x * w), int(lm.y * h)
                    if x > x_max:
                        x_max = x
                    if x < x_min:
                        x_min = x
                    if y > y_max:
                        y_max = y
                    if y < y_min:
                        y_min = y
                y_min -= 40
                y_max += 40
                x_min -= 50
                x_max += 50
                if x_min < 0:
                    x_min = 0
                if y_min < 0:
                    y_min = 0
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                # mp_drawing.draw_landmarks(frame, handLMs, mphands.HAND_CONNECTIONS)
                hand = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                hand = hand[y_min:y_max, x_min:x_max]



        cv2.imshow('RGB', frame)
        scaled = cv2.resize(hand, (28, 28))
        cv2.imshow("scaled", cv2.resize(scaled.reshape(28, 28), (300, 300)))
        input_im = scaled.reshape(-1, 28, 28, 1)
        results = model.predict(input_im)
        pred = np.argmax(results, axis=1)
        letter = dict(enumerate(string.ascii_uppercase))
        print(pred, letter[pred[0]])
    cap.release()


# cam_pred()
cam_pred_mediapipe()