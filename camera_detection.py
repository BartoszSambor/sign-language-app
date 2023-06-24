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
from utils import letter_to_number, number_to_letter
from hand_truncation import HandSearch

hand_finder = HandSearch()

# model = keras.models.load_model('./models/ASL/ASL.h5')

model = keras.models.load_model('./models/mobineNetFineT1/mobileNetV2+10.h5')

def prepare_image(image, output_size: (int, int), cmap, truncate_hands=True, equalize_value=True):
    image = image.copy()

    if equalize_value:
        # Convert to HSV and equalize V
        image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        h = image[:, :, 2]
        h = cv2.equalizeHist(h)
        image[:, :, 2] = h
        image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)

    if truncate_hands:
        hands = hand_finder.get_hand_image(image)
        if hands is not None:  # If hands found
            scaled = cv2.resize(hands, output_size)
        else:  # If hands not found
            scaled = cv2.resize(image, output_size)
    else:
        scaled = cv2.resize(image, output_size)

    if cmap == "GRAY":
        scaled = cv2.cvtColor(scaled, cv2.COLOR_RGB2GRAY)

    # cv2.imshow("scaled", cv2.resize(scaled, (300, 300)))
    # cv2.waitKey(100)

    # scaled = (scaled / np.max(scaled) * 255).astype("uint8")  # for uint8
    scaled = (scaled / np.max(scaled)).astype("float") # for float
    return scaled


# detect sing from camera input but it is very bad
def cam_pred_mediapipe():
    cap = cv2.VideoCapture(0)
    _, frame = cap.read()
    h, w, c = frame.shape

    cap = cv2.VideoCapture(0)
    while (cap.isOpened()):
        ret, frame = cap.read()
        if cv2.waitKey(100) & 0xFF == ord('q'):  # break the loop when the 'q' key is pressed
            break

        cv2.imshow('RGB', frame)
        # scaled = prepare_image(frame, (28, 28), "GRAY")
        # cv2.imshow("scaled", cv2.resize(scaled.reshape(28, 28), (300, 300)))
        # input_im = scaled.reshape(-1, 28, 28, 1)
        scaled = prepare_image(frame, (224, 224), "RGB", True, False)

        cv2.imshow("scaled", cv2.resize(scaled, (300, 300)))
        input_im = scaled.reshape(-1, 224, 224, 3)
        results = model.predict(input_im)
        pred = np.argmax(results, axis=1)
        print(pred, number_to_letter(pred))
        # print(results)
    cap.release()


cam_pred_mediapipe()
