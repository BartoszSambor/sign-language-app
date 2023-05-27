import os
import string

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from tensorflow import keras

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import cv2
import mediapipe as mp
import numpy as np

model = keras.models.load_model('./models/ASL/ASL.h5')


# input RGB image, cut and scale it for neural network
def prepare_image(image):
    # media pipe init
    mphands = mp.solutions.hands
    hands = mphands.Hands()

    h, w, c = image.shape

    hand_landmarks = hands.process(image).multi_hand_landmarks
    hand = np.zeros((300, 300, 3))
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
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            # mp_drawing.draw_landmarks(frame, handLMs, mphands.HAND_CONNECTIONS)
            hand = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            hand = hand[y_min:y_max, x_min:x_max]


    # scaled = cv2.resize(hand, (64, 64))
    # version without mediapipe hand recognition
    scaled = cv2.resize(image, (64, 64))

    # TODO: I don;t know about correct type and range for input image
    # scaled = (scaled / np.max(scaled) * 255).astype("uint8")
    return scaled


# God, please forgive me
def letter_to_number(letter):
    if letter == 'A':
        number = 0
    elif letter == 'B':
        number = 1
    elif letter == 'C':
        number = 2
    elif letter == 'D':
        number = 3
    elif letter == 'E':
        number = 4
    elif letter == 'F':
        number = 5
    elif letter == 'G':
        number = 6
    elif letter == 'H':
        number = 7
    elif letter == 'I':
        number = 8
    elif letter == 'J':
        number = 9
    elif letter == 'K':
        number = 10
    elif letter == 'L':
        number = 11
    elif letter == 'M':
        number = 12
    elif letter == 'N':
        number = 13
    elif letter == 'O':
        number = 14
    elif letter == 'P':
        number = 15
    elif letter == 'Q':
        number = 16
    elif letter == 'R':
        number = 17
    elif letter == 'S':
        number = 18
    elif letter == 'T':
        number = 19
    elif letter == 'U':
        number = 20
    elif letter == 'V':
        number = 21
    elif letter == 'W':
        number = 22
    elif letter == 'X':
        number = 23
    elif letter == 'Y':
        number = 24
    elif letter == 'Z':
        number = 25
    elif letter == 'del':
        number = 26
    elif letter == 'nothing':
        number = 27
    elif letter == 'space':
        number = 28
    else:
        number = -1  # Unknown
    return number


def number_to_letter(number):
    if number == 0:
        label = 'A'
    elif number == 1:
        label = 'B'
    elif number == 2:
        label = 'C'
    elif number == 3:
        label = 'D'
    elif number == 4:
        label = 'E'
    elif number == 5:
        label = 'F'
    elif number == 6:
        label = 'G'
    elif number == 7:
        label = 'H'
    elif number == 8:
        label = 'I'
    elif number == 9:
        label = 'J'
    elif number == 10:
        label = 'K'
    elif number == 11:
        label = 'L'
    elif number == 12:
        label = 'M'
    elif number == 13:
        label = 'N'
    elif number == 14:
        label = 'O'
    elif number == 15:
        label = 'P'
    elif number == 16:
        label = 'Q'
    elif number == 17:
        label = 'R'
    elif number == 18:
        label = 'S'
    elif number == 19:
        label = 'T'
    elif number == 20:
        label = 'U'
    elif number == 21:
        label = 'V'
    elif number == 22:
        label = 'W'
    elif number == 23:
        label = 'X'
    elif number == 24:
        label = 'Y'
    elif number == 25:
        label = 'Z'
    elif number == 26:
        label = 'del'
    elif number == 27:
        label = 'nothing'
    elif number == 28:
        label = 'space'
    else:
        label = 'unknown'
    return label


def load_dataset():
    folder = "./tests/daniel_my_dataset/"
    raw_data = []
    for filename in os.listdir(folder):
        letter = filename.split("/")[-1][0]  # first letter of filename is a label
        raw_data.append((cv2.imread(folder + filename), letter))

    images = []
    labels = []
    for image, letter in raw_data:
        images.append(prepare_image(image).reshape(-1, 64, 64, 3))
        labels.append(letter_to_number(letter))

    return images, labels


def draw_confusion_matrix(x_test, y_test):
    x_test = np.array(x_test).reshape((-1, 64, 64, 3))
    predictions = model.predict(x_test)
    predictions = np.argmax(predictions, axis=1)
    print(accuracy_score(y_test, predictions))
    cm = confusion_matrix(y_test, predictions)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot()
    plt.show()


draw_confusion_matrix(*load_dataset())
