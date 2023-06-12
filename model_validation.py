import os
import string

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from tensorflow import keras
from hand_truncation import HandSearch
from utils import letter_to_number, number_to_letter
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import cv2
import mediapipe as mp
import numpy as np

model = keras.models.load_model('./models/ASL/ASL.h5')
# model = keras.models.load_model('./models/ASLv2/model1.h5')

# model = keras.models.load_model('./models/mediapipe/smnist.h5')

hand_finder = HandSearch()


# input image, cut and scale it for neural network
# cmap: 'RGB' or 'GRAY'
# truncate_hands finds hand and cut out it
# equalize_value equalize histogram of third channel in HSV colorspace
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

    scaled = (scaled / np.max(scaled) * 255).astype("uint8")  # for uint8
    # scaled = (scaled / np.max(scaled)).astype("float") # for float
    return scaled



# cmap = "RGB" or "GRAY"
def load_dataset(output_size: (int, int), cmap):
    folder = "./tests/daniel_my_dataset/"
    raw_data = []
    for filename in os.listdir(folder):
        letter = filename.split("/")[-1][0]  # first letter of filename is a label
        raw_data.append((cv2.imread(folder + filename), letter))

    images = []
    labels = []
    for image, letter in raw_data:
        if cmap == "GRAY":
            channels = 1
        else:
            channels = 3
        images.append(prepare_image(image, output_size, cmap).reshape(-1, *output_size, channels))
        labels.append(letter_to_number(letter))

    return images, labels

# cmap = "RGB" or "GRAY"
def draw_confusion_matrix(x_test, y_test, output_size: (int, int), cmap):
    if cmap == "GRAY":
        channels = 1
    else:
        channels = 3
    x_test = np.array(x_test).reshape((-1, *output_size, channels))
    predictions = model.predict(x_test)
    predictions = np.argmax(predictions, axis=1)
    print(accuracy_score(y_test, predictions))
    cm = confusion_matrix(y_test, predictions)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot()
    plt.show()


def one_by_one_test(x_test, y_test):
    x_test = np.array(x_test).reshape((-1, 64, 64, 3))
    predictions = model.predict(x_test)
    predictions = np.argmax(predictions, axis=1)
    for i in range(len(x_test)):
        x = x_test[i]
        cv2.imshow("img", x)
        print(number_to_letter(predictions[i]), " expected: ", y_test[i])
        cv2.waitKey(0)


draw_confusion_matrix(*load_dataset((64, 64), "RGB"), (64, 64), "RGB")
# one_by_one_test(*load_dataset())
