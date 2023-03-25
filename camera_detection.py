from tensorflow import keras
import numpy as np
import pandas as pd
import cv2
import string

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
        scaled = cv2.resize(gray, (28, 28))
        # cv2.imshow("scaled", scaled)
        input_im = scaled.reshape(-1, 28, 28, 1)
        results = model.predict(input_im)
        pred = np.argmax(results, axis=1)
        letter = dict(enumerate(string.ascii_uppercase))
        print(pred, letter[pred[0]])
    cap.release()


cam_pred()