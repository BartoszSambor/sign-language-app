from tensorflow import keras
import numpy as np
import pandas as pd
import cv2
import string

model = keras.models.load_model('./models/model1')