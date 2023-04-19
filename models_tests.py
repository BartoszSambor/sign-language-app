from tensorflow import keras
import numpy as np
import pandas as pd
import cv2
import string

# model = keras.models.load_model('./models/model2/best.h5')
# model = keras.models.load_model('./models/model1')
model = keras.models.load_model('./notebooks/models/model_7_epoch.h5')

sign_test = pd.read_csv('./tests/sign_mnist_test/sign_mnist_test.csv')
letter = dict(enumerate(string.ascii_uppercase))


# test evry test case
def file_pred():
    sign_test.head()

    inputs_test = sign_test.iloc[:, 1:].to_numpy()
    targets_test = sign_test['label'].to_numpy()

    # normalize inputs
    inputs_test = inputs_test / 255.0

    inputs_test = inputs_test.reshape(-1, 28, 28, 1)

    results = model.predict(inputs_test)
    pred = np.argmax(results, axis=1)

    Y_train = keras.utils.to_categorical(targets_test)
    true = np.argmax(Y_train, axis=1)

    pred = [letter[i] for i in pred]
    true = [letter[i] for i in true]
    print("Predicted: ", pred)
    print("Truth: ", true)
    accuracy = [1 if pred[i] == true[i] else 0 for i in range(len(pred))]
    accuracy = np.sum(accuracy) / len(accuracy)
    print("Accuracy: ", accuracy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# show random images with NN prediction to visually evaluate images
# use 'n' key to show next image and 'q' to quit
def random_cases_inspect():
    i = 0
    key = ord("n")
    while True:
        if key & 0xFF == ord('q'):  # break the loop when the 'q' key is pressed
            break
        elif key & 0xFF == ord('n'):  # next frame on 'n' pressed
            cv2.destroyAllWindows()
            show_random_test()
            i += 1
            cv2.waitKey(10)
        key = cv2.waitKey(1)


# show random image with description from test set
def show_random_test():
    case = np.random.randint(0, len(sign_test))

    inputs_test = sign_test.iloc[:, 1:].to_numpy()[case]
    targets_test = sign_test['label'].to_numpy()[case]

    inputs_test = inputs_test.reshape(-1, 28, 28, 1)
    result = model.predict(inputs_test)
    result = np.argmax(result)
    result = letter[result]
    truth = letter[targets_test]
    print("Pred: ", result, " Expected: ", truth)
    cv2.imshow("Pred: " + str(result) + " Expe:" + str(truth),
               cv2.resize(inputs_test.reshape(28, 28).astype('uint8'), (300, 300)))


# file_pred()
random_cases_inspect()
