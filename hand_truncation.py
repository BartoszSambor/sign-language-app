import cv2
import matplotlib.pyplot as plt
import mediapipe as mp


class HandSearch:
    def __init__(self, confidence=0.3, offset=100):
        self.confidence = confidence
        self.offset = offset
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=confidence)
        self.mp_drawing = mp.solutions.drawing_utils

    def get_hand_image(self, image, debug=False):
        hand_image = image.copy()

        h, w, c = hand_image.shape

        result = self.hands.process(cv2.cvtColor(hand_image, cv2.COLOR_BGR2RGB))
        if result.multi_hand_landmarks:
            for handLMs in result.multi_hand_landmarks:
                x_max = 0
                y_max = 0
                x_min = w
                y_min = h
                # Find bounding box of hands
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
                # Add offset
                y_min -= self.offset
                y_max += self.offset
                x_min -= self.offset
                x_max += self.offset

                # Make bounding box square
                width = x_max - x_min
                height = y_max - y_min
                if width > height:
                    diff = (width - height) // 2
                    y_min -= diff
                    y_max += diff
                else:
                    diff = (height - width) // 2
                    x_min -= diff
                    x_max += diff

                # Check if bounding box is inside image
                x_min = max(x_min, 0)
                y_min = max(y_min, 0)
                x_max = min(x_max, w)
                y_max = min(y_max, h)

                if debug:
                    img_c = hand_image.copy()
                    cv2.rectangle(img_c, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    self.mp_drawing.draw_landmarks(img_c, handLMs, self.mp_hands.HAND_CONNECTIONS)
                    plt.imshow(hand_image)
                    plt.imshow(img_c)
                    plt.show()

                # Truncate image of hands
                hand = hand_image[y_min:y_max, x_min:x_max]

                # Resize if squaring above didn't work
                if hand.shape[0] != hand.shape[1]:
                    hand = cv2.resize(hand, (hand.shape[0], hand.shape[0]))
                if debug:
                    print(hand.shape)
                return hand
        return None

# hs = HandSearch()
# img = cv2.imread("./tests/my_dataset/K_0.jpg")
# cv2.imshow("Hand", hs.get_hand_image(img, True))
# cv2.waitKey(0)
