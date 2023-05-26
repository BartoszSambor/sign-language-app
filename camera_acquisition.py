import cv2
import os

# number of images per letter
samples_num = 3

# Create a VideoCapture object
cap = cv2.VideoCapture(0)

# Check if the camera is opened successfully
if not cap.isOpened():
    print("Failed to open camera")
    exit()


def generate_filename():
    # asl language letters
    letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K',
               'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
               'V', 'W', 'X', 'Y']
    # 3 images per letter
    for letter in letters:
        for i in range(samples_num):
            yield f"./tests/my_dataset/{letter}_{i}.jpg"


filenames = list(generate_filename())

print(f"Next filename: {filenames[0]}")
if os.path.isfile(filenames[0]):
    print("Warning, existing file will be overwritten")

while True:
    idx = 0
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Check if frame is captured successfully
    if not ret:
        print("Failed to capture frame")
        break

    # Display the resulting frame
    cv2.imshow("Camera", frame)

    # Wait for the space key to be pressed
    if cv2.waitKey(1) == ord(' '):
        filename = filenames[idx]
        # Save the current frame as an image
        cv2.imwrite(filename, frame)
        idx += 1
        if idx == len(filenames):
            print("Done")
            break

        print("Image captured! Next filename is", filenames[idx])

        if os.path.isfile(filenames[idx]):
            print("Warning, existing file will be overwritten")

# Release the VideoCapture object and close all windows
cap.release()
cv2.destroyAllWindows()

