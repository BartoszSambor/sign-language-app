import cv2
import os

# number of images per letter
samples_num = 50
# set this to not overwrite others images
# e.g. samples_num=2 starting_index=3 gives files A_3.jpg, A_4.jpg, B_3.jpg, B_4.jpg ,...
starting_index = 0
folder_name = "dataset_large_1"  # create directory before

# Create a VideoCapture object
cap = cv2.VideoCapture(0)

# Check if the camera is opened successfully
if not cap.isOpened():
    print("Failed to open camera")
    exit()


def generate_filename():
    # asl language letters
    # letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K',
    #            'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
    #            'V', 'W', 'X', 'Y']
    # letters = ['A', 'B', 'C', 'O', 'V', 'W']
    # letters = ['D', 'E', 'F', 'G', 'H', 'I', 'K',
    #            'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U',
    #            'X', 'Y']
    letters = ['U', 'X', 'Y']
    # 3 images per letter
    for letter in letters:
        for i in range(samples_num):
            yield f"./tests/{folder_name}/{letter}_{i + starting_index}.jpg"


filenames = list(generate_filename())
print(f"Next filename: {filenames[0]}")
if os.path.isfile(filenames[0]):
    print("Warning, existing file will be overwritten")

idx = 0
while True:
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

