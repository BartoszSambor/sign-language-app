import os
import random
from PIL import Image
import numpy as np
import cv2
from hand_truncation import HandSearch

from albumentations import (
    Compose,
    Rotate,
    RandomBrightnessContrast,
    Resize,
    RandomResizedCrop,
    GaussNoise,
)



def augment_images(in_dir, out_dir, num_augmentations_per_image, image_width, image_height):
    # Define the augmentation transformations
    transform = Compose([
        Rotate(limit=20, p=0.5),
        RandomBrightnessContrast(p=0.5),
        Resize(height=image_width, width=image_height),
        RandomResizedCrop(image_width, image_height, scale=(0.9, 0.9)),
        GaussNoise(15, p=0.75)
    ])

    if not os.path.isdir(in_dir):
        print("Directory doesn't exist.")
        print(os.path)

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    # Loop through each ASL file
    for filename in os.listdir(in_dir):

        # Get image name
        image_path = os.path.join(in_dir, filename)
        # Load image
        image = Image.open(image_path)

        # Crop left and right size of image, assumes x > y
        x, y = image.size
        padding = (x - y) // 2
        image = image.crop([padding, 0, x-padding, y])
        for i in range(num_augmentations_per_image):

            # Augmentation
            augmented_image = transform(image=np.array(image))
            augmented_image = Image.fromarray(augmented_image["image"])
            # Save image
            augmented_image_path = os.path.join(out_dir, f"{filename.split('.')[0]}_aug_{i}.{filename.split('.')[1]}")
            # print("Saved", augmented_image_path)
            augmented_image.save(augmented_image_path)
            # Debug:
            # image_np = np.array(augmented_image)
            # cv2.imshow("augumented", cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
            # cv2.waitKey(0)


# Usage example
augment_images("tests/dataset_large_1", "tests/dataset_aug", 2, 224, 224)


# transform = Compose([
#         Rotate(limit=20, p=0.75),
#         RandomBrightnessContrast(p=0.75),
#         Resize(height=256, width=256),
#         RandomResizedCrop(640, 480, scale=(0.9, 0.9), p=0.60),
#         GaussNoise(15, p=0.75)
#
#     ])
#
# image = Image.open("./tests/dataset_large_1/A_0.jpg")
# size = image.size
#
#
# for i in range(100):
#     image = Image.open("./tests/dataset_large_1/A_0.jpg")
#     augmented_image = transform(image=np.array(image))
#     augmented_image = Image.fromarray(augmented_image["image"])
#     image_np = np.array(augmented_image)
#     cv2.imshow("augumented", cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
#     cv2.waitKey(0)

