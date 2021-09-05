import os
import cv2
from tqdm import tqdm
import torch
from glob import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from facenet_pytorch import MTCNN

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=True, device=device)
TRAIN_IMAGE_DIR = "/opt/ml/input/data/train/images"
TRAIN_FACE_DIR = "/opt/ml/input/data/train/faces"
TEST_IMAGE_DIR = "/opt/ml/input/data/eval/images"
TEST_FACE_DIR = "/opt/ml/input/data/eval/faces"

os.makedirs(TRAIN_FACE_DIR, exist_ok=True)
os.makedirs(TEST_FACE_DIR, exist_ok=True)

dirs = glob(os.path.join(TRAIN_IMAGE_DIR, '*'))
dir_bar = tqdm(dirs)
length = len(dirs)

for idx, path in enumerate(dir_bar):
    new_path = path.replace('images', 'faces')

    if not os.path.exists(new_path):
        os.makedirs(new_path)

    if len(glob(os.path.join(path, '*'))) != 7:
        print("image_path")
        print(glob(os.path.join(path, '*')))

    for image_path in glob(os.path.join(path, '*')):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        boxes, probs = mtcnn.detect(image)
        if boxes is None:
            image = image[256 - 170:256 + 170, 192 - 128:192 + 128]
        else:
            x1, y1, x2, y2 = map(int, boxes[0])
            x1 = max(0, x1 - 30)
            y1 = max(0, y1 - 60)
            x2 = min(384, x2 + 30)
            y2 = min(512, y2 + 30)
            image = image[y1:y2, x1:x2]
        face_path = image_path.replace('images', 'faces')
        plt.imsave(face_path, image)

    if len(glob(os.path.join(new_path, '*'))) != 7:
        print("face_path")
        print(glob(os.path.join(new_path, '*')))

    dir_bar.set_description(f'{idx} / {length}')

dir_bar = tqdm(glob(os.path.join(TEST_IMAGE_DIR, '*')))
length = len(dir_bar)

for idx, image_path in enumerate(dir_bar):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    boxes, probs = mtcnn.detect(image)
    if boxes is None:
        image = image[256 - 170:256 + 170, 192 - 128:192 + 128]
    else:
        x1, y1, x2, y2 = map(int, boxes[0])
        x1 = max(0, x1 - 30)
        y1 = max(0, y1 - 60)
        x2 = min(384, x2 + 30)
        y2 = min(512, y2 + 30)
        image = image[y1:y2, x1:x2]
    face_path = image_path.replace('images', 'faces')
    plt.imsave(face_path, image)

    dir_bar.set_description(f'{idx} / {length}')