
import random
import os
import itertools
import cv2
import numpy as np
import matplotlib.pyplot as plt

images_path = '/path/to/your/image/'
labels_path = '/path/to/your/label/'
matches = [100, 200, 300, 400, 500, 600, 700, 800]
def get_img_label_paths(images_path, labels_path):
    res = []
    for dir_entry in os.listdir(images_path):
        if os.path.isfile(os.path.join(images_path, dir_entry)):
            file_name, _ = os.path.splitext(dir_entry)
            res.append((os.path.join(images_path, file_name+".png"),
                        os.path.join(labels_path, file_name+".png")))
    return res


def get_image_array(img):
    return np.float32(img) / 127.5 - 1


def get_segmentation_array(img, nClasses):
    seg_labels = np.zeros((256, 256, nClasses))

    for m in matches:
        img[img == m] = matches.index(m)

    for c in range(nClasses):
        seg_labels[:, :, c] = (img == c).astype(int)

    seg_labels = np.reshape(seg_labels, (256*256, nClasses))
    return seg_labels


def image_segmentation_generator(images_path, labels_path, batch_size, num_class):
    img_seg_pairs = get_img_label_paths(images_path, labels_path)
    random.shuffle(img_seg_pairs)
    zipped = itertools.cycle(img_seg_pairs)
    while True:
        X = []
        Y = []
        for _ in range(batch_size):
            img, seg = next(zipped)

            img = cv2.imread(img, cv2.IMREAD_UNCHANGED)
            seg = cv2.imread(seg, cv2.IMREAD_UNCHANGED)

            X.append(get_image_array(img))
            Y.append(get_segmentation_array(seg, num_class))

        yield np.array(X), np.array(Y)

