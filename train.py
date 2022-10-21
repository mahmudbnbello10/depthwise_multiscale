from dataloader import image_segmentation_generator
from ZNet import *

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def train(model, image_folder, label_folder, n_class, batch_size=4, epochs=4, weights_path=None):
    train_gen = image_segmentation_generator(
        image_folder, label_folder, batch_size, n_class)

    model.compile(loss='categorical_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])

    if weights_path != None:
        model.load_weights(weights_path)

    model.fit_generator(train_gen, 200, epochs=epochs)


def save_model(model, model_path):
    model.save_weights(model_path)


if __name__ == "__main__":
    weights_path = "weight.h5"
    image_folder = "train/images/"
    label_folder = "train/labels/"
    n_class = 8

    model = ZNet_v2()
    train(model, image_folder, label_folder,
          n_class, weights_path=weights_path)
