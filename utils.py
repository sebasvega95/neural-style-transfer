import cv2
import itertools
import numpy as np
import pathlib


def preprocess_image_from_path(path, target_shape=None):
    image = cv2.imread(str(path))
    if target_shape is not None:
        image = cv2.resize(image, target_shape)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32)
    image = np.expand_dims(image, axis=0)
    return image


def image_generator(images_path, batch_size=1, target_shape=None):
    paths = itertools.cycle(pathlib.Path(images_path).glob('*.jpg'))
    while True:
        batch_paths = itertools.islice(paths, batch_size)
        images = [preprocess_image_from_path(str(path), target_shape=target_shape) for path in batch_paths]
        batch = np.concatenate(images, axis=0)
        yield batch


def get_num_images(path):
    return len(list(pathlib.Path(path).glob('*.jpg')))


def path_exists(path):
    return pathlib.Path(path).exists()
