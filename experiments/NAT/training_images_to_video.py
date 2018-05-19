import sys
import os

import tensorflow as tf
import numpy as np
from tqdm import tqdm
import png
from scipy.misc import imsave


def save_images_from_training(path_to_events_file, path_to_photos):
    images = []
    for summary in tqdm(tf.train.summary_iterator(path_to_events_file)):
        try:
            step, wall_time = summary.step, summary.wall_time
            summary = summary.summary.value[0]
            image_tag = summary.tag
            assert image_tag.startswith('validation_noise_image')

            image_summary = summary.image
            _, _, image_bytes = image_summary.width, image_summary.height, image_summary.encoded_image_string
            png_pixels = png.Reader(bytes=image_bytes).asDirect()[2]
            image_array = np.vstack(png_pixels)
            images += [(image_array, step, wall_time)]
        except Exception as e:
            pass
    save_images_to_dir(images, path_to_photos)


def save_images_to_dir(images, path):
    max_step = max([step for _, step, _ in images])
    base = int(np.log10(max_step)) + 1
    name_format_string = f'{path}/image_rep_{{:0{base}d}}.png'
    os.makedirs(path, exist_ok=True)
    for image, step, _ in tqdm(images):
        imsave(name_format_string.format(step), image)


if __name__ == '__main__':
    args = sys.argv[1:]
    assert len(args) == 2
    save_images_from_training(*args)
