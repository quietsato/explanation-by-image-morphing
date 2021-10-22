import tensorflow as tf
from tensorflow.keras import datasets, optimizers

import matplotlib.pyplot as plt
import imageio
import os


tf.random.set_seed(42)


C_WEIGHT_FILENAME = None
VAE_WEIGHT_FILENAME = "CVAE.h5"


def main():
    time_str = get_time_str()
    OUT_DIR = create_out_dir("main")

    _, (test_images, test_labels) = datasets.mnist.load_data()
    test_images = preprocess_image(test_images)

    classifier = build_classifier()

    vae = IDCVAE()
    vae.call(test_images[0:1], test_labels[0:1])
    vae.compile()

    if C_WEIGHT_FILENAME is not None:
        weight_path = os.path.join(OUT_BASE_DIR, "C", C_WEIGHT_FILENAME)
        classifier.load_weights(weight_path)

    if VAE_WEIGHT_FILENAME is not None:
        weight_path = os.path.join(OUT_BASE_DIR, "VAE", VAE_WEIGHT_FILENAME)
        vae.load_weights(weight_path)

    encode_decode_test(test_images[:10], test_labels[:10], vae, OUT_DIR, time_str)


def encode_decode_test(xs, ys, vae, OUT_DIR: str, time_str: str):
    n = min(len(xs), len(ys))
    zs, _, _ = vae.encode(xs)
    rec_xs = vae.decode(zs, ys)
    plt.figure(figsize=(2, n))
    for i in range(n):
        plt.subplot(n, 2, 2*i+1)
        plt.imshow(xs[i, :, :, 0], cmap='gray')
        plt.title(ys[i])
        plt.axis('off')

        plt.subplot(n, 2, 2*i+2)
        plt.imshow(rec_xs[i, :, :, 0], cmap='gray')
        plt.axis('off')

    plt.savefig(os.path.join(OUT_DIR, f"{time_str}_encode_decode.png"))


if __name__ == "__main__":
    import sys
    sys.path.append(os.path.dirname(__file__))

    from config import *
    from util import create_out_dir, get_time_str, preprocess_image
    from models import build_classifier, IDCVAE

    main()
