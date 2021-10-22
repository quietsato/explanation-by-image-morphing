import tensorflow as tf
from tensorflow.keras import datasets, optimizers

import matplotlib.pyplot as plt
import imageio
import os

if __name__ == "__main__":
    import sys
    sys.path.append(os.path.dirname(__file__))

    from config import *
    from util import create_out_dir, get_time_str, preprocess_image
    from models import build_classifier, IDCVAE, reconstruction_loss


tf.random.set_seed(42)


C_WEIGHT_FILENAME = None
VAE_WEIGHT_FILENAME = "CVAE.h5"

OUT_DIR = create_out_dir("main")


def main():
    time_str = get_time_str()

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

    encode_decode_test(test_images[:10], test_labels[:10], vae, time_str)
    cvae_decode_single_image_with_every_label(test_images[0], test_labels[0], vae, "test0", time_str)


def encode_decode_test(xs: tf.Tensor, ys: tf.Tensor, vae: IDCVAE, time_str: str):
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


def cvae_decode_single_image_with_every_label(x: tf.Tensor, y: tf.Tensor, vae: IDCVAE, image_id: str, time_str: str):
    decode_cols = (num_classes+1)//2
    figure_cols = 1 + decode_cols

    z, _, _ = vae.encode(tf.expand_dims(x, 0))
    zs = tf.tile(z, [num_classes, 1])
    rec_xs = vae.decode(zs, tf.range(num_classes))

    plt.figure(figsize=(figure_cols, 3))

    plt.subplot(2, figure_cols, 1)
    plt.imshow(x[:, :, 0], cmap='gray')
    plt.axis('off')
    plt.title(f'orig: {y}')

    for i in range(num_classes):
        plt.subplot(2, figure_cols, figure_cols * (i // decode_cols) + (i % decode_cols) + 2)
        plt.imshow(rec_xs[i, :, :, 0], cmap='gray')
        plt.axis('off')
        plt.title(f'label: {i}')

    plt.savefig(
        os.path.join(OUT_DIR, f"{time_str}_{image_id}_decode_single_image_with_every_label.png")
    )


if __name__ == "__main__":
    main()
