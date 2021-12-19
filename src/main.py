import tensorflow as tf
from tensorflow.keras import datasets

import matplotlib.pyplot as plt
import os
from tqdm import tqdm

import argparse

if __name__ == "__main__":
    import sys
    sys.path.append(os.path.dirname(__file__))

    from config import *
    from util import create_out_dir, get_time_str, preprocess_image
    from models import IDCVAE

tf.random.set_seed(42)

IDCVAE_LATENT_DIM = 16
time_str = get_time_str()
OUT_DIR = create_out_dir(f"main/{time_str}")
TEST_IMAGE_OUT_DIR = create_out_dir(f"main/{time_str}/test_image")
TEST_MISCLASSIFIED_OUT_DIR = create_out_dir(f"main/{time_str}/test_misclassified")


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "model_weight_dir",
        metavar="MODEL_WEIGHT_DIR",
        type=str
    )
    arg_parser.add_argument(
        "-n",
        type=int,
        default=10,
        help="Number of morphing images to generate"
    )
    args = arg_parser.parse_args()

    WEIGHT_FILEPATH = args.model_weight_dir

    print("==> Setup dataset")
    (train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
    train_images = preprocess_image(train_images)
    test_images = preprocess_image(test_images)

    print("==> Setup model")
    model = IDCVAE(IDCVAE_LATENT_DIM)
    model.compile()

    if WEIGHT_FILEPATH is not None:
        model.encoder.load_weights(os.path.join(WEIGHT_FILEPATH, "model_encoder.h5"))
        model.decoder.load_weights(os.path.join(WEIGHT_FILEPATH, "model_decoder.h5"))

    print("==> Find representative points")
    representative = model.find_representative_points(train_images, train_labels)
    save_representative_images(
        representative,
        model,
        os.path.join(OUT_DIR, "representative_points.png")
    )

    print("==> Classify images")
    test_pred = model.predict(test_images, verbose=1, batch_size=1024)
    acc = count_successfully_classified(test_pred, test_labels).numpy()
    print(f"Test Accuracy: {acc}/{len(test_labels)}")

    misclassified_mask = tf.not_equal(test_pred, test_labels)
    test_images_misclassified = tf.boolean_mask(test_images, misclassified_mask)
    test_labels_misclassified = tf.boolean_mask(test_labels, misclassified_mask)
    test_pred_misclassified = tf.boolean_mask(test_pred, misclassified_mask)

    print("==> Create morphing images")

    test_index = list(range(10))
    for i in tqdm(test_index):
        generate_morphing_images(
            test_images[i],
            test_pred[i],
            representative,
            model,
            args.n,
            create_out_dir(
                os.path.join(
                    TEST_IMAGE_OUT_DIR,
                    f"{i:05}-correct_{test_labels[i]}-pred_{test_pred[i]}"
                )
            )
        )

    for i in tqdm(range(len(test_images_misclassified))):
        generate_morphing_images(
            test_images_misclassified[i],
            test_pred_misclassified[i],
            representative,
            model,
            args.n,
            create_out_dir(
                os.path.join(
                    TEST_MISCLASSIFIED_OUT_DIR,
                    f"{i:05}-correct_{test_labels_misclassified[i]}-pred_{test_pred_misclassified[i]}"
                )
            )
        )


def count_successfully_classified(y, y_pred) -> tf.Tensor:
    return tf.reduce_sum(tf.cast(y_pred == y, tf.int32))


def save_representative_images(representative: tf.Tensor,
                               model: IDCVAE,
                               out_path: str):
    plt.figure(figsize=((num_classes+1)//2, 2))

    ys = tf.range(num_classes, dtype=tf.int32)
    xs = model.decode(representative, ys)
    for i, x in enumerate(xs.numpy()):
        plt.subplot(2, (num_classes + 1)//2, i + 1)
        plot_single_image(x)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0)


def generate_morphing_images(x, y, representative, model: IDCVAE, n: int, out_dir: str):
    plt.clf()
    plt.figure(figsize=(1, 1))

    def get_image_save_path(i):
        return os.path.join(out_dir, f"{i:03}.png")

    if not os.path.exists(os.path.dirname(get_image_save_path(0))):
        os.makedirs(os.path.dirname(get_image_save_path(0)))

    plt.figure(figsize=(1, 1))
    plot_single_image(x)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "original.png"), bbox_inches='tight', pad_inches=0)
    plt.clf()

    xs = model.generate_morphing(x, y, representative, n)

    for i in range(n + 1):
        plt.clf()
        plt.figure(figsize=(1, 1))
        plot_single_image(xs[i])
        plt.tight_layout()
        plt.savefig(get_image_save_path(i), bbox_inches='tight', pad_inches=0)
        plt.clf()


def plot_single_image(x: tf.Tensor):
    plt.imshow(x[:, :, 0], cmap='gray')
    plt.axis('off')


if __name__ == "__main__":
    main()
