import tensorflow as tf
from tensorflow.keras import datasets

import matplotlib.pyplot as plt
import os

import argparse

if __name__ == "__main__":
    import sys
    sys.path.append(os.path.dirname(__file__))

    from config import *
    from util import create_out_dir, get_time_str, preprocess_image, save_model_summary
    from models import IDCVAE, reconstruction_loss

tf.random.set_seed(42)

IDCVAE_LATENT_DIM = 16
time_str = get_time_str()
OUT_DIR = create_out_dir(f"main/{time_str}")
TEST_IMAGE_MORPH_OUT_DIR = create_out_dir(f"main/{time_str}/test_image")
TEST_MISCLASSIFIED_MORPH_OUT_DIR = create_out_dir(f"main/{time_str}/test_misclassified")


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
    vae = IDCVAE(IDCVAE_LATENT_DIM)
    vae.compile()

    save_idcvae_summary(vae, OUT_DIR)

    if WEIGHT_FILEPATH is not None:
        vae.load_weights(WEIGHT_FILEPATH)

    print("==> Find representative points")
    representative = find_representative_points(
        train_images,
        train_labels,
        vae,
        os.path.join(OUT_DIR, "representative_points.png")
    )

    print("==> Classify images using IDCVAE")
    test_pred = classify(test_images, vae)
    acc = count_successfully_classified(test_pred, test_labels).numpy()
    print(f"Test Accuracy: {acc}/{len(test_labels)}")

    test_images_misclassified = tf.boolean_mask(test_images,
                                                tf.not_equal(test_pred, test_labels))
    test_pred_misclassified = tf.boolean_mask(test_pred,
                                              tf.not_equal(test_pred, test_labels))

    print("==> Create morphing images")

    test_index = list(range(10))
    for i in test_index:
        print(f"test {i:05}")
        generate_morphing_images(
            test_images[i],
            test_pred[i],
            representative,
            vae,
            10,
            os.path.join(TEST_IMAGE_MORPH_OUT_DIR, f"{i:05}")
        )
        decode_image_for_every_label(
            test_images[i],
            test_labels[i],
            vae,
            os.path.join(TEST_IMAGE_MORPH_OUT_DIR, f"{i:05}", "decode_for_every_label.png")
        )

    for i in range(len(test_images_misclassified)):
        print(f"test missclassified {i:05}")
        generate_morphing_images(
            test_images_misclassified[i],
            test_pred_misclassified[i],
            representative,
            vae,
            10,
            os.path.join(TEST_MISCLASSIFIED_MORPH_OUT_DIR, f"{i:05}")
        )
        decode_image_for_every_label(
            test_images_misclassified[i],
            test_pred_misclassified[i],
            vae,
            os.path.join(TEST_MISCLASSIFIED_MORPH_OUT_DIR, f"{i:05}", "decode_for_every_label.png")
        )


def save_idcvae_summary(vae: IDCVAE, out_dir: str):
    save_model_summary(
        vae.E,
        os.path.join(out_dir, "idcvae_encoder_summary.txt")
    )
    save_model_summary(
        vae.D,
        os.path.join(out_dir, "idcvae_decoder_summary.txt")
    )


def decode_image_for_every_label(
        x: tf.Tensor,
        y: tf.Tensor,
        vae: IDCVAE,
        out_path: str):
    decode_cols = (num_classes+1)//2
    figure_cols = 1 + decode_cols

    _, z, _ = vae.encode(tf.expand_dims(x, 0))
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

    plt.savefig(out_path)
    plt.close()


#
# Image classification using IDCVAE
#
def classify(xs: tf.Tensor, vae: IDCVAE):
    dataset = tf.data.Dataset.from_tensor_slices(xs).batch(1024)
    pred = []
    for xs in dataset:
        _, zs, _ = vae.encode(xs)
        loss_batch = []
        for y in range(num_classes):
            ys = tf.tile(tf.constant([y], tf.int32), [len(zs)])
            xs_rec = vae.decode(zs, ys)
            loss = reconstruction_loss(xs, xs_rec, foreach=True)
            loss_batch.append(loss)
        pred_batch = tf.argmin(tf.transpose(tf.convert_to_tensor(loss_batch)), axis=1)
        pred.append(pred_batch)
    return tf.concat(pred, axis=0)


def count_successfully_classified(y, y_pred) -> tf.Tensor:
    return tf.reduce_sum(tf.cast(y_pred == y, tf.int32))


def find_representative_points(xs: tf.Tensor,
                               ys: tf.Tensor,
                               vae: IDCVAE,
                               out_path: str) -> tf.Tensor:
    plt.figure(figsize=((num_classes+1)//2, 2))

    _, zs, _ = vae.encode(xs)
    representative = []
    for i in range(num_classes):
        zs_i = tf.boolean_mask(zs, tf.equal(ys, i))
        z = tf.reduce_mean(zs_i, axis=0, keepdims=True)
        representative.append(z)

        x = vae.decode(z, tf.expand_dims(i, axis=0))
        plt.subplot(2, (num_classes + 1)//2, i + 1)
        plot_single_image(x[0])

    plt.savefig(out_path)
    return tf.concat(representative, axis=0)


#
# Generate morphing images
#
def generate_morphing_images(x: tf.Tensor,
                             y: tf.Tensor,
                             representative: tf.Tensor,
                             vae: IDCVAE,
                             n: int,
                             out_dir: str):
    plt.clf()
    plt.figure(figsize=(1, 1))

    assert n > 0

    def get_image_save_path(i):
        return os.path.join(out_dir, f"{i:03}.png")

    if not os.path.exists(os.path.dirname(get_image_save_path(0))):
        os.makedirs(os.path.dirname(get_image_save_path(0)))

    plot_single_image(x)
    plt.savefig(get_image_save_path(0))

    _, z, _ = vae.encode(tf.expand_dims(x, axis=0))
    diff = representative[y:y+1] - z

    for i in range(n):
        z = z + diff / n
        x_dec = vae.decode(z, tf.expand_dims(y, axis=0))

        plt.clf()
        plt.figure(figsize=(1, 1))
        plot_single_image(x_dec[0])
        plt.savefig(get_image_save_path(i+1))


def plot_single_image(x: tf.Tensor):
    plt.imshow(x[:, :, 0], cmap='gray')
    plt.axis('off')


if __name__ == "__main__":
    main()
