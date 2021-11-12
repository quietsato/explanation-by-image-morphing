import tensorflow as tf
from tensorflow.keras import datasets, Model

import matplotlib.pyplot as plt
import os

if __name__ == "__main__":
    import sys
    sys.path.append(os.path.dirname(__file__))

    from config import *
    from util import create_out_dir, get_time_str, preprocess_image
    from models import build_classifier, IDCVAE, reconstruction_loss


tf.random.set_seed(42)

C_WEIGHT_FILENAME = None
VAE_WEIGHT_FILENAME = "VAE.h5"

OUT_DIR = create_out_dir(f"main/{get_time_str()}")


def main():
    print("==> Setup dataset")
    _, (test_images, test_labels) = datasets.mnist.load_data()
    test_images = preprocess_image(test_images)

    print("==> Setup model")
    classifier = build_classifier()

    vae = IDCVAE()
    vae.compile()

    if C_WEIGHT_FILENAME is not None:
        weight_path = os.path.join(OUT_BASE_DIR, "C", C_WEIGHT_FILENAME)
        classifier.load_weights(weight_path)

    if VAE_WEIGHT_FILENAME is not None:
        weight_path = os.path.join(OUT_BASE_DIR, "VAE", VAE_WEIGHT_FILENAME)
        vae.load_weights(weight_path)

    # IDCVAE Test
    print("==> IDCVAE Test")
    idcvae_encode_decode_test(test_images[:10], test_labels[:10], vae)
    idcvae_decode_single_image_with_every_label(test_images[0], test_labels[0], vae, "test0")

    # Method 1. Using both of an ID-CVAE and a classifier
    # test_pred_classifier = tf.argmax(classifier.predict(test_images), axis=1)
    # test_images_misclassified = tf.boolean_mask(test_images,
    #                                             tf.not_equal(test_pred_classifier, test_labels))
    # create_morphing_images_classifier_and_idcvae(
    #     test_images_misclassified[0], classifier, vae, 1e-2, 10, "test0")

    # Method 2. Using only an ID-CVAE
    print("==> Classify images using IDCVAE")
    test_pred_idcvae = classify_using_idcvae(test_images, vae)
    acc = count_successfully_classified(test_pred_idcvae, test_labels).numpy()
    print(f"Test Accuracy: {acc}/{len(test_labels)}")

    test_images_idcvae_misclassified = tf.boolean_mask(test_images,
                                                       tf.not_equal(test_pred_idcvae, test_labels))
    test_pred_idcvae_misclassified = tf.boolean_mask(test_pred_idcvae,
                                                     tf.not_equal(test_pred_idcvae, test_labels))

    print("==> Find representing points")
    representative = find_representative_points(test_images, test_pred_idcvae, vae)

    print("==> Create morphing images")

    print("test 00000")
    create_morphing_images_idcvae_only(
        test_images[0], test_pred_idcvae[0], representative, vae, 10, "test_00000")

    print("test 00008")
    create_morphing_images_idcvae_only(
        test_images[8], test_pred_idcvae[8], representative, vae, 10, "test_00008")

    for i in range(len(test_images_idcvae_misclassified)):
        print(f"test missclassified {i:05}")
        create_morphing_images_idcvae_only(
            test_images_idcvae_misclassified[i],
            test_pred_idcvae_misclassified[i],
            representative,
            vae,
            10,
            f"test_misclassified_{i:05}"
        )


def idcvae_encode_decode_test(xs: tf.Tensor, ys: tf.Tensor, vae: IDCVAE):
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

    plt.savefig(os.path.join(OUT_DIR, "encode_decode.png"))


def idcvae_decode_single_image_with_every_label(x: tf.Tensor, y: tf.Tensor, vae: IDCVAE, image_id: str):
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

    plt.savefig(os.path.join(OUT_DIR, f"{image_id}_decode_single_image_with_every_label.png"))
    plt.close()


#
# Image classification using IDCVAE
#
def classify_using_idcvae(xs: tf.Tensor, vae: IDCVAE):
    dataset = tf.data.Dataset.from_tensor_slices(xs).batch(1024)
    pred = []
    for xs in dataset:
        zs, _, _ = vae.encode(xs)
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


def find_representative_points(xs: tf.Tensor, ys: tf.Tensor, vae: IDCVAE) -> tf.Tensor:
    plt.figure(figsize=((num_classes+1)//2, 2))

    zs, _, _ = vae.encode(xs)
    representative = []
    for i in range(num_classes):
        zs_i = tf.boolean_mask(zs, tf.equal(ys, i))
        z = tf.reduce_mean(zs_i, axis=0, keepdims=True)
        representative.append(z)

        x = vae.decode(z, tf.expand_dims(i, axis=0))
        plt.subplot(2, (num_classes + 1)//2, i + 1)
        plot_single_image(x[0])

    plt.savefig(os.path.join(OUT_DIR, f"idcvae_representative_points.png"))
    return tf.concat(representative, axis=0)


#
# Create morphing images
#
# def create_morphing_images_classifier_and_idcvae(x: tf.Tensor,
#                                                  classifier: Model,
#                                                  vae: IDCVAE,
#                                                  epsilon: int,
#                                                  n: int,
#                                                  image_id: str):
#     assert n > 0
#     assert epsilon > 0

#     plt.clf()
#     plt.figure(figsize=(1, 1))

#     def get_image_save_path(i):
#         return os.path.join(OUT_DIR, f"{image_id}_morphing_classifier_and_idcvae/{i:03}.png")

#     def create_z_mat(z: tf.Tensor) -> tf.Tensor:
#         move_plus = tf.math.multiply(tf.tile(z, [latent_dim, 1]), tf.linalg.eye(latent_dim))
#         move_minus = tf.math.multiply(tf.tile(z, [latent_dim, 1]), -tf.linalg.eye(latent_dim))
#         return tf.concat([move_plus, move_minus], axis=0)

#     if not os.path.exists(os.path.dirname(get_image_save_path(0))):
#         os.makedirs(os.path.dirname(get_image_save_path(0)))

#     plot_single_image(x)
#     plt.savefig(get_image_save_path(0))

#     c_pred = classifier(tf.expand_dims(x, axis=0))
#     label = tf.argmax(c_pred, axis=1)[0]

#     for i in range(n):
#         z, _, _ = vae.encode(tf.expand_dims(x, axis=0))
#         z_mat = create_z_mat(z)
#         x_dec = vae.decode(z_mat, label * tf.ones([len(z_mat)], dtype=tf.int64))
#         next_c_pred = classifier(x_dec)[:, label]

#         next_i = tf.argmax(next_c_pred)

#         plt.figure(figsize=(1, 1))
#         plot_single_image(x_dec[next_i])
#         plt.savefig(get_image_save_path(i+1))

#         z = z_mat[next_i:next_i+1]


def create_morphing_images_idcvae_only(x: tf.Tensor,
                                       y: tf.Tensor,
                                       representative: tf.Tensor,
                                       vae: IDCVAE,
                                       n: int,
                                       image_id: str):
    plt.clf()
    plt.figure(figsize=(1, 1))

    assert n > 0

    def get_image_save_path(i):
        return os.path.join(OUT_DIR, f"{image_id}_morphing_idcvae_only/{i:03}.png")

    if not os.path.exists(os.path.dirname(get_image_save_path(0))):
        os.makedirs(os.path.dirname(get_image_save_path(0)))

    plot_single_image(x)
    plt.savefig(get_image_save_path(0))

    z, _, _ = vae.encode(tf.expand_dims(x, axis=0))
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
