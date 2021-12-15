import tensorflow as tf
from tensorflow.keras import datasets, optimizers, callbacks
import os

if __name__ == "__main__":
    import sys
    sys.path.append(os.path.dirname(__file__))

    from config import *
    from models import IDCVAE
    from util import get_time_str, preprocess_image, create_out_dir


os.environ['TF_DETERMINISTIC_OPS'] = '1'
tf.random.set_seed(42)

epochs = 3
batch_size = 128

verbose = 2


def main():
    time_str = get_time_str()
    OUT_DIR = create_out_dir(f"train/{time_str}")
    CHECKPOINT_DIR = create_out_dir(f"train/{time_str}/checkpoints")

    (dataset_images, dataset_labels), _ = datasets.mnist.load_data()
    dataset_images = preprocess_image(dataset_images)

    train_len = len(dataset_images) // 10 * 9
    train_images, valid_images = dataset_images[:train_len], dataset_images[train_len:]
    train_labels, valid_labels = dataset_labels[:train_len], dataset_labels[train_len:]
    assert len(train_images) + len(valid_labels) == len(dataset_images)
    assert len(train_labels) + len(valid_labels) == len(dataset_labels)

    VAE = IDCVAE(latent_dim=16)
    VAE.compile(
        optimizer=optimizers.Adam(learning_rate=1e-4)
    )

    #
    # Callbacks
    #
    csv_logger = callbacks.CSVLogger(
        os.path.join(OUT_DIR, "log.csv") # 0 based indexing epochs
    )
    early_stopping = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5
    )
    model_checkpoint = callbacks.ModelCheckpoint(
        os.path.join(
            CHECKPOINT_DIR,
            "{epoch:03d}.h5" # 1 based indexing epochs
        ),
        monitor='val_loss',
        save_weights_only=True,
    )

    #
    # Train
    #
    VAE.fit(
        train_images,
        train_labels,
        validation_split=.1,
        batch_size=batch_size,
        epochs=epochs,
        shuffle=True,
        callbacks=[csv_logger, early_stopping, model_checkpoint],
        verbose=verbose
    )


if __name__ == "__main__":
    main()
