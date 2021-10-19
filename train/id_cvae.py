from tensorflow.keras import datasets, optimizers, callbacks
import os

epochs = 3
batch_size = 128


def main():
    LOG_DIR = create_log_dir("VAE")
    OUT_DIR = create_out_dir("VAE")

    time_str = get_time_str()
    (train_images, train_labels), _ = datasets.mnist.load_data()
    train_images = preprocess_image(train_images)

    VAE = IDCVAE()
    VAE.compile(
        optimizer=optimizers.Adam(learning_rate=1e-4)
    )

    # Callbacks
    csv_logger = callbacks.CSVLogger(
        os.path.join(LOG_DIR, f"{time_str}.csv")
    )
    early_stopping = callbacks.EarlyStopping(
        monitor='loss',
        patience=5
    )
    model_checkpoint = callbacks.ModelCheckpoint(
        os.path.join(OUT_DIR,
                     time_str + "_weights_{epoch:03d}_{loss:04.3f}_{reconstruction_loss:04.3f}_{kl_loss:03.3f}.hdf5"),
        monitor='loss',
        save_weights_only=True,
    )

    # Train
    VAE.fit(
        train_images,
        train_labels,
        batch_size=batch_size,
        epochs=epochs,
        shuffle=True,
        callbacks=[csv_logger, early_stopping, model_checkpoint]
    )


if __name__ == "__main__":
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

    from config import *
    from models import IDCVAE
    from util import get_time_str, preprocess_image, create_log_dir, create_out_dir

    main()
