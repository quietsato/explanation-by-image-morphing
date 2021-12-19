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

    (train_images, train_labels), _ = datasets.mnist.load_data()
    train_images = preprocess_image(train_images)

    model = IDCVAE(latent_dim=16)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-4)
    )

    #
    # Callbacks
    #
    csv_logger = callbacks.CSVLogger(
        os.path.join(OUT_DIR, "log.csv")  # 0 based indexing epochs
    )
    early_stopping = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5
    )

    #
    # Train
    #
    model.fit(
        train_images,
        train_labels,
        validation_split=.1,
        batch_size=batch_size,
        epochs=epochs,
        shuffle=True,
        callbacks=[csv_logger, early_stopping],
        verbose=verbose
    )

    #
    # Save Models
    #
    model.encoder.save(os.path.join(OUT_DIR, "model_encoder.h5"))
    model.decoder.save(os.path.join(OUT_DIR, "model_decoder.h5"))


if __name__ == "__main__":
    main()
