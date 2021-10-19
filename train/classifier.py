import tensorflow as tf
from tensorflow.keras import datasets, optimizers, callbacks, Model, losses
import os

epochs = 3
batch_size = 256

def main():
    time_str = get_time_str()
    (train_images, train_labels), _ = datasets.mnist.load_data()
    train_images = preprocess_image(train_images)
    train_labels = tf.one_hot(train_labels, num_classes)


    C: Model = build_classifier()
    C.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss=losses.CategoricalCrossentropy(from_logits=False),
        metrics=['acc']
    )

    # Callbacks
    csv_logger = callbacks.CSVLogger(
        os.path.join(LOG_DIR, "C", f"{time_str}.csv")
    )
    early_stopping = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3
    )
    model_checkpoint = callbacks.ModelCheckpoint(
        os.path.join(OUT_DIR,
                     "C",
                     time_str + "_weights_{epoch:03d}_{loss:02.3f}_{acc:02.3f}_{val_loss:02.3f}_{val_acc:02.3f}.hdf5"),
        monitor='val_loss',
        save_weights_only=True,
    )

    # Train
    C.fit(
        train_images,
        train_labels,
        batch_size=batch_size,
        epochs=epochs,
        shuffle=True,
        validation_split=.2,
        callbacks=[csv_logger, early_stopping, model_checkpoint]
    )

    C.save_weights(
        os.path.join(OUT_DIR, "C", f"{time_str}_complete")
    )


if __name__ == "__main__":
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

    from config import *
    from models import build_classifier
    from util import get_time_str, preprocess_image

    main()
