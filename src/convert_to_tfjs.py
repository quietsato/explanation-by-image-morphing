import tensorflow as tf
import tensorflowjs as tfjs
import json
import os

from tensorflow.keras import datasets

from util import preprocess_image
from models import IDCVAE

def main(
    SRC_DIR: str,
    DEST_DIR: str
):

    (train_images, train_labels), _ = datasets.mnist.load_data()
    train_images = preprocess_image(train_images)

    model = IDCVAE()
    model.encoder.load_weights(os.path.join(SRC_DIR, "model_encoder.h5"))
    model.decoder.load_weights(os.path.join(SRC_DIR, "model_decoder.h5"))

    if not os.path.exists(DEST_DIR): os.makedirs(DEST_DIR)

    representatives = model.find_representative_points(train_images, train_labels)

    representatives_jsonstr = json.dumps(representatives.numpy().tolist()) 
    with open(os.path.join(DEST_DIR, "representatives.json"), "w+") as f:
        f.write(representatives_jsonstr) 

    tfjs.converters.save_keras_model(model.encoder, os.path.join(DEST_DIR, "model_encoder"))
    tfjs.converters.save_keras_model(model.decoder, os.path.join(DEST_DIR, "model_decoder"))

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()

    parser.add_argument("SRC_DIR", type=str)
    parser.add_argument("DEST_DIR", type=str)

    args = parser.parse_args()

    main(
        args.SRC_DIR,
        args.DEST_DIR
    )
