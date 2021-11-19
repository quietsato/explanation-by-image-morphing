import numpy as np
from datetime import datetime, timezone, timedelta

from config import *


def get_time_str():
    JST = timezone(timedelta(hours=+9), 'JST')
    return datetime.now(JST).strftime("%Y%m%d%H%M%S")


def preprocess_image(image: np.ndarray):
    image = image / 255.
    image = image.reshape(-1, image_size, image_size, image_channels)
    return image


def create_log_dir(name: str) -> str:
    return _create_dir(LOG_BASE_DIR, name)


def create_out_dir(name: str) -> str:
    return _create_dir(OUT_BASE_DIR, name)


def _create_dir(base: str, name: str):
    target_dir = os.path.join(base, name)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    return target_dir
