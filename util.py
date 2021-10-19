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
    log_dir = os.path.join(LOG_BASE_DIR, name)
    os.makedirs(log_dir)
    return log_dir


def create_out_dir(name: str) -> str:
    out_dir = os.path.join(OUT_BASE_DIR, name)
    os.makedirs(out_dir)
    return out_dir
