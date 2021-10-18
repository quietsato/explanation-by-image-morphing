import os
import csv
from typing import List
from datetime import datetime, timezone, timedelta


def get_time_str():
    JST = timezone(timedelta(hours=+9), 'JST')
    return datetime.now(JST).strftime(f"%Y%m%d_%H%M%S")


class CsvLogger():
    def __init__(self, path: str, headers: List[int]) -> None:
        self.path = path
        self.headers = headers

        if not os.path.exists(os.path.dirname(self.path)):
            os.makedirs(os.path.dirname(self.path))

        with open(self.path, 'a+', newline='', encoding='utf-8') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(self.headers)

    def log(self, data: List):
        with open(self.path, 'a', newline='', encoding='utf-8') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(data)
