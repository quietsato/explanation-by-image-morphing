import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

import os

def main(
    CSV_FILE_PATH: str,
    OUTPUT_FILE_PATH: str
):
    sns.set()

    df = pd.read_csv(CSV_FILE_PATH)

    plt.plot(df['loss'], label="train_loss")
    plt.plot(df['val_loss'], label="val_loss")

    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.legend()

    d = os.path.dirname(OUTPUT_FILE_PATH)
    if not os.path.exists(d):
        os.makedirs(d)

    plt.tight_layout()
    plt.savefig(OUTPUT_FILE_PATH, dpi=300)


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()

    parser.add_argument("CSV_FILE_PATH", type=str)
    parser.add_argument("OUTPUT_FILE_PATH", type=str)

    args = parser.parse_args()

    main(
        args.CSV_FILE_PATH,
        args.OUTPUT_FILE_PATH
    )
