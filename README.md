# Explain MNIST Classification Using ID-CVAE

卒研実装コード

## Training

### Command

```shell
python3 ./src/train.py
```

### Outputs

```
./out/
└── train
    └── 2021-11-22-09-29-57
        ├── checkpoints
        │   └── 001.h5
        └── log.csv
```

## Generate figures

### Command

```
python3 ./src/main.py
```

### Outputs

```
./out/
└── main
    └── 2021-11-22-14-16-36
        ├── idcvae_decoder_summary.txt
        ├── idcvae_encoder_summary.txt
        ├── representative_points.png
        ├── test_image
        │   ├── 00000
        │   │   ├── 000.png
        │   │   ├── ...
        │   │   ├── 010.png
        │   │   └── decode_for_every_label.png
        │   ├── 00001
        │   ├── 00002
        │   └── ...
        └── test_misclassified
            ├── 00000
            │   ├── 000.png
            │   ├── ...
            │   ├── 010.png
            │   └── decode_for_every_label.png
            ├── 00001
            ├── 00002
            └── ...
```


## References
