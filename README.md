# An Image Classification Model Featuring Explainability by Image Morphing

![An example of explanation](docs/fig-explanation-example.png)

## Method

![The process of generating an explanation](docs/fig-algorithm.png)

## Train a Model

```shell
python3 ./src/train.py
```

This script outputs logs and trained models to `out/train/[exec_start_time]`.

## Test a Model

```shell
python3 ./src/main.py path/to/train_log
```

This script outputs explanation images to `out/main/[exec_start_time]`.

## References

- Lopez-Martin M, Carro B, Sanchez-Esguevillas A, Lloret J. Conditional Variational Autoencoder for Prediction and Feature Recovery Applied to Intrusion Detection in IoT. Sensors. 2017; 17(9):1967. https://doi.org/10.3390/s17091967
