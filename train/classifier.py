import torch
from torch import Tensor, nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.datasets.mnist import MNIST

import matplotlib.pyplot as plt
import os

torch.manual_seed(42)

LOG_DIR = os.path.join(os.path.dirname(__file__), "../log")
OUT_DIR = os.path.join(os.path.dirname(__file__), "../out")

restore_weights = False
save_weights_per_epoch = 2

image_size = 28
image_channels = 1
num_classes = 10

validation_split = .2
training_epochs = 10


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    # Logging
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
    log_path = os.path.join(LOG_DIR, f"{get_time_str()}_C_train.csv")
    csv_logger = CsvLogger(log_path, ['epoch', 'train_loss', 'val_loss', 'val_acc', 'time'])

    # Datasets
    train_val_dataset: MNIST = datasets.MNIST(
        root=os.path.join(os.path.dirname(__file__), '..', 'data'),
        train=True,
        transform=transforms.ToTensor(),
        download=True,
    )
    len_validation_dataset = int(len(train_val_dataset) * validation_split)
    len_train_dataset = len(train_val_dataset) - len_validation_dataset
    train_dataset, validation_dataset = random_split(train_val_dataset,
                                                     [len_train_dataset, len_validation_dataset])

    test_dataset: MNIST = datasets.MNIST(
        root=os.path.join(os.path.dirname(__file__), '..', 'data'),
        train=False,
        transform=transforms.ToTensor(),
        download=True,
    )

    # Dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=128,
                                  shuffle=True, num_workers=os.cpu_count())
    validation_dataloader = DataLoader(validation_dataset, batch_size=128,
                                       shuffle=False, num_workers=os.cpu_count())
    test_dataloader = DataLoader(test_dataset, batch_size=128,
                                 shuffle=False, num_workers=os.cpu_count())

    C = Classifier(image_size,
                   image_channels,
                   num_classes,
                   conv_out_channels=[32, 64],
                   conv_kernel_size=[3, 3],
                   pool_kernel_size=[2, 2]).to(device)

    print(C)

    # Loss & Optimiser
    loss = nn.CrossEntropyLoss()
    opt = optim.Adam(C.parameters(), lr=1e-3)

    # Create a directory to save state dict
    C_state_dict_dir = os.path.join(OUT_DIR, "C")
    if not os.path.exists(C_state_dict_dir):
        os.makedirs(C_state_dict_dir)

    # Training
    C.train(True)
    for epoch in range(1, training_epochs+1):
        print(f"[Epoch {epoch:03}]")

        train_loss = 0
        for (batch_x, batch_y) in train_dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            opt.zero_grad()

            y: Tensor = C(batch_x)
            batch_loss = loss(y, batch_y)
            batch_loss.backward()
            opt.step()

            train_loss += batch_loss.item()
        
        train_loss /= len(train_dataloader)

        # Validate model
        val_loss = 0
        correct_count = 0
        with torch.no_grad():
            for (val_x, val_y) in validation_dataloader:
                val_x, val_y = val_x.to(device), val_y.to(device)

                y = C(val_x)
                val_loss += loss(y, val_y).item()
                is_correct: Tensor = y.argmax(1) == val_y
                correct_count += is_correct.int().sum().item()
        
        val_loss /= len(validation_dataloader)
        val_acc = correct_count / len(validation_dataset)

        print(f"train_loss: {train_loss:3.6f}, val_loss: {val_loss:3.6f}, val_acc: {val_acc:.6f}")
        csv_logger.log([epoch, train_loss, val_loss, val_acc, get_time_str()])

        if save_weights_per_epoch > 0 and epoch % save_weights_per_epoch == 0:
            torch.save(C.state_dict(), os.path.join(C_state_dict_dir, f"C_{epoch:03}_state_dict"))

    C.train(False)


if __name__ == "__main__":
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from models.classifier import Classifier
    from utils.csv_logger import get_time_str, CsvLogger

    main()
