import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.datasets.mnist import MNIST

import tqdm
from tqdm import tqdm
import os

torch.manual_seed(42)

LOG_DIR = os.path.join(os.path.dirname(__file__), "../log/VAE")
OUT_DIR = os.path.join(os.path.dirname(__file__), "../out/VAE")

restore_weights = False
save_weights_per_epoch = 2

image_size = 28
image_channels = 1
num_classes = 10
latent_dim = 16

validation_split = .2
training_epochs = 5


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    # Logging
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
    log_path = os.path.join(LOG_DIR, f"{get_time_str()}_train.csv")
    csv_logger = CsvLogger(log_path, ['epoch', 'train_loss', 'rec_loss', 'kl_loss', 'time'])

    # Datasets
    train_dataset: MNIST = datasets.MNIST(
        root=os.path.join(os.path.dirname(__file__), '..', 'data'),
        train=True,
        transform=transforms.ToTensor(),
        download=True,
    )

    # Dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=64,
                                  shuffle=True, num_workers=os.cpu_count())

    VAE = build_id_cvae().to(device)

    print(VAE)

    # Loss & Optimiser
    opt = optim.Adam(VAE.parameters(), lr=1e-4)

    # Create a directory to save state dict
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    # Training
    VAE.train(True)
    for epoch in range(1, training_epochs+1):
        print(f"[Epoch {epoch:03}]")

        train_kl_loss = 0
        train_rec_loss = 0
        for i, (batch_x, batch_y) in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            z, z_mean, z_log_var = VAE.encode(batch_x)
            x = VAE.decode(z, batch_y)

            batch_rec_loss = F.mse_loss(x, batch_x, reduction='none').sum(axis=[1, 2, 3]).mean()
            batch_kl_loss = -.5 * (1. + z_log_var - z_mean.square() - z_log_var.exp()).sum(1).mean()
            batch_loss = batch_rec_loss + batch_kl_loss

            opt.zero_grad()
            batch_loss.backward()
            opt.step()

            train_kl_loss += batch_kl_loss.item()
            train_rec_loss += batch_rec_loss.item()

            if (i + 1) % 100 == 0:
                tqdm.write(f"iter {i+1:>4}: loss: {batch_loss.item():4.6f}")

        train_kl_loss /= len(train_dataloader)
        train_rec_loss /= len(train_dataloader)
        train_loss = train_kl_loss + train_rec_loss

        print(f"train_loss: {train_loss:4.6f}",
              f"rec_loss: {train_rec_loss:4.6f}",
              f"kl_loss: {train_kl_loss:4.6f}",
              sep=', ')
        csv_logger.log([epoch, train_loss, train_rec_loss, train_kl_loss, get_time_str()])

        if save_weights_per_epoch > 0 and epoch % save_weights_per_epoch == 0:
            torch.save(VAE.state_dict(), os.path.join(OUT_DIR, f"{epoch:03}_state_dict"))

    VAE.train(False)


if __name__ == "__main__":
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from models.id_cvae import build_id_cvae
    from utils.logger import get_time_str, CsvLogger

    main()
