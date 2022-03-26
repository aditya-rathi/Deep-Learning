'''
train and test VAE model on airfoils
'''

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np

from dataset import AirfoilDataset
from vae import VAE
from utils import *

def loss_func(recon_x, label, mu, logvar):
    bce = F.mse_loss(recon_x,label,reduction="sum")
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return bce+0.008*kld


def main():
    # check if cuda available
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    #device = 'cpu'

    # define dataset and dataloader
    dataset = AirfoilDataset()
    airfoil_x = dataset.get_x()
    airfoil_dim = airfoil_x.shape[0]
    airfoil_dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # hyperparameters
    latent_dim = 16 # please do not change latent dimension
    lr = 0.003      # learning rate
    num_epochs = 40

    # build the model
    vae = VAE(airfoil_dim=airfoil_dim, latent_dim=latent_dim).to(device)
    print("VAE model:\n", vae)

    # define your loss function here
    # loss = ?

    # define optimizer for discriminator and generator separately
    optim = Adam(vae.parameters(), lr=lr)
    #scheduler = ReduceLROnPlateau(optim)
    
    # train the VAE model
    for epoch in range(num_epochs):
        for n_batch, (local_batch, __) in enumerate(airfoil_dataloader):
            y_real = local_batch.to(device)

            # train VAE
            recon, mu, logvar = vae(y_real)
            loss = loss_func(recon, y_real, mu, logvar)

            # calculate customized VAE loss
            # loss = your_loss_func(...)

            optim.zero_grad()
            loss.backward()
            optim.step()

            # print loss while training
            if (n_batch + 1) % 30 == 0:
                print("Epoch: [{}/{}], Batch: {}, loss: {}".format(
                    epoch, num_epochs, n_batch, loss.item()))
        #scheduler.step(loss.item())

    # test trained VAE model
    num_samples = 100

    # reconstuct airfoils
    real_airfoils = dataset.get_y()[:num_samples]
    recon_airfoils, __, __ = vae(torch.from_numpy(real_airfoils).to(device))
    if 'cuda' in device:
        recon_airfoils = recon_airfoils.detach().cpu().numpy()
    else:
        recon_airfoils = recon_airfoils.detach().numpy()
    
    # randomly synthesize airfoils
    noise = torch.randn((num_samples, latent_dim)).to(device)   # create random noise 
    gen_airfoils = vae.decode(noise)
    if 'cuda' in device:
        gen_airfoils = gen_airfoils.detach().cpu().numpy()
    else:
        gen_airfoils = gen_airfoils.detach().numpy()

    # plot real/reconstructed/synthesized airfoils
    plot_airfoils(airfoil_x, real_airfoils)
    plot_airfoils(airfoil_x, recon_airfoils)
    plot_airfoils(airfoil_x, gen_airfoils)
    

if __name__ == "__main__":
    main()