'''
train and test GAN model on airfoils
'''

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
import numpy as np
from torch.autograd import Variable

from dataset import AirfoilDataset
from gan import Discriminator, Generator
from utils import *
import matplotlib.pyplot as plt




def main():
    # check if cuda available
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    #Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    # define dataset and dataloader
    dataset = AirfoilDataset()
    airfoil_x = dataset.get_x()
    airfoil_dim = airfoil_x.shape[0]
    airfoil_dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # hyperparameters
    latent_dim = 16 # please do not change latent dimension
    lr_dis = 0.001 # discriminator learning rate
    lr_gen = 0.0002 # generator learning rate
    num_epochs = 100
    
    # build the model
    dis = Discriminator(input_dim=airfoil_dim).to(device)
    gen = Generator(latent_dim=latent_dim, airfoil_dim=airfoil_dim).to(device)
    print("Distrminator model:\n", dis)
    print("Generator model:\n", gen)

    # define your GAN loss function here
    # you may need to define your own GAN loss function/class
    # loss = ?
    loss = nn.BCELoss()
    loss_epoch_gen = []
    loss_epoch_dis = []

    # define optimizer for discriminator and generator separately
    optim_dis = Adam(dis.parameters(), lr=lr_dis)
    optim_gen = Adam(gen.parameters(), lr=lr_gen)
    
    # train the GAN model
    for epoch in range(num_epochs):
        for n_batch, (local_batch, __) in enumerate(airfoil_dataloader):
            y_real = local_batch.to(device)

            #Ground Truths
            valid = torch.Tensor(y_real.size(0),1).fill_(1.0).to(device)
            fake = torch.Tensor(y_real.size(0),1).fill_(0.0).to(device)

            #Sample noise
            z = torch.Tensor(np.random.normal(0,1,(y_real.shape[0],latent_dim))).to(device)

            # Generate a batch of images
            gen_imgs = gen(z)

            # train discriminator
            real_loss = loss(dis(y_real), valid)
            fake_loss = loss(dis(gen_imgs.detach()), fake)
            loss_dis = (real_loss + fake_loss) / 2
            # calculate customized GAN loss for discriminator
            # enc_loss = loss(...)

            optim_dis.zero_grad()
            loss_dis.backward()
            optim_dis.step()

            # train generator

            # calculate customized GAN loss for generator
            # enc_loss = loss(...)
            loss_gen = loss(dis(gen_imgs),valid)

            optim_gen.zero_grad()
            loss_gen.backward()
            optim_gen.step()

            # print loss while training
            if (n_batch + 1) % 30 == 0:
                print("Epoch: [{}/{}], Batch: {}, Discriminator loss: {:.3f}, Generator loss: {:.3f}".format(
                    epoch, num_epochs, n_batch, loss_dis.item(), loss_gen.item()))
        loss_epoch_gen.append(loss_gen.item())
        loss_epoch_dis.append(loss_dis.item())

    # test trained GAN model
    num_samples = 100
    # create random noise 
    noise = torch.randn((num_samples, latent_dim)).to(device)
    # generate airfoils
    gen_airfoils = gen(noise)
    if 'cuda' in device:
        gen_airfoils = gen_airfoils.detach().cpu().numpy()
    else:
        gen_airfoils = gen_airfoils.detach().numpy()

    # plot generated airfoils
    plot_airfoils(airfoil_x, gen_airfoils)

    #Plot Loss
    plt.plot(range(len(loss_epoch_dis)),loss_epoch_dis,'r',range(len(loss_epoch_dis)),loss_epoch_gen,'g')
    plt.title("Loss vs Epochs")
    plt.legend(['Discriminator','Generator'])
    plt.show()

    #Save model
    torch.save({
            'generator_state_dict': gen.state_dict(),
            'discriminator_state_dict': dis.state_dict(),
            'generator_adam_state_dict': optim_gen.state_dict(),
            'discriminator_adam_state_dict': optim_dis.state_dict(),
            }, 'p1_gan_model.pth')



if __name__ == "__main__":
    main()

