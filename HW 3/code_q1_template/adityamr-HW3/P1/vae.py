import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Encoder, self).__init__()
        # build your model here
        # your model should output a predicted mean and a predicted std of the encoding
        # both should be of dim (batch_size, latent_dim)
        mid_dim = int((input_dim+latent_dim)/2)
        quarter_dim = int((latent_dim +mid_dim)/2)
        three_quarter_dim = int((input_dim+ mid_dim)/2)

        self.fc1 = nn.Linear(input_dim,three_quarter_dim)
        self.fc2 = nn.Linear(three_quarter_dim,mid_dim)
        self.fc3 = nn.Linear(mid_dim,quarter_dim)
        #self.fc4 = nn.Linear(mid_dim,mid_dim)
        self.fc_mean = nn.Linear(quarter_dim,latent_dim)
        self.fc_logvar = nn.Linear(quarter_dim,latent_dim)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
    
    def forward(self, x):
        # define your feedforward pass
        x = self.fc1(x)
        x = self.tanh(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        # x = self.fc4(x)
        # x = self.relu(x)
        return self.fc_mean(x),self.fc_logvar(x)


class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(Decoder, self).__init__()
        # build your model here
        # your output should be of dim (batch_size, output_dim)
        # you can use tanh() as the activation for the last layer
        # since y coord of airfoils range from -1 to 1
        mid_dim = int((latent_dim+output_dim)/2)
        quarter_dim = int((latent_dim +mid_dim)/2)
        three_quarter_dim = int((output_dim+ mid_dim)/2)

        self.fc1 = nn.Linear(latent_dim,quarter_dim)
        self.fc12 = nn.Linear(quarter_dim,mid_dim)
        self.fc23 = nn.Linear(mid_dim,three_quarter_dim)
        self.fc2 = nn.Linear(three_quarter_dim,output_dim)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc12(x)
        x = self.relu(x)
        x = self.fc23(x)
        x = self.relu(x)
        x = self.fc2(x)
        return self.tanh(x)


class VAE(nn.Module):
    def __init__(self, airfoil_dim, latent_dim):
        super(VAE, self).__init__()
        self.enc = Encoder(airfoil_dim, latent_dim)
        self.dec = Decoder(latent_dim, airfoil_dim)
    
    def reparameterize(self,mu,logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu+eps*logvar
    
    def forward(self, x):
        mean,logvar = self.enc(x)
        z = self.reparameterize(mean,logvar)
        return self.decode(z), mean, logvar

    def decode(self, z):
        # given random noise z, generate airfoils
        return self.dec(z)

