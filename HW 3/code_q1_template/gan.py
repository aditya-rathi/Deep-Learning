import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        # build your model here
        # your output should be of dim (batch_size, 1)
        # since discriminator is a binary classifier
        self.model = nn.Sequential(
            nn.Linear(input_dim,int(3*input_dim/4)),
            nn.ReLU(),
            nn.Linear(int(3*input_dim/4),int(input_dim/2)),
            nn.ReLU(),
            nn.Linear(int(input_dim/2),int(input_dim/4)),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(int(input_dim/4),1),
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        # define your feedforward pass
        x = self.model(x)
        return x


class Generator(nn.Module):
    def __init__(self, latent_dim, airfoil_dim):
        super(Generator, self).__init__()
        # build your model here
        # your output should be of dim (batch_size, airfoil_dim)
        # you can use tanh() as the activation for the last layer
        # since y coord of airfoils range from -1 to 1
        mid_dim = int((latent_dim+airfoil_dim)/2)
        quarter_dim = int((latent_dim +mid_dim)/2)
        three_quarter_dim = int((airfoil_dim+ mid_dim)/2)
        self.fc1 = nn.Linear(latent_dim,quarter_dim)
        self.fc2 = nn.Linear(quarter_dim,mid_dim)
        self.fc3 = nn.Linear(mid_dim,three_quarter_dim)
        self.fc4 = nn.Linear(three_quarter_dim,airfoil_dim)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    
    def forward(self, x):
        # define your feedforward pass
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        return self.tanh(x)

