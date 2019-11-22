import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):

    def __init__(self, num_input, z_dim=32, keep_prob=0.9):
        super(Encoder, self).__init__()
        n_hidden=512
        self.num_input = num_input
        self.n_hidden = n_hidden
        self.z_dim = z_dim
        self.keep_prob = keep_prob

        self.net = nn.Sequential(
            nn.Linear(num_input, n_hidden),
            nn.ReLU(),
            #nn.Dropout(1-keep_prob),

            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            #nn.Dropout(1-keep_prob),
            
            nn.Linear(n_hidden, 128),
            nn.ReLU(),
            #nn.Dropout(1-keep_prob),

            nn.Linear(128, z_dim*2)

        )
        
    def forward(self, x):
        mu_sigma = self.net(x)

        mean = mu_sigma[:, :self.z_dim]
        # The standard deviation must be positive. Parametrize with a softplus and
        # add a small epsilon for numerical stability
        stddev = 1e-6 + F.softplus(mu_sigma[:, self.z_dim:])
        #stddev = 1e-6 + mu_sigma[:, self.z_dim:]
        return mean, stddev

    
class Decoder(nn.Module):

    def __init__(self,  n_output, dim_z=32, keep_prob=0.9):
        super(Decoder, self).__init__()
        n_hidden=256
        self.dim_z = dim_z
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.keep_prob = keep_prob
        self.net = nn.Sequential(
            nn.Linear(dim_z, 128),
            nn.ReLU(),
            #nn.Dropout(1-keep_prob),
            
            nn.Linear(128, 512),
            nn.ReLU(),
            #nn.Dropout(1-keep_prob),

            nn.Linear(512, 512),
            nn.ReLU(),
            #nn.Dropout(1-keep_prob),

            nn.Linear(512, n_output)
            #nn.Sigmoid()
        )

    def forward(self, z):
        return self.net(z)
    
    

class AVE(nn.Module):
    def __init__(self,  num_input, n_output, dim_z=32,keep_prob=0.9):
        super(AVE, self).__init__()
        self.encoder=Encoder(num_input=num_input, z_dim=dim_z, keep_prob=keep_prob)
        self.decoder=Decoder(n_output=n_output, dim_z=dim_z, keep_prob=keep_prob)
        
    def forward(self, x):
        mu, sigma = self.encoder(x)
        z = mu + sigma * torch.randn_like(mu)
        y = self.decoder(z)
        #y = torch.clamp(y, 1e-8, 1 - 1e-8)
        return y,mu,sigma,z

