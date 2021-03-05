import torch
import numpy as np 
from basics import Method
from model import MLP, soft_update, hard_update
from torch.optim import Adam

class wgan(Method):
    def __init__(self, input_shape, vol):
        super(wgan, self).__init__()
        self.input_shape = input_shape
        self.vol = vol
        self.clamp_max = 0.01
        self.losses = np.zeros((self.max_iter,))
        self.vals = np.zeros((self.max_iter,))

        self.device = torch.device("cuda")
        self.disc = MLP(input_shape, hidden_dim=64, num_outputs=1).to(device=self.device)
        self.disc_optim = Adam(self.disc.parameters(), lr=0.002)

    def update_parameters(self, As, Bs, shuffle=True):
        if shuffle:
            np.random.shuffle(As)
            np.random.shuffle(Bs)
        As = torch.FloatTensor(As).to(self.device)
        Bs = torch.FloatTensor(Bs).to(self.device)
        VAs = self.disc(As)
        VBs = self.disc(Bs)

        loss1 = VAs.mean()
        loss2 = -VBs.mean()
        self.disc_optim.zero_grad()
        loss1.backward()
        loss2.backward()
        self.disc_optim.step()
        for p in self.disc.parameters():
            p.data.clamp_(-self.clamp_max, self.clamp_max)

        return (loss1 + loss2).item()

    def estimate(self, As, Bs):
        As = torch.FloatTensor(As).to(self.device)
        Bs = torch.FloatTensor(Bs).to(self.device)
        VAs = self.disc(As)
        VBs = self.disc(Bs)
        rv = torch.abs(VAs.mean() - VBs.mean())
        return rv.squeeze().detach().cpu().numpy()

    def train(self, As, Bs):
        for i in range(self.max_iter):
            loss = self.update_parameters(As, Bs)
            self.losses[i] = loss       
            self.vals[i] = self.estimate(As, Bs)         

class bgtf(Method):
    def __init__(self, input_shape, vol):
        super(bgtf, self).__init__()
        self.input_shape = input_shape
        self.vol = vol
        self.gamma = 1.0 # control the effect of softmax
        self.losses = np.zeros((self.max_iter,))
        self.vals = np.zeros((self.max_iter,))

        self.device = torch.device("cuda")
        self.mu = MLP(input_shape, hidden_dim=64, num_outputs=1).to(device=self.device)    
        self.nu = MLP(input_shape, hidden_dim=64, num_outputs=1).to(device=self.device)    
        self.tf_optim = Adam(list(self.mu.parameters()) + list(self.nu.parameters()), lr=0.002)   

    def update_parameters(self, As, Bs, shuffle=True):
        if shuffle:
            np.random.shuffle(As)
            np.random.shuffle(Bs)        
        As = torch.FloatTensor(As).to(self.device)
        Bs = torch.FloatTensor(Bs).to(self.device)
        VAs = self.mu(As)
        VBs = self.nu(Bs)

        cost = torch.norm(As - Bs, p=2, dim=-1)
        damping = VAs.squeeze() - VBs.squeeze() - cost
        damping = self.gamma * torch.exp(damping / self.gamma)
        loss = -VAs.mean() + VBs.mean() + damping.mean()

        self.tf_optim.zero_grad()
        loss.backward()
        self.tf_optim.step()

        return loss.item()

    def estimate(self, As, Bs):
        As = torch.FloatTensor(As).to(self.device)
        Bs = torch.FloatTensor(Bs).to(self.device)
        VAs = self.mu(As)
        VBs = self.nu(Bs)
        rv = torch.abs(VAs.mean() - VBs.mean())
        return rv.squeeze().detach().cpu().numpy()

    def train(self, As, Bs):
        for i in range(self.max_iter):
            loss = self.update_parameters(As, Bs)
            self.losses[i] = loss       
            self.vals[i] = self.estimate(As, Bs)  