import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import BatchNorm1d, Dropout, LeakyReLU, Linear, Module, ReLU, Sequential
from octgan.odeblocks import ODEFuncG, ODEFuncD, ODEBlockG, ODEBlockD

class Generator(Module):
    def __init__(self, embedding_dim, gen_dims, data_dim):
        super(Generator, self).__init__()
        dim = embedding_dim
        self.ode = ODEBlockG(ODEFuncG(dim))

        seq = []
        for item in list(gen_dims):
            seq += [
                Residual(dim, item)
            ]
            dim += item
        seq.append(Linear(dim, data_dim))
        self.seq = Sequential(*seq)

    def forward(self, input):
        data = self.ode(input)
        data = self.seq(data)
        return data

class Discriminator(Module):
    def __init__(self, input_dim, dis_dims, num_split, pack=1):
        super(Discriminator, self).__init__()
        dim = input_dim * pack
        self.pack = pack
        self.packdim = dim
        self.num_split = num_split

        seq = []
        for item in list(dis_dims):
            seq += [
                Linear(dim, item),
                LeakyReLU(0.2),
                Dropout(0.5)
            ]
            dim = item
        self.seq = Sequential(*seq)
        self.ode = ODEBlockD(ODEFuncD(dim), self.num_split)

        self.traj_dim = dim * (self.num_split + 1)
        self.last1 = nn.Linear(self.traj_dim, self.traj_dim*2)
        self.last3 = nn.Linear(self.traj_dim*2, self.traj_dim)
        self.last4 = nn.Linear(self.traj_dim, 1)

    def forward(self, x):
        value = x[0]
        time = x[1]
        out = self.seq(value.view(-1, self.packdim))
        out1_time = [out, time]
        out = self.ode(out1_time)
        out = torch.cat(out, dim = 1)

        out = F.leaky_relu(self.last1(out))
        out = F.leaky_relu(self.last3(out))
        out = self.last4(out)
        return out


class Residual(Module):
    def __init__(self, i, o):
        super(Residual, self).__init__()
        self.fc = Linear(i, o)
        self.bn = BatchNorm1d(o)
        self.relu = ReLU()

    def forward(self, input):
        out = self.fc(input)
        out = self.bn(out)
        out = self.relu(out)
        return torch.cat([out, input], dim=1)

