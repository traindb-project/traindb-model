import torch
import torch.nn as nn
from torch.nn import Sequential
from torchdiffeq import odeint 

class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input / torch.sqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)

class ODEFuncG(nn.Module):

    def __init__(self, first_layer_dim):
        super(ODEFuncG, self).__init__()

        self.dim = first_layer_dim

        self.layer_start = PixelNorm()
        seq = [ nn.Linear(first_layer_dim + 1, first_layer_dim + 1),
                nn.LeakyReLU(0.2) ]
        seq *= 7
        seq.append(nn.Linear(first_layer_dim + 1, first_layer_dim)) 
        self.layer_t = Sequential(*seq)

        for m in self.layer_t:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, x):

        out = self.layer_start(x)
        tt = torch.ones_like(x[:,[0]]) * t
        out = torch.cat([out, tt],dim = 1)
        out = self.layer_t(out)
        return out

class ODEFuncD(nn.Module):

    def __init__(self, first_layer_dim):
        super(ODEFuncD, self).__init__()
        self.layer_start = nn.Sequential(nn.BatchNorm1d(first_layer_dim),
                                    nn.ReLU())

        self.layer_t = nn.Sequential(nn.Linear(first_layer_dim + 1, first_layer_dim * 2),
                                     nn.BatchNorm1d(first_layer_dim * 2),
                                     nn.ReLU(),
                                     nn.Linear(first_layer_dim * 2, first_layer_dim * 1),
                                     nn.BatchNorm1d(first_layer_dim * 1),
                                     nn.ReLU())
        for m in self.layer_t:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, x):
        out = self.layer_start(x)
        tt = torch.ones_like(x[:,[0]]) * t
        out = torch.cat([out, tt],dim = 1)
        out = self.layer_t(out)
        return out

class ODEBlockG(nn.Module):
    def __init__(self, odefunc):
        super(ODEBlockG, self).__init__()
        self.odefunc = odefunc
        self.integration_time = torch.tensor([0, 1]).float()
        
    def forward(self, x):
        
        self.integration_time = self.integration_time.type_as(x)
        out = odeint(self.odefunc, x, self.integration_time, rtol=1e-3, atol=1e-3) 
        
        return out[1] 

class ODEBlockD(nn.Module):
    def __init__(self, odefunc, num_split):
        super(ODEBlockD, self).__init__()
        self.odefunc = odefunc
        self.num_split = num_split
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x):

        initial_value = x[0]
        integration_time = torch.cat(x[1], dim = 0).to(self.device)
        zero = torch.tensor([0.], requires_grad=False).to(self.device)
        one = torch.tensor([1.], requires_grad=False).to(self.device)

        all_time = torch.cat( [zero, integration_time, one],dim=0).to(self.device)
        self.total_integration_time = [all_time[i:i+2] for i in range(self.num_split)]

        out = [[1, initial_value]]
        for i in range(len(self.total_integration_time)):
            self.integration_time = self.total_integration_time[i].type_as(initial_value)
            out_ode = odeint(self.odefunc, out[i][1], self.integration_time, rtol=1e-3, atol=1e-3)

            out.append(out_ode)
        return [i[1] for i in out]
