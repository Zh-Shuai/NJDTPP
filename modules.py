import torch.nn as nn


# compute the running average
class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.vals = []
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.vals = []
        self.val = None
        self.avg = 0

    def update(self, val):
        self.vals.append(val)
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val


# multi-layer perceptron
class MLP(nn.Module):

    def __init__(self, dim_in, dim_out, dim_hidden=32, num_hidden=2, sigma=0.01, activation=nn.Tanh()):
        super(MLP, self).__init__()

        if num_hidden == 0:
            self.linears = nn.ModuleList([nn.Linear(dim_in, dim_out)])
        elif num_hidden >= 1:
            self.linears = nn.ModuleList()
            self.linears.append(nn.Linear(dim_in, dim_hidden))
            self.linears.extend([nn.Linear(dim_hidden, dim_hidden) for _ in range(num_hidden-1)])
            self.linears.append(nn.Linear(dim_hidden, dim_out))
        else:
            raise Exception('number of hidden layers must be positive')

        for m in self.linears:
            nn.init.normal_(m.weight, mean=0, std=sigma)
            nn.init.uniform_(m.bias, a=-sigma, b=sigma)

        self.activation = activation

    def forward(self, x):
        for m in self.linears[:-1]:
            x = self.activation(m(x))

        return self.linears[-1](x)

    
class NJDSDE(nn.Module):
    """
    Neural Jump-Diffusion SDE
    """
    def __init__(self, device, dim_eta, dim_hidden=32, num_hidden=2, batch_size=30, sigma=0.01, activation=nn.Tanh()):
        super(NJDSDE, self).__init__()

        self.dim_eta = dim_eta
        self.device = device
        self.batch_size = batch_size

        self.F = MLP(dim_eta, dim_eta, dim_hidden, num_hidden, sigma, activation)
        self.G = MLP(dim_eta, dim_eta, dim_hidden, num_hidden, sigma, activation)
        self.H = MLP(dim_eta, dim_eta * dim_eta, dim_hidden, num_hidden, sigma, activation)

    # drift net
    def f(self, y):
        return self.F(y).view(self.batch_size, self.dim_eta)
        
    # diffusion net
    def g(self, y):
        return self.G(y).view(self.batch_size, self.dim_eta)
    
    # jump net
    def h(self, y):
        return self.H(y).view(self.batch_size, self.dim_eta, self.dim_eta)