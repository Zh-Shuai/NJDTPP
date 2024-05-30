import sys
sys.path.append('..')
import signal
import argparse
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from modules import RunningAverageMeter, NJDSDE
from utils import create_outpath, forward_pass, recovery_intensity, plot_poisson_recovery_intensity

signal.signal(signal.SIGINT, lambda sig, frame: sys.exit(0))

parser = argparse.ArgumentParser('poisson')
parser.add_argument('--niters', type=int, default=300)
parser.add_argument('--dataset', type=str, default='poisson')
parser.add_argument('--paramr', type=str, default='params.pth')
parser.add_argument('--paramw', type=str, default='params.pth')
parser.add_argument('--batch_size', type=int, default=30)
parser.add_argument('--nsave', type=int, default=50)
parser.set_defaults(restart=False, seed=True, debug=False)
parser.add_argument('--restart', dest='restart', action='store_true')
parser.add_argument('--seed', dest='seed', action='store_true')
parser.add_argument('--debug', dest='debug', action='store_true')
parser.add_argument('--num_divide', type=int, default=10, help='number of divided points in the time interval')
args = parser.parse_args()

outpath = create_outpath('poisson')


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # write all parameters to output file
    print(args, flush=True)

    # fix seeding for randomness
    if args.seed:
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)

    with open('../data/poisson/train.pkl', 'rb') as file:
        train_time = pickle.load(file)
        
    with open('../data/poisson/test.pkl', 'rb') as file:
        test_time = pickle.load(file)

    train_time = [torch.tensor(train_time[i], device=device) for i in range(len(train_time))]
    train_seq_lengths = [torch.tensor(len(times), device=device) for times in train_time]
    train_tmax = max([max(times) for times in train_time])
    train_time = nn.utils.rnn.pad_sequence(train_time, batch_first=True, padding_value=train_tmax+1)
    train_type = torch.zeros_like(train_time)
    train_mask = (train_time != train_tmax+1).float()

    test_time = [torch.tensor(test_time[i], device=device) for i in range(len(test_time))]

    # initialize / load model
    dim_eta = 1
    sde = NJDSDE(device, dim_eta, dim_hidden=32, num_hidden=2, batch_size=args.batch_size, sigma=0.01, activation=nn.Tanh()).to(device)
    eta0 = torch.randn(dim_eta, requires_grad=True, device=device)
    it0 = 0
    optimizer = optim.Adam([
            {'params': eta0, 'lr': 2e-2},
            {'params': sde.F.parameters(), 'lr': 1e-2, 'weight_decay': 1e-4},
            {'params': sde.G.parameters(), 'lr': 1e-2, 'weight_decay': 1e-4},
            {'params': sde.H.parameters(), 'lr': 1e-3, 'weight_decay': 1e-4},
        ]) 

    if args.restart:
        checkpoint = torch.load(args.paramr)
        sde.load_state_dict(checkpoint['sde_state_dict'])
        eta0 = checkpoint['eta0'].to(device)
        it0 = checkpoint['it0']
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    loss_meter = RunningAverageMeter()

    # maximize likelihood
    it = it0
    while it < args.niters:
        # clear out gradients for variables
        optimizer.zero_grad()

        # sample a mini-batch
        batch_id = np.random.choice(len(train_time), args.batch_size, replace=False)
        batch_train_time = torch.stack([train_time[seqid] for seqid in batch_id])
        batch_train_type = torch.stack([train_type[seqid] for seqid in batch_id])
        batch_train_mask = torch.stack([train_mask[seqid] for seqid in batch_id])

        # forward pass 
        eta_batch_l, eta_batch_r, loss = forward_pass(sde, eta0, batch_train_time, batch_train_type, batch_train_mask, device, batch_size=args.batch_size, dim_eta=dim_eta, num_divide=args.num_divide, args=args)

        loss_meter.update(loss.item() / args.batch_size)

        # backward prop
        loss.backward()
        print("iter: {}, current loss: {:10.4f}, running ave loss: {:10.4f}".format(it, loss.item()/args.batch_size, loss_meter.avg), flush=True)

        # step
        optimizer.step()

        it = it+1

        # validate 
        if it % args.nsave == 0:
            # save
            torch.save({'sde_state_dict': sde.state_dict(), 'eta0': eta0, 'it0': it, 'optimizer_state_dict': optimizer.state_dict()}, outpath + '/' + '{:05d}'.format(it) + args.paramw)

        # visualize
        check_list = list(range(0, 301, 50))
        if it in check_list:
            t, lmbda = recovery_intensity(sde, eta0, test_time[4], device, num_divide=args.num_divide)
            t = t.view(-1)
            lmbda = lmbda.view(-1)
            plot_poisson_recovery_intensity(t, lmbda, seq=test_time[4], xlabel='$t$', ylabel='$\lambda_t$', title='Poisson')
            plt.savefig(f"poisson_{it}.png")
