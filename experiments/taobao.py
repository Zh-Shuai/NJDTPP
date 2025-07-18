import sys
sys.path.append('..')
import signal
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from modules import RunningAverageMeter, NJDSDE
from utils import create_outpath, process_loaded_sequences, forward_pass, next_predict


signal.signal(signal.SIGINT, lambda sig, frame: sys.exit(0))

parser = argparse.ArgumentParser('taobao')
parser.add_argument('--niters', type=int, default=2000)
parser.add_argument('--paramr', type=str, default='params.pth')
parser.add_argument('--paramw', type=str, default='params.pth')
parser.add_argument('--batch_size', type=int, default=30)
parser.add_argument('--nsave', type=int, default=500)
parser.set_defaults(restart=False, seed=True, debug=False)
parser.add_argument('--restart', dest='restart', action='store_true')
parser.add_argument('--seed', dest='seed', action='store_true')
parser.add_argument('--debug', dest='debug', action='store_true')
parser.add_argument('--num_divide', type=int, default=10, help='number of divided points in the time interval')
args = parser.parse_args()

outpath = create_outpath('taobao')


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # write all parameters to output file
    print(args, flush=True)

    # fix seeding for randomness
    if args.seed:
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)

    train_time, train_type, train_num_types, train_seq_lengths, train_mask = \
        process_loaded_sequences(device=device, source_dir='../data/taobao/train.pkl', split='train')
    dev_time, dev_type, dev_num_types, dev_seq_lengths, dev_mask = \
        process_loaded_sequences(device=device, source_dir='../data/taobao/dev.pkl', split='dev')
    test_time, test_type, test_num_types, test_seqs_lengths, test_mask = \
        process_loaded_sequences(device=device, source_dir='../data/taobao/test.pkl', split='test')
    num_test_event = torch.sum(test_mask).item()


    # The mean of the maximum inter-event time across all training sequences is used for the Integral Upper Limit Estimation in Eq.(21).
    intervals = torch.diff(train_time, dim=1)
    
    bool_mask = train_mask.bool()
    interval_mask = bool_mask[:, :-1] & bool_mask[:, 1:]
    
    masked_intervals = torch.where(interval_mask, intervals, torch.tensor(float('-inf'), device=device))
    row_max_intervals = torch.max(masked_intervals, dim=1)[0]
    valid_row_max = row_max_intervals[row_max_intervals != float('-inf')]
    mean_max_train_dt = torch.mean(valid_row_max).item()


    # initialize / load model
    dim_eta = train_num_types
    sde = NJDSDE(device, dim_eta=dim_eta, dim_hidden=32, num_hidden=2, batch_size=args.batch_size, sigma=0.01, activation=nn.Tanh()).to(device)
    eta0 = torch.empty(dim_eta, device=device).normal_(mean=0, std=0.1).requires_grad_()
    it0 = 0
    optimizer = optim.Adam([{'params': sde.parameters(),'lr': 1e-3, 'weight_decay':1e-5},
                            {'params': eta0, 'lr': 1e-2, 'weight_decay':1e-5},
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


    # test
    test_loss, RMSE, acc, f1 = next_predict(sde, eta0, test_time, test_type, test_mask, device, batch_size=args.batch_size, dim_eta=dim_eta, num_divide=args.num_divide, h=8, mean_max_train_dt=mean_max_train_dt, n_samples=1000, args=args)
    print("iter: {:5d}, nll: {:10.4f}, RMSE: {:10.4f}, acc: {:10.4f}, f1: {:10.4f}".format(it, test_loss/num_test_event, RMSE, acc, f1), flush=True)
