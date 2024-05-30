import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import pickle
from sklearn.metrics import accuracy_score, f1_score


# create the outdir
def create_outpath(dataset):
    path = os.getcwd()
    pid = os.getpid()

    wsppath = os.path.join(path, 'workspace')
    if not os.path.isdir(wsppath):
        os.mkdir(wsppath)

    outpath = os.path.join(wsppath, 'dataset:'+dataset + '-' + 'pid:'+str(pid))
    assert not os.path.isdir(outpath), 'output directory already exist (process id coincidentally the same), please retry'
    os.mkdir(outpath)

    return outpath


def forward_pass(sde, eta0, time_seqs, type_seqs, mask, device, batch_size, dim_eta, num_divide, args):

    padded_seq_length = len(time_seqs[0])
    sde.batch_size = batch_size

    eta_batch_l = torch.zeros(batch_size, padded_seq_length, dim_eta, device=device)
    # eta_batch_l[i,j,:] is the left-limit of \mathbf{eta} at the j-th event in the i-th sequence
    eta_batch_r = torch.zeros(batch_size, padded_seq_length, dim_eta, device=device)

    eta_time_l = torch.zeros(batch_size, padded_seq_length, device=device)
    # eta_time_l[i,j] == eta_batch_l[i,j,:][type_seqs[i,j]]
    eta_time_r = torch.zeros(batch_size, padded_seq_length, device=device)


    eta_batch_l[:, 0, :] = eta0.unsqueeze(0).repeat(batch_size,1)
    event_type = type_seqs[:, 0].tolist()
    eta_time_l[:, 0] = eta_batch_l[list(range(0, batch_size)), 0, event_type]

    eta_batch_r[:, 0, :] = eta_batch_l[:, 0, :] + sde.h(eta_batch_l[:, 0, :].clone())[list(range(0, batch_size)), :, event_type]
    eta_time_r[:, 0] = eta_batch_r[list(range(0, batch_size)), 0, event_type]
    
    tsave = torch.Tensor().to(device)
    eta_tsave = torch.Tensor().to(device)
    
    eta_initial = eta_batch_r[:, 0, :]

    for i in range(padded_seq_length-1):
        adjacent_events = time_seqs[:, i:i+2]
        ts, eta_ts_l = EulerSolver(sde, eta_initial, adjacent_events, num_divide, device)
        tsave = torch.cat((tsave, ts), dim=1)
        eta_tsave = torch.cat((eta_tsave, eta_ts_l), dim=2)

        eta_batch_l[:, i+1, :] = eta_ts_l[:, :, -1]

        eta_ts_r = eta_ts_l.clone()
        event_type = type_seqs[:, i+1].tolist()
        eta_ts_r[:, :, -1] = eta_ts_l[:, :, -1] + sde.h(eta_ts_l[:, :, -1])[list(range(0, batch_size)), :, event_type]

        eta_batch_r[:, i+1, :] = eta_ts_r[:, :, -1]

        eta_time_l[:, i+1] = eta_batch_l[list(range(0, batch_size)), i+1, event_type]
        eta_time_r[:, i+1] = eta_batch_r[list(range(0, batch_size)), i+1, event_type]

        eta_initial = eta_ts_r[:, :, -1]

    masked_eta_time_l = eta_time_l * mask
    sum_term = torch.sum(masked_eta_time_l)

    mask_without_first_col = mask[:, 1:]
    expanded_mask = mask_without_first_col.unsqueeze(2).repeat(1, 1, num_divide+1).view(mask.size(0), -1)
    expanded_mask = expanded_mask.unsqueeze(1).repeat(1,dim_eta,1)
    
    eta_tsave = eta_tsave * expanded_mask # mask the eta_tsave

    expanded_diff_tsave = torch.diff(tsave).unsqueeze(1).repeat(1, dim_eta, 1)
    
    integral_term = torch.sum((torch.exp(eta_tsave)[:, :, :-1] * expanded_mask[:, :, :-1] + torch.exp(eta_tsave)[:, :, 1:] * expanded_mask[:, :, 1:]) * (expanded_diff_tsave * expanded_mask[:, :, 1:])) / 2  # reason for mask: e^0=1

    log_likelihood = sum_term - integral_term
    loss = - log_likelihood
    
    return eta_batch_l, eta_batch_r, loss


def EulerSolver(sde, eta_initial, adjacent_events, num_divide, device):
    
    dt = torch.diff(adjacent_events, dim=1) / num_divide
    ts = torch.cat([adjacent_events[:, 0].unsqueeze(dim=1) + dt * j for j in range(num_divide+1)], dim=1)

    eta_ts = torch.Tensor().to(device)
    eta_ts = torch.cat((eta_ts, eta_initial.unsqueeze(2)), dim=2)

    for _ in range(num_divide):
        eta_output = eta_initial + sde.f(eta_initial.clone())*dt + sde.g(eta_initial.clone())*torch.sqrt(dt)*torch.randn_like(eta_initial).to(device)

        eta_ts = torch.cat((eta_ts, eta_output.unsqueeze(2)), dim=2)
        eta_initial = eta_output

    return ts, eta_ts


def next_predict(sde, eta0, time_seqs, type_seqs, device, dim_eta, num_divide, h=8, n_samples=1000, args=None):
    """
    Predict the time and type of the next event given historical events.
    """

    estimate_dt = []
    next_dt = []
    error_dt = []
    estimate_type = []
    next_type = []
    loss_list = []

    for idx, seq in enumerate(time_seqs):
        
        seq_time = seq.unsqueeze(0)
        seq_type = type_seqs[idx].unsqueeze(0)
        mask = torch.ones(1, len(seq), device=device)

        eta_seq_l, eta_seq_r, loss_idx = forward_pass(sde, eta0, seq_time, seq_type, mask, device, batch_size=1, dim_eta=dim_eta, num_divide=num_divide, args=args)
        
        dt_seq = torch.diff(seq)
        max_dt = torch.max(dt_seq)

        estimate_seq_dt = []
        next_seq_dt = dt_seq.tolist()
        error_seq_dt = []
        estimate_seq_type = []
        next_seq_type = type_seqs[idx][1:].tolist()

        for i in range(len(seq)-1):
            last_t = seq[i]
            n_dt = dt_seq[i]
            timestep = h * max_dt / n_samples
            tau = last_t + torch.linspace(0, h * max_dt, n_samples).to(device)
            d_tau = tau - last_t

            eta_last_t = eta_seq_r[:, i, :]

            adjacent_events = torch.tensor([last_t, last_t+h * max_dt], device=device).unsqueeze(0)

            _, eta_tau = EulerSolver(sde, eta_last_t, adjacent_events, n_samples-1, device)

            eta_tau = eta_tau.squeeze(0)

            intens_tau = torch.exp(eta_tau)
            intens_tau_sum = intens_tau.sum(dim=0)
            integral = torch.cumsum(timestep * intens_tau_sum, dim=0)
            # density for the time-until-next-event law
            density = intens_tau_sum * torch.exp(-integral) 
            
            d_tau_f_tau = d_tau * density
            # trapezoidal method
            e_dt = (timestep * 0.5 * (d_tau_f_tau[1:] + d_tau_f_tau[:-1])).sum()
            err_dt = torch.abs(e_dt - n_dt)
            e_type = torch.argmax(eta_seq_l[0][i+1])

            estimate_seq_dt.append(e_dt.item())
            error_seq_dt.append(err_dt.item())
            estimate_seq_type.append(e_type.item())

        loss_list.append(loss_idx)
        estimate_dt.extend(estimate_seq_dt)
        next_dt.extend(next_seq_dt)
        error_dt.extend(error_seq_dt)
        
        estimate_type.extend(estimate_seq_type)
        next_type.extend(next_seq_type)
        
    error_dt_tensor = torch.tensor(error_dt)
    RMSE = np.linalg.norm(error_dt_tensor.detach().numpy(), ord=2) / (len(error_dt_tensor) ** 0.5)
    acc = accuracy_score(next_type, estimate_type)
    f1 = f1_score(next_type, estimate_type, average='weighted') 
    loss = sum(loss_list)

    return loss, RMSE, acc, f1


def self_correcting_intensity(t, seq, mu, alpha):
    intensity_val = np.exp(mu * t - alpha*len(seq[(seq>0) & (seq < t)]))
    return intensity_val


def hawkes_intensity(t, seq, mu, alpha, beta):
    intensity_val = mu + alpha * np.sum(np.exp(-beta * (t - seq[seq < t])))
    return intensity_val


def poisson_intensity(t, seq, l):
    intensity_val = l
    return intensity_val


def recovery_intensity(sde, eta0, seq, device, num_divide):
    t = torch.Tensor().to(device)
    lmbda = torch.Tensor().to(device) 
    
    eta_initial = eta0.unsqueeze(0)
    for i in range(len(seq)-1):
        adjacent_events = seq[i:i+2].unsqueeze(0)
        sde.batch_size = 1
        ts, eta_ts_l = EulerSolver(sde, eta_initial, adjacent_events, num_divide, device)
        t = torch.cat((t, ts))
        lmbda = torch.cat((lmbda, torch.exp(eta_ts_l)))
        eta_ts_r = eta_ts_l.clone()
        eta_ts_r[0,0,-1] = eta_ts_l[0,0,-1] + sde.h(eta_ts_l[0,0,-1].unsqueeze(0))
        eta_initial = eta_ts_r[:,:,-1]

    return t, lmbda


def plot_poisson_recovery_intensity(t, lmbda, seq, xlabel, ylabel, title=''):
    t = t.to('cpu')
    t = t.detach().numpy()
    lmbda = lmbda.to('cpu')
    lmbda = lmbda.detach().numpy()
    seq = seq.to('cpu')
    seq = seq.detach().numpy()

    true_poisson = [poisson_intensity(time, seq, l=1) for time in t]
    plt.figure(figsize=(10, 4))

    plt.plot(t, lmbda, linewidth=2, label=f'NJDTPP', color='#FF6666')
    plt.plot(t, true_poisson, linewidth=2, label='ground truth', color='gray')
    plt.scatter(seq, np.ones_like(seq) * min(true_poisson) * 0.05, marker='o', color='#1f77b4', label='events', s=5)

    plt.title(title)
    plt.xlabel(xlabel, fontsize=13)
    plt.ylabel(ylabel, fontsize=13)
    plt.legend()
    plt.show()


def plot_hawkes_recovery_intensity(t, lmbda, seq, xlabel, ylabel, title=''):
    t = t.to('cpu')
    t = t.detach().numpy()
    lmbda = lmbda.to('cpu')
    lmbda = lmbda.detach().numpy()
    seq = seq.to('cpu')
    seq = seq.detach().numpy()

    true_hawkes = [hawkes_intensity(time, seq, mu = 0.2, alpha = 0.8, beta = 1.0) for time in t]
    plt.figure(figsize=(10, 4))

    plt.plot(t, lmbda, linewidth=2, label=f'NJDTPP', color='#FF6666')
    plt.plot(t, true_hawkes, linewidth=2, label='ground truth', color='gray')
    plt.scatter(seq, np.ones_like(seq) * min(true_hawkes) * 0.05, marker='x', color='#1f77b4', label='events', s=5)

    plt.title(title)
    plt.xlabel(xlabel, fontsize=13)
    plt.ylabel(ylabel, fontsize=13)
    plt.legend()
    plt.show()


def plot_self_correcting_recovery_intensity(t, lmbda, seq, xlabel, ylabel, title=''):
    t = t.to('cpu')
    t = t.detach().numpy()
    lmbda = lmbda.to('cpu')
    lmbda = lmbda.detach().numpy()
    seq = seq.to('cpu')
    seq = seq.detach().numpy()

    true_self_correcting = [self_correcting_intensity(time, seq, mu=0.5, alpha=0.2) for time in t]
    plt.figure(figsize=(10, 4))

    plt.plot(t, lmbda, linewidth=2, label=f'NJDTPP', color='#FF6666')
    plt.plot(t, true_self_correcting, linewidth=2, label='ground truth', color='gray')
    plt.scatter(seq, np.ones_like(seq) * min(true_self_correcting) * 0.05, marker='o', color='#1f77b4', label='events', s=5)

    plt.title(title)
    plt.xlabel(xlabel, fontsize=13)
    plt.ylabel(ylabel, fontsize=13)
    plt.legend()
    plt.show()


def load_pickle(file_dir):
    """Load from pickle file.

    Args:
        file_dir (BinaryIO): dir of the pickle file.

    Returns:
        any type: the loaded data.
    """
    with open(file_dir, 'rb') as file:
        try:
            data = pickle.load(file, encoding='latin-1')
        except Exception:
            data = pickle.load(file)

    return data


def build_input_from_pkl(device, source_dir: str, split: str):
    """
        Args:
            split (str, optional): denote the train, dev and test set. 
    """
    data = load_pickle(source_dir)

    num_event_types = data["dim_process"]
    source_data = data[split]
    time_seqs = [[float(x["time_since_start"]) for x in seq] for seq in source_data if seq]
    type_seqs = [[x["type_event"] for x in seq] for seq in source_data if seq]
    type_seqs = [torch.tensor(type_seqs[i], device=device) for i in range(len(type_seqs))]

    mins = [min(seq) for seq in time_seqs]
    time_seqs = [[round(time - min_val, 6) for time in time_seq] for time_seq, min_val in zip(time_seqs, mins)]
    time_seqs = [torch.tensor(time_seqs[i], device=device) for i in range(len(time_seqs))]
    
    seqs_lengths = [torch.tensor(len(seq), device=device) for seq in time_seqs]

    return time_seqs, type_seqs, num_event_types, seqs_lengths


def process_loaded_sequences(device, source_dir: str, split: str):
    """
    Preprocess the dataset by padding the sequences.
    """

    time_seqs, type_seqs, num_event_types, seqs_lengths = \
        build_input_from_pkl(device, source_dir, split)
    
    tmax = max([max(seq) for seq in time_seqs])

    #  Build a data tensor by padding
    time_seqs = nn.utils.rnn.pad_sequence(time_seqs, batch_first=True, padding_value=tmax+1)
    type_seqs = nn.utils.rnn.pad_sequence(type_seqs, batch_first=True, padding_value=0)
    mask = (time_seqs != tmax+1).float()

    return time_seqs, type_seqs, num_event_types, seqs_lengths, mask