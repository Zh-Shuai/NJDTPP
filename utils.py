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
    
    padded_seq_length = time_seqs.shape[1]

    eta_batch_l = torch.zeros(batch_size, padded_seq_length, dim_eta, device=device)
    # eta_batch_l[i,j,:] is the left-limit of \mathbf{eta} at the j-th event in the i-th sequence
    eta_batch_r = torch.zeros(batch_size, padded_seq_length, dim_eta, device=device)
    eta_time_l = torch.zeros(batch_size, padded_seq_length, device=device)
    # eta_time_l[i,j] == eta_batch_l[i,j,:][type_seqs[i,j]]
    eta_time_r = torch.zeros(batch_size, padded_seq_length, device=device)

    eta_batch_l[:, 0, :] = eta0.unsqueeze(0).repeat(batch_size, 1)
    
    # Use mask to only process valid first events
    valid_first_mask = mask[:, 0].bool()  # [batch_size]
    if valid_first_mask.any():
        valid_indices = torch.where(valid_first_mask)[0]
        num_valid = len(valid_indices)
        event_type = type_seqs[valid_indices, 0].tolist()
        
        # Only compute for valid sequences
        eta_batch_l_valid = eta_batch_l[valid_indices, 0, :]
        sde.batch_size = num_valid
        h_output = sde.h(eta_batch_l_valid.clone())  # [num_valid, dim_eta, dim_eta]
        
        batch_range = torch.arange(len(valid_indices), device=device)
        # Extract eta_time_l for valid sequences
        eta_time_l[valid_indices, 0] = eta_batch_l[valid_indices, 0, event_type]
        # Update eta_batch_r for valid sequences
        eta_batch_r[valid_indices, 0, :] = (
            eta_batch_l[valid_indices, 0, :] + 
            h_output[batch_range, :, event_type]
        )
        # Extract eta_time_r for valid sequences
        eta_time_r[valid_indices, 0] = eta_batch_r[valid_indices, 0, event_type]


    tsave = torch.empty(batch_size, 0, device=device)
    eta_tsave = torch.empty(batch_size, dim_eta, 0, device=device)
    eta_initial = eta_batch_r[:, 0, :]

    # Find the maximum actual sequence length in this batch
    max_seq_len = int(torch.max(torch.sum(mask, dim=1)).item())

    mask_bool = mask.bool()
    interval_mask = mask_bool[:, :-1] & mask_bool[:, 1:]  # [batch_size, padded_seq_length-1]

    # Process each event position
    for i in range(max_seq_len - 1):

        valid_indices = torch.where(interval_mask[:, i])[0]  # Indices of sequences with valid intervals at position i
        num_valid = len(valid_indices)
        
        # Extract valid sequences for current interval
        valid_adjacent_events = time_seqs[valid_indices, i:i+2]  # [num_valid, 2]
        valid_eta_initial = eta_initial[valid_indices]  # [num_valid, dim_eta]
        
        # Solve SDE only for valid sequences
        sde.batch_size = num_valid
        ts_valid, eta_ts_l_valid = EulerSolver(sde, valid_eta_initial, valid_adjacent_events, num_divide, device)
        # ts_valid: [num_valid, num_divide+1], eta_ts_l_valid: [num_valid, dim_eta, num_divide+1]
        
        # Pad ts and eta_ts to full batch size
        ts_full = torch.zeros(batch_size, ts_valid.shape[1], device=device)
        eta_ts_l_full = torch.zeros(batch_size, dim_eta, eta_ts_l_valid.shape[2], device=device)
        
        ts_full[valid_indices] = ts_valid
        eta_ts_l_full[valid_indices] = eta_ts_l_valid
        
        # Concatenate to saved trajectories
        tsave = torch.cat((tsave, ts_full), dim=1)
        eta_tsave = torch.cat((eta_tsave, eta_ts_l_full), dim=2)

        # Update eta_batch_l for valid sequences
        eta_batch_l[valid_indices, i+1, :] = eta_ts_l_valid[:, :, -1]

        # Compute jump for valid sequences
        eta_ts_r_valid = eta_ts_l_valid.clone()
        
        h_output_valid = sde.h(eta_ts_l_valid[:, :, -1])  # [num_valid, dim_eta, dim_eta]
        valid_batch_range = torch.arange(len(valid_indices), device=device)
        valid_event_types = type_seqs[valid_indices, i+1].tolist()
        
        eta_ts_r_valid[:, :, -1] = (
            eta_ts_l_valid[:, :, -1] + 
            h_output_valid[valid_batch_range, :, valid_event_types]
        )
        eta_batch_r[valid_indices, i+1, :] = eta_ts_r_valid[:, :, -1]

        eta_time_l[valid_indices, i+1] = eta_batch_l[valid_indices, i+1, valid_event_types]
        eta_time_r[valid_indices, i+1] = eta_batch_r[valid_indices, i+1, valid_event_types]
        
        # Update eta_initial for next iteration
        eta_initial[valid_indices] = eta_ts_r_valid[:, :, -1]

    # Compute loss with proper masking
    masked_eta_time_l = eta_time_l * mask
    sum_term = torch.sum(masked_eta_time_l)

    # Redefine mask to only include columns 1 to max_seq_len-1
    mask_redefined = mask[:, 1:max_seq_len]  # [batch_size, max_seq_len-1]

    # Expand mask for eta_tsave dimension
    expanded_mask = mask_redefined.unsqueeze(2).repeat(1, 1, num_divide+1).view(mask.size(0), -1)
    expanded_mask = expanded_mask.unsqueeze(1).repeat(1, dim_eta, 1)
    
    # Apply mask to eta_tsave
    eta_tsave_masked = eta_tsave * expanded_mask

    expanded_diff_tsave = torch.diff(tsave, dim=1).unsqueeze(1).repeat(1, dim_eta, 1)
    
    # Trapezoidal integration with masking
    integral_term = torch.sum(
        (torch.exp(eta_tsave_masked)[:, :, :-1] * expanded_mask[:, :, :-1] + 
         torch.exp(eta_tsave_masked)[:, :, 1:] * expanded_mask[:, :, 1:]) * 
        (expanded_diff_tsave * expanded_mask[:, :, 1:])
    ) / 2   # reason for mask: e^0=1

    log_likelihood = sum_term - integral_term
    loss = -log_likelihood
    
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



def next_predict(sde, eta0, time_seqs, type_seqs, mask, device, batch_size, dim_eta, num_divide, h, mean_max_train_dt, n_samples=1000, args=None):
    """
    Following THP and EasyTPP, we predict the next event time using historical events, and predict the next event type using historical events and the next event time.
    """
    
    # time_seqs and type_seqs are already padded tensors with shape [num_sequences, padded_seq_length]
    num_sequences, padded_seq_length = time_seqs.shape
    
    # all_estimate_dt = []
    # all_next_dt = []
    all_error_dt = []
    all_estimate_type = []
    all_next_type = []
    all_losses = []
    
    # Process sequences in batches
    for batch_start in range(0, num_sequences, batch_size):
        batch_end = min(batch_start + batch_size, num_sequences)
        current_batch_size = batch_end - batch_start
        
        batch_time_seqs = time_seqs[batch_start:batch_end]  # [current_batch_size, padded_seq_length]
        batch_type_seqs = type_seqs[batch_start:batch_end]  # [current_batch_size, padded_seq_length]
        batch_mask = mask[batch_start:batch_end]        # [current_batch_size, padded_seq_length]
        

        eta_batch_l, eta_batch_r, batch_loss = forward_pass(
            sde, eta0, batch_time_seqs, batch_type_seqs, batch_mask, 
            device, batch_size=current_batch_size, dim_eta=dim_eta, 
            num_divide=num_divide, args=args
        )
        
        all_losses.append(batch_loss)
        
        # Calculate time intervals for the batch
        dt_batch = torch.diff(batch_time_seqs, dim=1)  # [current_batch_size, padded_seq_length-1]
        
        # Find the maximum actual sequence length in this batch
        max_seq_len = int(torch.max(torch.sum(batch_mask, dim=1)).item())

        # Create mask for valid intervals (both current and next positions must be valid)
        batch_mask_bool = batch_mask.bool()
        interval_mask = batch_mask_bool[:, :-1] & batch_mask_bool[:, 1:]  # [current_batch_size, padded_seq_length-1]
        

        for i in range(max_seq_len - 1):
            # Get the mask for current position across all sequences in batch
            pos_mask = interval_mask[:, i]  # [current_batch_size]
            valid_seqs = torch.where(pos_mask)[0]  # Indices of sequences with valid intervals at position i
                
            # Extract data for valid sequences at position i
            last_t_batch = batch_time_seqs[valid_seqs, i]  # [num_valid_seqs]
            n_dt_batch = dt_batch[valid_seqs, i]           # [num_valid_seqs]
            eta_last_t_batch = eta_batch_r[valid_seqs, i, :]  # [num_valid_seqs, dim_eta]
            

            timestep = h * mean_max_train_dt / n_samples
            tau_batch = last_t_batch.unsqueeze(1) + torch.linspace(0, h * mean_max_train_dt, n_samples).to(device).unsqueeze(0)  # [num_valid_seqs, n_samples]
            d_tau_batch = tau_batch - last_t_batch.unsqueeze(1) 
            

            adjacent_events_batch = torch.stack([
                last_t_batch, 
                last_t_batch + h * mean_max_train_dt
            ], dim=1)  # [num_valid_seqs, 2]
            

            # EulerSolver can handle batch dimension
            sde.batch_size = len(valid_seqs)
            _, eta_tau_batch = EulerSolver(sde, eta_last_t_batch, adjacent_events_batch, n_samples-1, device)
            # eta_tau_batch: [num_valid_seqs, dim_eta, n_samples]
            
            intens_tau_batch = torch.exp(eta_tau_batch)  # [num_valid_seqs, dim_eta, n_samples]
            intens_tau_sum_batch = intens_tau_batch.sum(dim=1)  # [num_valid_seqs, n_samples]
            
            integral_batch = torch.cumsum(timestep * intens_tau_sum_batch, dim=1)  # [num_valid_seqs, n_samples]
            density_batch = intens_tau_sum_batch * torch.exp(-integral_batch)  # [num_valid_seqs, n_samples]
            

            d_tau_f_tau_batch = d_tau_batch * density_batch  # [num_valid_seqs, n_samples]
            # trapezoidal method
            e_dt_batch = (timestep * 0.5 * (d_tau_f_tau_batch[:, 1:] + d_tau_f_tau_batch[:, :-1])).sum(dim=1)  # [num_valid_seqs]
            err_dt_batch = torch.abs(e_dt_batch - n_dt_batch)
            

            e_type_batch = torch.argmax(eta_batch_l[valid_seqs, i+1], dim=1)  # [num_valid_seqs]
            
            # Store results
            # all_estimate_dt.extend(e_dt_batch.tolist())
            # all_next_dt.extend(n_dt_batch.tolist())
            all_error_dt.extend(err_dt_batch.tolist())
            all_estimate_type.extend(e_type_batch.tolist())
            all_next_type.extend(batch_type_seqs[valid_seqs, i+1].tolist())
    
    # Calculate metrics
    error_dt_tensor = torch.tensor(all_error_dt)
    RMSE = np.linalg.norm(error_dt_tensor.detach().cpu().numpy(), ord=2) / (len(error_dt_tensor) ** 0.5)
    
    acc = accuracy_score(all_next_type, all_estimate_type)
    f1 = f1_score(all_next_type, all_estimate_type, average='weighted')

    total_loss = sum(all_losses)
    
    return total_loss, RMSE, acc, f1



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