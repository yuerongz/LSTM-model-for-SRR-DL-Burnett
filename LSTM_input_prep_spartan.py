from RNN_data_extraction_spartan import fetch_event_info
import torch
import numpy as np


def normalise_by_1in500_peak(q_series):
    """
    Peak discharge in the 1in500 event was calculated to be 18470 m3/s
    Normalise Q series to [0,1] by adjusting based on 1in500 peak Q
    """
    pk_q = 18470.0
    q_series = q_series / pk_q
    return q_series


def split_for_validation(input_series, wl_set, set_no, seq_len=192):
    event_input_length = {'1971': 1719, '2010': 1119, '2013': 759, 'design': 959}
    event_info = fetch_event_info('/data/gpfs/projects/punim0728/Burnett_Proj/LSTM_data_extraction/event_summary.csv')
    # event_info = fetch_event_info('Spartan_files/event_summary.csv')
    evnet_names = list(event_info.keys())
    targeted_idxs = []
    curr_pt_idx = 0
    for idx in range(len(evnet_names)):
        next_pt_idx = curr_pt_idx + (event_input_length[event_info[evnet_names[idx]]['Hydro_pattern']] - seq_len + 1)
        if int(event_info[evnet_names[idx]]['CV_set_no']) == set_no:
            targeted_idxs.extend(list(range(curr_pt_idx, next_pt_idx)))
        curr_pt_idx = next_pt_idx
    input_series_val = np.take(input_series, targeted_idxs, axis=0)
    input_series = np.delete(input_series, targeted_idxs, axis=0)
    wl_set_val = np.take(wl_set, targeted_idxs, axis=0)
    wl_set = np.delete(wl_set, targeted_idxs, axis=0)
    return input_series, wl_set, input_series_val, wl_set_val


def prep_tensor_sets(pt_raw_data, seq_len=192, cv_set_no=0):
    """
    Prepare X, y data as data frame for RNN training.
    output:
    X.shape = (batch, 192, 2) for 24h long and 15min interval: Q and WL t-192:t-1
    y.shape = (batch, 1) for water level at t
    """
    # event_info = fetch_event_info('LSTM_6pt_tests/')     # recorded event info, for cross-validation split
    q_series = []
    wl_set = []
    sl_series = []
    cf_series = {}
    for col_name in pt_raw_data.keys():
        curr_seq = pt_raw_data[col_name]
        stacked_seq = [curr_seq[i:i+seq_len] for i in range(len(curr_seq)-seq_len+1)]   # len=192
        if  col_name.split('_')[-1] == 'Q':
            q_series.extend(stacked_seq)
        elif col_name.split('_')[-1] == 'WL':
            wl_set.extend([[elem[-1]] for elem in stacked_seq])     # 1-element series WL(t)
        elif col_name.split('_')[-1] == 'SL':
            sl_series.extend(stacked_seq)
        elif col_name.split('_')[-1] == 'CF':
            if col_name.split('_')[-2] in cf_series:
                cf_series[col_name.split('_')[-2]].extend(stacked_seq)
            else:
                cf_series[col_name.split('_')[-2]] = []
                cf_series[col_name.split('_')[-2]].extend(stacked_seq)
        else:
            raise KeyError(f"{col_name}: data column name not ending with Q, WL, SL or CF")
    # construct model input    # = (x, vector length, input_dim)
    q_series = np.array(q_series)   # size = (61952, 192)
    q_series = normalise_by_1in500_peak(q_series)
    model_input = np.empty((15, q_series.shape[0], q_series.shape[1]))
    model_input[0, :, :] = q_series
    wl_set = np.array(wl_set)   # size = (x, 1)
    sl_series = np.array(sl_series)
    sl_series = (sl_series + 1.5) / 3.6     # Normalise to (0, 1)
    model_input[1, :, :] = sl_series
    for cf_no in cf_series.keys():
        curr_cf_series = np.array(cf_series[cf_no])
        curr_cf_series = normalise_by_1in500_peak(curr_cf_series)
        model_input[int(cf_no)+2, :, :] = curr_cf_series
    model_input = np.moveaxis(model_input, 0, -1)
    # print(model_input.shape)  # check model input dimensions
    # Cross-validation split
    model_input, wl_set, model_input_val, wl_set_val = split_for_validation(model_input, wl_set, cv_set_no)
    return torch.from_numpy(model_input).view(-1, seq_len, 15), \
           torch.from_numpy(wl_set).view(-1, 1, 1), \
           torch.from_numpy(model_input_val).view(-1, seq_len, 15), \
           torch.from_numpy(wl_set_val).view(-1, 1, 1)
