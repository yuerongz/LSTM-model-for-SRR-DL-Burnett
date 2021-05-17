import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader, TensorDataset
from LSTM_input_prep import prep_tensor_sets
from data_extraction import fetch_raw_data
import os
import csv


def directory_checking(target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        print("Directory ", target_dir, " Created.")
    else:
        print("Directory ", target_dir, " already exists.")


def check_for_nodata(q_series_full, wl_set_full, nodata=-999):
    # CHECK FOR NO-DATA IN pt_raw_data when wl_nodata=-999
    filter_tensor = wl_set_full[:,0,0]
    q_series = q_series_full[filter_tensor!=nodata,:,:]
    wl_set = wl_set_full[filter_tensor!=nodata,:,:]
    return q_series, wl_set


class LSTM_net_q2wl(nn.Module):
    def __init__(self, input_dim, hidden_dim, lstm_dim, out_dim):
        super(LSTM_net_q2wl, self).__init__()   # define weights calculation functions
        self.lstm_dim = lstm_dim
        self.hidden1 = nn.Linear(input_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, lstm_dim, batch_first=True)  # use the first dimension as batch_no
        self.hidden2 = nn.Linear(lstm_dim, out_dim)

    def forward(self, input_series):    # define activate functions here
        lstm_in = self.hidden1(input_series)    # input_series: tensor(batch, seq_len=192, input_size=1)
        _, (lstm_out, _) = self.lstm(torch.relu(lstm_in))
        water_level = self.hidden2(torch.tanh(lstm_out.view(-1, 1, self.lstm_dim)))
        return water_level


def pytorch_cuda_check():
    if torch.cuda.is_available():
        device = torch.device("cuda")  # you can continue going on here, like cuda:1 cuda:2....etc.
        print("Running on the GPU...")
    else:
        device = torch.device("cpu")
        print("Running on the CPU...")
    return device


def pytorch_model_training(testing_structures, test_str_no, device, epoch_len,
                           train_loader, test_loader,
                           work_dir, pt_no):
    losses = []
    val_losses = []
    model = LSTM_net_q2wl(15, testing_structures[test_str_no][0], testing_structures[test_str_no][1], 1).to(device)
    model.float()
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    # train the model
    for epoch in range(epoch_len):
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            model.train()  # set model to TRAIN mode
            optimizer.zero_grad()
            wl_t = model(x_batch.float())
            wl_target = y_batch.float()
            loss = loss_function(wl_t, wl_target)
            losses.append(loss)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            for x_test, y_test in test_loader:
                x_test, y_test = x_test.to(device), y_test.to(device)
                model.eval()  # set model to TEST mode
                y_hat = model(x_test.float())
                val_loss = loss_function(y_hat, y_test.float())
                val_losses.append(val_loss.item())

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'loss_his': losses,
        'val_loss_his': val_losses
    }, f"{work_dir}grid_{pt_no}_model.pt")

    return model, losses, val_losses


def load_saved_lstm_model(work_dir, testing_structures, pt_no, seed_no, device):
    model = LSTM_net_q2wl(15, testing_structures[0][0], testing_structures[0][1], 1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    checkpoint = torch.load(f"{work_dir}Training/Seed_{seed_no}/grid_{pt_no}_model.pt")
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    losses = checkpoint['loss_his']
    val_losses = checkpoint['val_loss_his']
    model.eval()
    return model


if __name__ == '__main__':
    data_dir = 'directory-to-raw-data/trial_2/'
    seq_len = 192
    epoch_len = 40
    testing_structures = [[10, 20]]
    seed_no = spartan_seed_no
    work_dir = f"directory-to-working-folder/Seed_{seed_no}/"
    directory_checking(work_dir)
    device = pytorch_cuda_check()
    pt_no = spartan_grid_no
    raw_data = fetch_raw_data(data_dir, pt_no)

    for pt_coords in list(raw_data.keys()):
        cross_vali_set = 0  # leave-4-out validation

        # prepare train and test datasets
        q_series_full, wl_set_full, q_series_val_full, wl_set_val_full = prep_tensor_sets(raw_data[pt_coords],
                                                                                          cv_set_no=cross_vali_set)
        q_series, wl_set = check_for_nodata(q_series_full, wl_set_full)
        q_series_val, wl_set_val = check_for_nodata(q_series_val_full, wl_set_val_full)
        data_len = q_series.shape[0]
        all_data = TensorDataset(q_series, wl_set)

        torch.manual_seed(seed_no)
        train_dataset, test_dataset = random_split(all_data, [round(0.8 * data_len), data_len - round(0.8 * data_len)])

        train_loader = DataLoader(dataset=train_dataset, batch_size=200, shuffle=True)
        test_loader = DataLoader(dataset=test_dataset, batch_size=200, shuffle=True)

        for test_str_no in range(len(testing_structures)):
            model, losses, val_losses = pytorch_model_training(testing_structures, test_str_no, device, epoch_len,
                                                               train_loader, test_loader,
                                                               work_dir, pt_no)
        
        
