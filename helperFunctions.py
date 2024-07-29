import torch
import glob
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import welch, hann
from enums import CLASSES_WITH, CLASSES_WITHOUT


def create_sequences(data, SEQLEN, dataType = all):
    dataX = []
    labels = []
    for key in data.keys():
        if 'NoGNSS' in key and (dataType == 'noGNSS' or dataType == 'all'):
            for i in range(0,len(data[key]), SEQLEN):
                if len(data[key][i:i+SEQLEN]) >= SEQLEN:
                    labels.append(CLASSES_WITHOUT[key] - 1)
                    dataX.append(data[key][i:i+SEQLEN])
        if not 'NoGNSS' in key and (dataType == 'withGNSS' or dataType == 'all'):
            for i in range(0,len(data[key]), SEQLEN):
                if len(data[key][i:i+SEQLEN]) >= SEQLEN:
                    labels.append(CLASSES_WITH[key] - 1)
                    dataX.append(data[key][i:i+SEQLEN])
    return torch.tensor(np.array(dataX), dtype=torch.float32), torch.tensor(np.array(labels), dtype=torch.long)

def do_fft(col, window, BW, scaler, type=1):
    data = []
    if type == 2:
        data = torch.fft.fftshift(torch.abs(torch.fft.fft(torch.tensor(col), window)))
    else:
        win = hann(window, True)
        _, pxx= welch(col, BW, window=win, noverlap=window//2 , nfft=window, scaling='density', return_onesided=False)
        data = torch.fft.fftshift(10*torch.log10(torch.tensor(pxx)))

    data= scaler.fit_transform(data.reshape(-1,1)).flatten()
    return np.array(data)

def get_data_files(files_to_read):
    data_files = []
    data_files = glob.glob(files_to_read)
    print("Number of files:", len(data_files))
    return data_files

def read_data(files_to_read, nbr, window):
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_files = get_data_files(files_to_read)
    dataset = {}
    file_nbr = 0
    for dfile in data_files:
        fname = dfile.replace('dat', '').replace('DAT','').split('/')
        file_name = fname[-1]
        fname = fname[-2]
        if not 'Jammer' in fname:
            continue
        seek_samplingrate = int(file_name.split('-')[-2].replace('e6', ''))
        BW = seek_samplingrate*(1e6)
        bin_type = np.int16
        offset_ = 100
        # here the bin file of integer is read, and then it will be converted to Complex
        num_samples = int(BW*(1e-3)) * nbr
        raw_signal = np.fromfile(open(dfile), dtype=bin_type, offset=offset_, count=num_samples * 2)
        data = raw_signal.astype(np.float32).view(np.complex64)
        after_fft = []
        step_size= 4 *int(BW*(1e-3))
        file_nbr += 1
        for i in range(0, len(data), step_size):
            if len(data[i: i + step_size]) < step_size:
                break 
            after_fft.append(do_fft(data[i: i + step_size], window, BW, scaler))
        if fname in dataset.keys():
            dataset[fname] = np.concatenate((np.array(dataset[fname]), np.array(after_fft)), axis=0)
        else:
            dataset[fname] = np.array(after_fft)
    return dataset

def plot_heat_map(data_matrix, CLASS_NAMES):
    sns.set_theme(rc={'figure.figsize':(13,10)})
    x = np.array(data_matrix)
    x_scaled = x / np.sum(x, axis=1, keepdims=True)
    data = pd.DataFrame(x_scaled, columns= CLASS_NAMES, index=CLASS_NAMES)
    sns.heatmap(data, annot=True, cmap='Blues')
