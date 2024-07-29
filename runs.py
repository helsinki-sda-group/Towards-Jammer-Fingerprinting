#%%
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from GenerealLSTM import LSTMTrad, validate_traditional, train_traditional
from BayesianNetwork import SimpleBayesianNetwork, train, validate
from helperFunctions import read_data, create_sequences, plot_heat_map
from enums import CLASS_NAMES

torch.manual_seed(42)
np.random.seed(42)

WINDOW = 1024
## Change this to correspond your data
parent = input("Give your parent directory: ") 
files = input("Type your folder with with an empty space between: ")
files = files.split(' ')
data_chamber1= read_data(f'{parent}{files[0]}', 100, WINDOW)
data_chamber2 = read_data(f'{parent}{files[1]}', 100, WINDOW)
data_chamber3= read_data(f'{parent}{files[2]}',100, WINDOW)
data_chamber4 = read_data(f'{parent}{files[3]}', 100, WINDOW)
data_norway1 = read_data(f'{parent}{files[4]}',100, WINDOW)
data_norway2 = read_data(f'{parent}{files[5]}',100, WINDOW)

all_dicts = [data_norway1, data_chamber1, data_chamber2, data_chamber3, data_chamber4, data_norway2]
keys_to_loop1 = list(data_norway1.keys()) + list(data_chamber1.keys()) + list(data_chamber2.keys())
keys_to_loop1 = list(set(keys_to_loop1))
print(keys_to_loop1)
#%%
data = {}
for key in keys_to_loop1:
    for d in all_dicts:
        if key in data.keys() and key in d.keys(): 
            data[key] = np.concatenate((np.array(data[key]), np.array(d[key])), axis=0)
        elif key in d.keys():
            data[key] = np.array(d[key])
    print(data[key].shape)
data_test = {}
print(data)
X, y = create_sequences(data, 4, 'all')
print(X)
plt.plot(X[0][0])
# %%
X_train_all, X_test, y_train_all, y_test = train_test_split(X,
                                                    y,
                                                    test_size=.20,
                                                    random_state=42,
                                                    shuffle=True)

X_train, X_val, y_train, y_val = train_test_split(X_train_all,
                                                    y_train_all,
                                                    test_size=.05,
                                                    random_state=42,
                                                    shuffle=True)

print(y_train.size())
print(y_test.size())
print(y_val.size())

#%%
ds = torch.utils.data.TensorDataset(X_train, y_train)
ds_val = torch.utils.data.TensorDataset(X_val, y_val)
ds_test = torch.utils.data.TensorDataset(X_test, y_test)

dataloader_train = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=True)
dataloader_val = torch.utils.data.DataLoader(ds_val, batch_size=64, shuffle=True)
dataloader_test = torch.utils.data.DataLoader(ds_test, batch_size=1, shuffle=False)

jammers_nbr = 21
model = SimpleBayesianNetwork(WINDOW, jammers_nbr)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 50
best_model = train(dataloader_train, dataloader_val, model, criterion, optimizer, epochs, 'my_BLSTM.pt')
validate(dataloader_test, best_model)
NUM_SAMPLE_INFERENCE = len(X_test)
NUM_INFERENCES = 10

best_model.eval()
softmax_predictions = torch.stack([(best_model(X_test.to(torch.float)))
                                   for _ in range(NUM_INFERENCES)], axis=0)

#%%
jammers_mode = np.zeros((21,21))
jammers_all = np.zeros((21,21))
for x in range(NUM_SAMPLE_INFERENCE):
    class_preds = []
    for ind in range(NUM_INFERENCES):
        prediction_this_inference = torch.argmax((softmax_predictions[ind][x]))
        class_preds.append(prediction_this_inference)
    class_preds = torch.tensor(class_preds)

    for i in class_preds:
        jammers_all[y_test[x]][i] += 1
    jammers_mode[y_test[x]][torch.mode(class_preds)[0]] += 1

plot_heat_map(jammers_mode, CLASS_NAMES)
#%%
model2 = LSTMTrad(WINDOW, jammers_nbr)
optimizer = optim.Adam(model2.parameters(), lr=0.001, weight_decay=0.000005)
best_model = train_traditional(epochs, model2, dataloader_train, dataloader_val, criterion, optimizer, 'arfidaas_TLSTM.pt')
predicted_trad = validate_traditional(dataloader_test, best_model)
jammers_all2 = np.zeros((21,21))

for x in range(len(y_test)):
    jammers_all2[y_test[x].item()][predicted_trad[x].item()] += 1

plot_heat_map(jammers_all2, CLASS_NAMES)
# %%
